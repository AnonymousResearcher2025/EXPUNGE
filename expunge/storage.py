"""Transactional on-disk graph, insertion versions, and cell postings."""

from __future__ import annotations
import json
import hashlib
from bisect import bisect_left, insort
import sqlite3
from pathlib import Path
import numpy as np
from .native import EMPTY, Node


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


class Store:
    def __init__(self, path, create=False):
        self.path = Path(path)
        if create and self.path.exists():
            raise FileExistsError(self.path)
        if not create and not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA secure_delete=ON")
        if create:
            self.db.executescript("""
            CREATE TABLE meta(key TEXT PRIMARY KEY, value BLOB NOT NULL);
            CREATE TABLE vectors(id INTEGER PRIMARY KEY, cell INTEGER NOT NULL, value BLOB, inserted INTEGER NOT NULL);
            CREATE INDEX vectors_cell ON vectors(cell, id);
            CREATE TABLE graph(id INTEGER PRIMARY KEY, state BLOB NOT NULL, level INTEGER NOT NULL);
            CREATE INDEX graph_level ON graph(level DESC,id);
            CREATE TABLE checkpoint(id INTEGER PRIMARY KEY, state BLOB NOT NULL, level INTEGER NOT NULL);
            CREATE TABLE operations(seq INTEGER PRIMARY KEY, vertex INTEGER UNIQUE NOT NULL, before_meta BLOB NOT NULL,
                after_meta BLOB NOT NULL, bitmap BLOB NOT NULL, scored BLOB NOT NULL, candidates BLOB NOT NULL, cpu REAL NOT NULL);
            CREATE TABLE versions(owner INTEGER NOT NULL, seq INTEGER NOT NULL REFERENCES operations(seq) ON DELETE CASCADE,
                before_state BLOB NOT NULL, after_state BLOB NOT NULL, PRIMARY KEY(owner,seq));
            CREATE INDEX versions_seq ON versions(seq,owner);
            CREATE TABLE postings(cell INTEGER NOT NULL, seq INTEGER NOT NULL REFERENCES operations(seq) ON DELETE CASCADE,
                PRIMARY KEY(cell,seq));
            CREATE TABLE edges(owner INTEGER NOT NULL, level INTEGER NOT NULL, neighbor INTEGER NOT NULL,
                PRIMARY KEY(owner,level,neighbor));
            CREATE INDEX edges_incoming ON edges(neighbor,owner);
            CREATE TABLE removed(seq INTEGER PRIMARY KEY);
            CREATE TABLE merkle(level INTEGER NOT NULL, position INTEGER NOT NULL, hash BLOB NOT NULL, PRIMARY KEY(level,position));
            CREATE TABLE maintenance(seq INTEGER PRIMARY KEY, record BLOB NOT NULL, signature BLOB NOT NULL);
            """)
            self.db.commit()
        self.reload_removed()

    def reload_removed(self):
        self.removed = [
            seq for (seq,) in self.db.execute("SELECT seq FROM removed ORDER BY seq")
        ]

    def mark_removed(self, seq):
        self.db.execute("INSERT INTO removed VALUES (?)", (seq,))
        insort(self.removed, seq)

    def put(self, key, value):
        self.db.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?)", (key, canonical(value))
        )

    def get(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def vector(self, vertex):
        row = self.db.execute(
            "SELECT value FROM vectors WHERE id=?", (vertex,)
        ).fetchone()
        if row is None or row[0] is None:
            raise KeyError(f"No live payload for vertex {vertex}")
        return np.frombuffer(row[0], dtype="<f4")

    def node(self, vertex):
        row = self.db.execute(
            "SELECT state FROM graph WHERE id=?", (vertex,)
        ).fetchone()
        return Node.unpack(row[0] if row else EMPTY)

    def state_at(self, vertex, prefix, skipped, replacements):
        best_seq, state = -1, EMPTY
        local = replacements.get(vertex, ())
        for seq, after in reversed(local):
            if seq <= prefix:
                best_seq, state = seq, after
                break
        for seq, after in self.db.execute(
            "SELECT seq,after_state FROM versions WHERE owner=? AND seq<=? ORDER BY seq DESC",
            (vertex, prefix),
        ):
            if seq <= best_seq:
                break
            if seq not in skipped:
                return Node.unpack(after)
        if best_seq >= 0:
            return Node.unpack(state)
        row = self.db.execute(
            "SELECT state FROM checkpoint WHERE id=?", (vertex,)
        ).fetchone()
        return Node.unpack(row[0] if row else EMPTY)

    def graph_write(self, vertex, node):
        if node.active:
            self.db.execute(
                "INSERT OR REPLACE INTO graph VALUES (?,?,?)",
                (vertex, node.pack(), node.level),
            )
        else:
            self.db.execute("DELETE FROM graph WHERE id=?", (vertex,))
        self.db.execute("DELETE FROM edges WHERE owner=?", (vertex,))
        self.db.executemany(
            "INSERT INTO edges VALUES (?,?,?)",
            (
                (vertex, level, other)
                for level, ids in enumerate(node.edges)
                for other in ids
            ),
        )
        capacity = self.get("shape")[0]
        height = (capacity - 1).bit_length()
        empty = hashlib.sha256(b"EXPUNGE-empty-v1").digest()
        value = (
            hashlib.sha256(
                b"EXPUNGE-vertex-v1"
                + int(vertex).to_bytes(4, "little")
                + node.pack()
                + self.vector(vertex).tobytes()
            ).digest()
            if node.active
            else empty
        )
        position = vertex
        for level in range(height + 1):
            if value == empty:
                self.db.execute(
                    "DELETE FROM merkle WHERE level=? AND position=?", (level, position)
                )
            else:
                self.db.execute(
                    "INSERT OR REPLACE INTO merkle VALUES (?,?,?)",
                    (level, position, value),
                )
            if level < height:
                sibling = self.db.execute(
                    "SELECT hash FROM merkle WHERE level=? AND position=?",
                    (level, position ^ 1),
                ).fetchone()
                sibling = sibling[0] if sibling else empty
                left, right = (sibling, value) if position & 1 else (value, sibling)
                value = hashlib.sha256(left + right).digest()
                empty = hashlib.sha256(empty + empty).digest()
                position //= 2

    def root_digest(self):
        height = (self.get("shape")[0] - 1).bit_length()
        row = self.db.execute(
            "SELECT hash FROM merkle WHERE level=? AND position=0", (height,)
        ).fetchone()
        if row:
            return row[0]
        empty = hashlib.sha256(b"EXPUNGE-empty-v1").digest()
        for _ in range(height):
            empty = hashlib.sha256(empty + empty).digest()
        return empty

    def verify_root(self):
        capacity = self.get("shape")[0]
        height = (capacity - 1).bit_length()
        empty = hashlib.sha256(b"EXPUNGE-empty-v1").digest()
        leaves = {}
        for vertex, raw, level, value in self.db.execute(
            "SELECT graph.id,state,level,value FROM graph JOIN vectors ON graph.id=vectors.id"
        ):
            if Node.unpack(raw).level != level or value is None:
                raise AssertionError("Invalid persistent node or payload")
            leaves[vertex] = hashlib.sha256(
                b"EXPUNGE-vertex-v1" + int(vertex).to_bytes(4, "little") + raw + value
            ).digest()
        for _ in range(height):
            parents = {}
            for position in {p // 2 for p in leaves}:
                parents[position] = hashlib.sha256(
                    leaves.get(2 * position, empty)
                    + leaves.get(2 * position + 1, empty)
                ).digest()
            leaves = parents
            empty = hashlib.sha256(empty + empty).digest()
        if leaves.get(0, empty) != self.root_digest():
            raise AssertionError("Graph/payload digest verification failed")

    def operation(self, seq):
        row = self.db.execute(
            "SELECT vertex,before_meta,after_meta,bitmap,scored,candidates,cpu FROM operations WHERE seq=?",
            (seq,),
        ).fetchone()
        if row is None:
            raise KeyError(seq)
        vertex, before, after, bitmap, scored, candidates, cpu = row
        before, after = list(json.loads(before)), list(json.loads(after))
        checkpoint = self.get("checkpoint_metadata")
        before[2] = checkpoint[2] + seq - bisect_left(self.removed, seq)
        after[2] = before[2] + 1
        before[3] = after[3] = checkpoint[3]
        pairs = np.frombuffer(
            scored, dtype=np.dtype([("id", "<u4"), ("distance", "<f4")])
        )
        return {
            "seq": seq,
            "vertex": vertex,
            "before": tuple(before),
            "after": tuple(after),
            "bitmap": bitmap,
            "scored": dict(zip(map(int, pairs["id"]), map(float, pairs["distance"]))),
            "candidates": candidates,
            "cpu": cpu,
        }

    def writes(self, seq):
        return {
            int(owner): (Node.unpack(before), Node.unpack(after))
            for owner, before, after in self.db.execute(
                "SELECT owner,before_state,after_state FROM versions WHERE seq=?",
                (seq,),
            )
        }

    def log(self, operation, writes, cells):
        seq = operation["seq"]
        scored = np.asarray(
            sorted(operation["scored"].items()),
            dtype=np.dtype([("id", "<u4"), ("distance", "<f4")]),
        ).tobytes()
        self.db.execute("DELETE FROM operations WHERE seq=?", (seq,))
        self.db.execute(
            "INSERT INTO operations VALUES (?,?,?,?,?,?,?,?)",
            (
                seq,
                operation["vertex"],
                canonical(operation["before"]),
                canonical(operation["after"]),
                operation["bitmap"],
                scored,
                operation["candidates"],
                operation["cpu"],
            ),
        )
        self.db.executemany(
            "INSERT INTO versions VALUES (?,?,?,?)",
            (
                (owner, seq, before.pack(), after.pack())
                for owner, (before, after) in writes.items()
            ),
        )
        self.db.executemany(
            "INSERT INTO postings VALUES (?,?)",
            ((cell, seq) for cell in sorted(set(cells))),
        )

    def checkpoint(self, metadata):
        self.db.execute("DELETE FROM checkpoint")
        self.db.execute("INSERT INTO checkpoint SELECT * FROM graph")
        self.db.execute("DELETE FROM operations")
        self.db.execute("DELETE FROM removed")
        self.removed = []
        self.put("next_seq", 0)
        self.put("checkpoint_metadata", metadata)
        self.put("epoch", self.get("epoch") + 1)

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()
