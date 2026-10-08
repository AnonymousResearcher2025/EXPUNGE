"""Insertion logging, counterfactual replay, canonical scrub, and work governance."""

from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import math
import heapq
import hmac
import os
from pathlib import Path
import threading
import time
import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .data import Partition
from .native import NONE, Graph, Node
from .storage import Store, canonical

DEFAULTS = dict(
    backend="hnsw",
    degree=32,
    beam=400,
    alpha=1.2,
    metric="l2",
    seed=42,
    cells=2048,
    partition_sample=65536,
    hops=2,
    query_width=100,
    cpu_fraction=0.10,
    cores=os.cpu_count() or 1,
    probe=4096,
    scrub_key="expunge-canonical-order-v1",
    inplace_width=128,
    inplace_candidates=50,
    inplace_copies=3,
    spatch_r=1.0,
    spatch_alpha=1.2,
    warm_replay=True,
)


class ReplayAborted(RuntimeError):
    pass


class ReadWriteLock:
    """Writer preference gives graph commits a bounded queue of existing readers."""

    def __init__(self):
        self.condition = threading.Condition()
        self.readers = self.waiting = 0
        self.writing = False

    @contextmanager
    def read(self):
        with self.condition:
            while self.writing or self.waiting:
                self.condition.wait()
            self.readers += 1
        try:
            yield
        finally:
            with self.condition:
                self.readers -= 1
                self.condition.notify_all()

    def __enter__(self):
        with self.condition:
            self.waiting += 1
            while self.writing or self.readers:
                self.condition.wait()
            self.waiting -= 1
            self.writing = True
        return self

    def __exit__(self, *_):
        with self.condition:
            self.writing = False
            self.condition.notify_all()


class Trace:
    def __init__(self, index, graph, vertex, ensure=None):
        self.index, self.graph, self.vertex, self.ensure = index, graph, vertex, ensure
        self.before = {}
        self.cells = set()
        self.preparing = True

    def read(self, owner):
        if self.ensure and not (owner == self.vertex and not self.preparing):
            self.ensure(owner)
        if owner not in self.before:
            self.before[owner] = self.graph.get(owner)
        node = self.before[owner] if owner == self.vertex else self.graph.get(owner)
        self.cells.add(int(self.index.assignments[owner]))
        for layer in node.edges:
            self.cells.update(int(self.index.assignments[other]) for other in layer)
        if -1 in self.cells:
            raise RuntimeError("Construction read an unassigned stable ID")

    def execute(self, seq, warm=None):
        before_meta = self.graph.metadata
        self.read(self.vertex)
        self.preparing = False
        self.graph.on_read(self.read)
        start = time.thread_time()
        try:
            self.graph.insert(self.vertex, self.index.store.vector(self.vertex), warm)
        finally:
            self.graph.on_read(None)
        cpu = time.thread_time() - start
        writes = {}
        for owner, before in self.before.items():
            after = self.graph.get(owner)
            if before != after:
                writes[owner] = (before, after)
        operation = dict(
            seq=seq,
            vertex=self.vertex,
            before=before_meta,
            after=self.graph.metadata,
            bitmap=self.index.partition.bitmap(self.cells),
            scored=self.graph.scored(),
            candidates=self.graph.candidates(),
            cpu=cpu,
        )
        return operation, writes


class Index:
    def __init__(self, path, _building=False):
        self.store = Store(path)
        if not _building and not self.store.get("build_complete"):
            self.store.close()
            raise RuntimeError("Index construction did not finish")
        if self.store.get("engine_binary") != Graph.library_fingerprint():
            self.store.close()
            raise RuntimeError(
                "Native binary differs from the index builder; rebuild the index with this binary"
            )
        self.config = self.store.get("config")
        self.capacity, self.dimension = self.store.get("shape")
        centers = np.asarray(self.store.get("centers"), np.float32)
        self.partition = Partition(centers, self.config["metric"])
        self.assignments = np.full(self.capacity, -1, np.int32)
        for vertex, cell in self.store.db.execute("SELECT id,cell FROM vectors"):
            self.assignments[vertex] = cell
        self.graph = self._load_graph()
        self._lock, self._mutation = ReadWriteLock(), threading.Lock()
        self._closed = False
        self.tokens = float(self.config["cpu_fraction"] * self.config["cores"])
        self.last_refill = time.monotonic()
        key_path = Path(str(self.store.path) + ".key")
        if not key_path.is_file():
            raise FileNotFoundError(f"Maintenance signing key missing: {key_path}")
        self.key = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
        public = (
            self.key.public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            .hex()
        )
        if public != self.store.get("public_key"):
            raise ValueError("Signing key does not match the index public key")

    @classmethod
    def build(
        cls, path, values, config=None, capacity=None, order=None, partition=None
    ):
        fingerprint = Graph.library_fingerprint()
        cfg = dict(DEFAULTS, **(config or {}))
        if cfg["backend"] not in ("hnsw", "vamana") or cfg["metric"] not in (
            "l2",
            "cosine",
        ):
            raise ValueError("Unknown backend or metric")
        if (
            not 0 < cfg["cpu_fraction"] <= 1
            or cfg["cores"] < 1
            or cfg["hops"] < 0
            or cfg["probe"] < 1
        ):
            raise ValueError("Invalid governor or region configuration")
        integers = (
            "degree",
            "beam",
            "cells",
            "partition_sample",
            "seed",
            "hops",
            "cores",
            "probe",
            "query_width",
            "inplace_width",
            "inplace_candidates",
            "inplace_copies",
        )
        if any(not isinstance(cfg[k], int) or cfg[k] < 0 for k in integers):
            raise ValueError(
                "Integer configuration parameters must be nonnegative integers"
            )
        if (
            cfg["degree"] < 2
            or cfg["beam"] < cfg["degree"]
            or cfg["seed"] >= 2**32
            or cfg["query_width"] < 1
        ):
            raise ValueError("Invalid construction or query parameters")
        if any(
            cfg[k] < 1
            for k in ("inplace_width", "inplace_candidates", "inplace_copies")
        ):
            raise ValueError("In-place repair parameters must be positive")
        if (
            not math.isfinite(cfg["alpha"])
            or cfg["alpha"] < 1
            or not math.isfinite(cfg["spatch_r"])
            or cfg["spatch_r"] <= 0
            or not math.isfinite(cfg["spatch_alpha"])
            or cfg["spatch_alpha"] <= 0
        ):
            raise ValueError("Invalid pruning or SPatch parameters")
        if values.ndim != 2:
            raise ValueError("Expected a vector matrix")
        n, dimension = values.shape
        capacity = n if capacity is None else capacity
        if not 0 < capacity < NONE or n > capacity or not 0 < dimension < NONE:
            raise ValueError("Invalid capacity")
        ids = list(range(n)) if order is None else list(map(int, order))
        if len(set(ids)) != len(ids) or any(v < 0 or v >= n for v in ids):
            raise ValueError("Order must contain unique valid stable IDs")
        partition = partition or Partition.fit(
            values, cfg["cells"], cfg["partition_sample"], cfg["seed"], cfg["metric"]
        )
        if (
            len(partition.centers) != cfg["cells"]
            or partition.centers.shape[1] != dimension
            or partition.metric != cfg["metric"]
        ):
            raise ValueError("Frozen partition differs from configured cell count")
        assignments = partition.assign(values)
        key_path = Path(str(path) + ".key")
        if key_path.exists():
            raise FileExistsError(key_path)
        store = Store(path, create=True)
        try:
            key = Ed25519PrivateKey.generate()
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(
                    key.private_bytes(
                        serialization.Encoding.Raw,
                        serialization.PrivateFormat.Raw,
                        serialization.NoEncryption(),
                    )
                )
            with store.db:
                store.put("config", cfg)
                store.put("engine_binary", fingerprint)
                store.put("shape", [capacity, dimension])
                store.put("centers", partition.centers.tolist())
                store.put("global", [NONE, -1, 0, 0])
                store.put("checkpoint_metadata", [NONE, -1, 0, 0])
                store.put("next_seq", len(ids))
                store.put("build_complete", False)
                store.put("epoch", 0)
                store.put(
                    "costs",
                    {
                        "insert_cpu": 0,
                        "replay_cpu": 0,
                        "insert_count": 0,
                        "replay_count": 0,
                    },
                )
                store.put(
                    "public_key",
                    key.public_key()
                    .public_bytes(
                        serialization.Encoding.Raw, serialization.PublicFormat.Raw
                    )
                    .hex(),
                )
                for position, vertex in enumerate(ids):
                    vector = np.asarray(values[vertex], dtype="<f4")
                    if not np.isfinite(vector).all():
                        raise ValueError("Nonfinite input vector")
                    store.db.execute(
                        "INSERT INTO vectors VALUES (?,?,?,?)",
                        (vertex, int(assignments[vertex]), vector.tobytes(), position),
                    )
        except BaseException:
            store.close()
            for suffix in ("", "-wal", "-shm", ".key"):
                Path(str(path) + suffix).unlink(missing_ok=True)
            raise
        store.close()
        index = cls(path, _building=True)
        try:
            with index.store.db:
                total_cpu = 0.0
                for seq, vertex in enumerate(ids):
                    trace = Trace(index, index.graph, vertex)
                    op, writes = trace.execute(seq)
                    index.store.log(op, writes, trace.cells)
                    for owner, (_, node) in writes.items():
                        index.store.graph_write(owner, node)
                    total_cpu += op["cpu"]
                index.store.put("global", index.graph.metadata)
                index.store.put(
                    "costs",
                    dict(
                        insert_cpu=total_cpu,
                        replay_cpu=0,
                        insert_count=len(ids),
                        replay_count=0,
                    ),
                )
                index.store.put("build_complete", True)
        except BaseException:
            index.close()
            for suffix in ("", "-wal", "-shm", ".key"):
                Path(str(path) + suffix).unlink(missing_ok=True)
            raise
        return index

    def _load_graph(self):
        graph = Graph(self.capacity, self.dimension, self.config)
        for vertex, raw in self.store.db.execute(
            "SELECT id,state FROM graph ORDER BY id"
        ):
            graph.set(vertex, Node.unpack(raw), self.store.vector(vertex))
        graph.metadata = self.store.get("global")
        return graph

    def close(self):
        with self._mutation, self._lock:
            if not self._closed:
                self._closed = True
                self.graph.close()
                self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def query(self, vector, k=10, width=None):
        with self._lock.read():
            if self._closed:
                raise RuntimeError("Index is closed")
            return self.graph.query(
                vector, k, self.config["query_width"] if width is None else width
            )

    def insert(self, vertex, vector):
        with self._mutation, self._lock:
            if vertex < 0 or vertex >= self.capacity or self.assignments[vertex] >= 0:
                raise ValueError(
                    "Insertion needs an unused stable ID within reserved capacity"
                )
            vector = self.graph.vector(vector)
            cell = int(self.partition.assign(vector)[0])
            seq = self.store.get("next_seq")
            position = self.store.db.execute(
                "SELECT COALESCE(MAX(inserted),-1)+1 FROM vectors"
            ).fetchone()[0]
            self.assignments[vertex] = cell
            try:
                with self.store.db:
                    self.store.db.execute(
                        "INSERT INTO vectors VALUES (?,?,?,?)",
                        (vertex, cell, vector.astype("<f4").tobytes(), position),
                    )
                    self.store.put("next_seq", seq + 1)
                    trace = Trace(self, self.graph, vertex)
                    op, writes = trace.execute(seq)
                    self.store.log(op, writes, trace.cells)
                    for owner, (_, after) in writes.items():
                        self.store.graph_write(owner, after)
                    self.store.put("global", self.graph.metadata)
                    costs = self.store.get("costs")
                    costs["insert_cpu"] += op["cpu"]
                    costs["insert_count"] += 1
                    self.store.put("costs", costs)
            except BaseException:
                self.store.reload_removed()
                self.assignments[vertex] = -1
                self.graph.close()
                self.graph = self._load_graph()
                raise

    def region(self, target, hops=None):
        hops = self.config["hops"] if hops is None else hops
        seen, fringe = {target}, {target}
        for _ in range(hops):
            next_fringe = set()
            for owner in fringe:
                node = self.store.node(owner)
                for layer in node.edges:
                    next_fringe.update(layer)
            next_fringe -= seen
            seen |= next_fringe
            fringe = next_fringe
        cells = {int(self.assignments[v]) for v in seen}
        region = set()
        for cell in cells:
            region.update(
                v
                for (v,) in self.store.db.execute(
                    "SELECT vectors.id FROM vectors JOIN graph ON vectors.id=graph.id WHERE cell=?",
                    (cell,),
                )
            )
        return region

    def estimate(self, target, region):
        row = self.store.db.execute(
            "SELECT seq FROM operations WHERE vertex=?", (target,)
        ).fetchone()
        costs = self.store.get("costs")
        insertion = costs["insert_cpu"] / max(1, costs["insert_count"])
        replay = (
            costs["replay_cpu"] / costs["replay_count"]
            if costs["replay_count"]
            else insertion
        )
        if row is None:
            return dict(
                eligible=False,
                replay_operations=None,
                replay_cpu=None,
                scrub_cpu=len(region) * insertion,
                truncated=False,
            )
        seq = row[0]
        cells = {int(self.assignments[target])}
        cells.update(int(self.assignments[owner]) for owner in self.store.writes(seq))
        candidates = set()
        truncated = False
        for cell in cells:
            for (operation,) in self.store.db.execute(
                "SELECT seq FROM postings WHERE cell=? AND seq>? ORDER BY seq LIMIT ?",
                (cell, seq, self.config["probe"] + 1),
            ):
                candidates.add(operation)
                if len(candidates) > self.config["probe"]:
                    truncated = True
                    break
            if truncated:
                break
        suffix = self.store.db.execute(
            "SELECT COUNT(*) FROM operations WHERE seq>?", (seq,)
        ).fetchone()[0]
        estimated = suffix if truncated else len(candidates)
        # A truncated one-hop probe is an estimate; transitive propagation is measured during replay.
        return dict(
            eligible=True,
            replay_operations=estimated,
            replay_cpu=estimated * replay,
            scrub_cpu=len(region) * insertion,
            truncated=truncated,
        )

    def _replay(self, target, cpu_limit=None):
        row = self.store.db.execute(
            "SELECT seq FROM operations WHERE vertex=?", (target,)
        ).fetchone()
        if row is None:
            raise ValueError(
                "Exact replay is unavailable for a vertex predating the current checkpoint"
            )
        target_seq = int(row[0])
        last = int(
            self.store.db.execute("SELECT MAX(seq) FROM operations").fetchone()[0]
        )
        old_target = self.store.operation(target_seq)
        target_writes = self.store.writes(target_seq)
        trial = Graph(self.capacity, self.dimension, self.config)
        trial.metadata = old_target["before"]
        skipped, replacements, loaded = {target_seq}, defaultdict(list), {}
        changed_owners = set(target_writes) | {target}
        records = []
        prefix = target_seq - 1
        start_cpu = time.thread_time()
        frontier, queued, affected_cells = [], set(), set()
        global_divergence = old_target["before"][:2] != old_target["after"][:2]

        def ensure(owner):
            if loaded.get(owner) == prefix:
                return
            node = self.store.state_at(owner, prefix, skipped, replacements)
            metadata = trial.metadata
            trial.set(owner, node, self.store.vector(owner) if node.active else None)
            trial.metadata = metadata
            loaded[owner] = prefix

        def admit_cell(cell, after):
            if cell in affected_cells:
                return
            affected_cells.add(cell)
            for (seq,) in self.store.db.execute(
                "SELECT seq FROM postings WHERE cell=? AND seq>? ORDER BY seq",
                (cell, after),
            ):
                if seq not in queued:
                    heapq.heappush(frontier, seq)
                    queued.add(seq)

        for owner in changed_owners:
            admit_cell(int(self.assignments[owner]), target_seq)
        next_global = target_seq + 1
        admitted = changed = equal = 0
        try:
            while frontier or (global_divergence and next_global <= last):
                if global_divergence:
                    row = self.store.db.execute(
                        "SELECT MIN(seq) FROM operations WHERE seq>=?", (next_global,)
                    ).fetchone()
                    if row[0] is None:
                        break
                    seq = int(row[0])
                    next_global = seq + 1
                    if seq in queued:
                        queued.remove(seq)
                else:
                    seq = heapq.heappop(frontier)
                    if seq not in queued:
                        continue
                    queued.remove(seq)
                if seq <= target_seq or seq in skipped:
                    continue
                if cpu_limit is not None and time.thread_time() - start_cpu > cpu_limit:
                    raise ReplayAborted(
                        "Observed replay CPU exceeded the scrub work estimate"
                    )
                old = self.store.operation(seq)
                # Entry metadata follows original untouched operations when it has converged.
                if not global_divergence:
                    entry, level, _, _ = old["before"]
                    trial.metadata = (
                        entry,
                        level,
                        old["before"][2] - 1,
                        old["before"][3],
                    )
                else:
                    entry, level, _, _ = trial.metadata
                    trial.metadata = (
                        entry,
                        level,
                        old["before"][2] - 1,
                        old["before"][3],
                    )
                prefix = seq - 1
                trace = Trace(self, trial, old["vertex"], ensure)
                operation, writes = trace.execute(
                    seq, old["scored"] if self.config.get("warm_replay", True) else None
                )
                original_writes = self.store.writes(seq)
                skipped.add(seq)
                owners = set(writes) | set(original_writes)
                different = False
                for owner in owners:
                    before = trace.before.get(owner)
                    if before is None:
                        ensure(owner)
                        before = trial.get(owner)
                    after = writes.get(owner, (before, before))[1]
                    replacements[owner].append((seq, after.pack()))
                    changed_owners.add(owner)
                    original_after = original_writes.get(owner, (None, before))[1]
                    if after != original_after:
                        different = True
                        admit_cell(int(self.assignments[owner]), seq)
                global_divergence = operation["after"][:2] != old["after"][:2]
                next_global = seq + 1
                if global_divergence:
                    different = True
                if (
                    operation["candidates"] != old["candidates"]
                    or operation["scored"] != old["scored"]
                ):
                    different = True
                admitted += 1
                changed += int(different)
                equal += int(not different)
                records.append((operation, writes, trace.cells))
            final_meta = self.store.get("global")
            if global_divergence:
                final_meta = (
                    trial.metadata[0],
                    trial.metadata[1],
                    final_meta[2] - 1,
                    final_meta[3],
                )
            else:
                final_meta = (
                    final_meta[0],
                    final_meta[1],
                    final_meta[2] - 1,
                    final_meta[3],
                )
            patch = {
                owner: self.store.state_at(owner, last, skipped, replacements)
                for owner in changed_owners
            }
            return (
                patch,
                final_meta,
                records,
                dict(
                    admitted=admitted,
                    changed=changed,
                    equal_outputs=equal,
                    cells=len(affected_cells),
                    replay_cpu_seconds=time.thread_time() - start_cpu,
                    distances=trial.counters[0],
                    warm_hits=trial.counters[1],
                    warm_replay=self.config.get("warm_replay", True),
                ),
                target_seq,
            )
        finally:
            trial.close()

    def canonical_order(self, region):
        ids = sorted(region)
        key = self.config["scrub_key"].encode()
        seed = hmac.new(
            key, np.asarray(ids, dtype="<u4").tobytes(), hashlib.sha256
        ).digest()
        return sorted(
            ids,
            key=lambda vertex: (
                hmac.new(
                    seed, int(vertex).to_bytes(4, "little"), hashlib.sha256
                ).digest(),
                vertex,
            ),
        )

    def _scrub(self, target, region):
        survivors = {
            v
            for v in region
            if v != target and self.store.node(v).active and not self.store.node(v).tomb
        }
        trial = Graph(self.capacity, self.dimension, self.config)
        try:
            for vertex in self.canonical_order(survivors):
                trial.insert(vertex, self.store.vector(vertex))
            patch = {v: trial.get(v) if v in survivors else Node() for v in region}
            canonical_hash = hashlib.sha256(b"EXPUNGE-canonical-stage-v1")
            for v in sorted(survivors):
                canonical_hash.update(int(v).to_bytes(4, "little"))
                canonical_hash.update(patch[v].pack())
            patch[target] = Node()
            self.store.db.execute(
                "CREATE TEMP TABLE IF NOT EXISTS region_ids(id INTEGER PRIMARY KEY)"
            )
            self.store.db.execute("DELETE FROM region_ids")
            self.store.db.executemany(
                "INSERT INTO region_ids VALUES (?)", ((v,) for v in region | {target})
            )
            boundary = [
                v
                for (v,) in self.store.db.execute(
                    "SELECT DISTINCT owner FROM edges JOIN region_ids ON neighbor=region_ids.id WHERE owner NOT IN (SELECT id FROM region_ids)"
                )
            ]
            scratch = Graph(self.capacity, self.dimension, self.config)
            try:
                # Retain exterior connection candidates; only the preceding isolated stage is canonical.
                for owner in sorted(survivors):
                    original, rebuilt = self.store.node(owner), patch[owner]
                    layers = []
                    for level, internal in enumerate(rebuilt.edges):
                        external = [
                            v
                            for v in original.edges[level]
                            if v not in region
                            and v != target
                            and not self.store.node(v).tomb
                        ]
                        candidates = list(internal) + external
                        scratch.set(owner, rebuilt, self.store.vector(owner))
                        for v in set(candidates):
                            node = patch[v] if v in patch else self.store.node(v)
                            scratch.set(v, node, self.store.vector(v))
                        bound = self.config["degree"] * (
                            2 if self.config["backend"] == "hnsw" and level == 0 else 1
                        )
                        layers.append(
                            scratch.prune(owner, candidates, bound)
                            if len(candidates) > bound
                            else tuple(candidates)
                        )
                    patch[owner] = replace(rebuilt, edges=tuple(layers))
                for owner in boundary:
                    node = self.store.node(owner)
                    next_edges = []
                    candidates, _ = trial.query(
                        self.store.vector(owner),
                        min(self.config["beam"], max(1, len(survivors))),
                        self.config["beam"],
                    )
                    for level, old in enumerate(node.edges):
                        replacement = [
                            v for v in old if v not in region and v != target
                        ]
                        replacement += [
                            int(v) for v in candidates if patch[int(v)].level >= level
                        ]
                        scratch.set(owner, node, self.store.vector(owner))
                        for v in set(replacement):
                            n = patch[v] if v in patch else self.store.node(v)
                            scratch.set(v, n, self.store.vector(v))
                        bound = self.config["degree"] * (
                            2 if self.config["backend"] == "hnsw" and level == 0 else 1
                        )
                        next_edges.append(scratch.prune(owner, replacement, bound))
                    patch[owner] = replace(node, edges=tuple(next_edges))
            finally:
                scratch.close()
            metadata = list(self.graph.metadata)
            metadata[2] -= sum(
                self.store.node(v).active for v in patch if not patch[v].active
            )
            metadata[3] -= sum(
                self.store.node(v).tomb for v in patch if not patch[v].active
            )
            metadata[:2] = self._entry_for_patch(patch)
            return (
                patch,
                tuple(metadata),
                dict(
                    region_size=len(region),
                    rebuilt=len(survivors),
                    boundary=len(boundary),
                    region_ids=sorted(region),
                    canonical_stage_digest=canonical_hash.hexdigest(),
                    boundary_policy="bidirectional candidate union and native pruning",
                    final_interior_is_canonical=False,
                ),
            )
        finally:
            trial.close()

    def _entry_for_patch(self, patch):
        entry, level, _, _ = self.graph.metadata
        if entry != NONE:
            node = patch[entry] if entry in patch else self.store.node(entry)
            if node.active:
                return entry, level
        for owner, raw in self.store.db.execute(
            "SELECT id,state FROM graph ORDER BY level DESC,id"
        ):
            node = patch[owner] if owner in patch else Node.unpack(raw)
            if node.active:
                return owner, node.level
        return NONE, -1

    def _repair(self, target, mode):
        deleted = self.store.node(target)
        if mode == "tombstone":
            return (
                {target: replace(deleted, tomb=True)},
                (*self.graph.metadata[:3], self.graph.metadata[3] + 1),
                {},
            )
        incoming = [
            v
            for (v,) in self.store.db.execute(
                "SELECT DISTINCT owner FROM edges WHERE neighbor=?", (target,)
            )
            if v != target
        ]
        pools = {}

        def pool(owner, level):
            key = owner, level
            if key not in pools:
                pools[key] = [
                    v for v in self.store.node(owner).edges[level] if v != target
                ]
            return pools[key]

        def distance(a, b):
            x, y = self.store.vector(a), self.store.vector(b)
            if self.config["metric"] == "cosine":
                return max(
                    0.0,
                    1
                    - float(x @ y)
                    / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-30),
                )
            return float(np.sum((x - y) ** 2))

        for owner in incoming:
            for level, old in enumerate(self.store.node(owner).edges):
                if target in old:
                    pool(owner, level)
        if mode == "consolidate":
            for owner in incoming:
                for level, old in enumerate(self.store.node(owner).edges):
                    if target in old and level <= deleted.level:
                        pool(owner, level).extend(deleted.edges[level])
        elif mode == "inplace":
            visited = set()
            self.graph.on_read(lambda v: visited.add(v))
            try:
                candidates, _ = self.graph.query(
                    self.store.vector(target),
                    self.config["inplace_candidates"],
                    self.config["inplace_width"],
                )
            finally:
                self.graph.on_read(None)
            candidates = [
                int(v)
                for v in candidates
                if v != target and not self.store.node(int(v)).tomb
            ]
            copies = self.config["inplace_copies"]
            for level in range(deleted.level + 1):
                usable = [v for v in candidates if self.store.node(v).level >= level]
                for owner in incoming:
                    node = self.store.node(owner)
                    if (
                        owner in visited
                        and node.level >= level
                        and target in node.edges[level]
                    ):
                        pool(owner, level).extend(
                            sorted(
                                (v for v in usable if v != owner),
                                key=lambda v: (distance(owner, v), v),
                            )[:copies]
                        )
                for out in deleted.edges[level]:
                    for owner in sorted(
                        (v for v in usable if v != out),
                        key=lambda v: (distance(out, v), v),
                    )[:copies]:
                        pool(owner, level).append(out)
        elif mode == "randomwalk":
            # SPatch's deterministic top-weight star-mesh sparsification, applied per layer.
            for level in range(deleted.level + 1):
                ins = [
                    v
                    for v in incoming
                    if self.store.node(v).level >= level
                    and target in self.store.node(v).edges[level]
                ]
                outs = [v for v in deleted.edges[level] if not self.store.node(v).tomb]
                if not ins or not outs:
                    continue
                r2 = self.config["spatch_r"] ** 2
                logs = [-r2 * distance(v, target) for v in ins] + [
                    -r2 * distance(target, v) for v in outs
                ]
                maximum = max(logs)
                log_degree = maximum + np.log(sum(np.exp(v - maximum) for v in logs))
                weighted = []
                for owner in ins:
                    existing = set(pool(owner, level))
                    for out in outs:
                        if out == owner:
                            continue
                        added = (
                            -r2 * (distance(owner, target) + distance(target, out))
                            - log_degree
                        )
                        original = (
                            -r2 * distance(owner, out) if out in existing else -np.inf
                        )
                        weighted.append(
                            (float(np.logaddexp(original, added)), owner, out)
                        )
                    pools[owner, level] = [
                        v for v in pool(owner, level) if v not in outs
                    ]
                keep = int(
                    np.ceil(self.config["spatch_alpha"] * (len(ins) + len(outs)))
                )
                for _, owner, out in sorted(
                    weighted, key=lambda p: (-p[0], p[1], p[2])
                )[:keep]:
                    pool(owner, level).append(out)
        else:
            raise ValueError("Unknown repair mode")
        patch = {target: Node()}
        for owner in sorted({owner for owner, _ in pools}):
            node = self.store.node(owner)
            layers = []
            for level, old in enumerate(node.edges):
                candidates = pools.get((owner, level), list(old))
                candidates = list(
                    dict.fromkeys(
                        v
                        for v in candidates
                        if v != target
                        and v != owner
                        and self.store.node(v).active
                        and not self.store.node(v).tomb
                        and self.store.node(v).level >= level
                    )
                )
                bound = self.config["degree"] * (
                    2 if self.config["backend"] == "hnsw" and level == 0 else 1
                )
                layers.append(
                    self.graph.prune(owner, candidates, bound)
                    if len(candidates) > bound
                    else tuple(candidates)
                )
            patch[owner] = replace(node, edges=tuple(layers))
        metadata = list(self.graph.metadata)
        metadata[2] -= 1
        metadata[3] -= int(deleted.tomb)
        metadata[:2] = self._entry_for_patch(patch)
        return patch, tuple(metadata), dict(repaired=len(patch) - 1, policy=mode)

    def _full(self, target):
        if self.store.get("epoch") != 0:
            raise ValueError(
                "Full retained-history replay requires an unmodified insertion-history epoch"
            )
        graph = Graph(self.capacity, self.dimension, self.config)
        records = []
        try:
            for seq, vertex in list(
                self.store.db.execute("SELECT seq,vertex FROM operations ORDER BY seq")
            ):
                if vertex == target:
                    continue
                trace = Trace(self, graph, vertex)
                operation, writes = trace.execute(seq)
                records.append((operation, writes, trace.cells))
            patch = {
                v: graph.get(v)
                for (v,) in self.store.db.execute("SELECT id FROM graph")
            }
            return patch, graph.metadata, records
        finally:
            graph.close()

    def digest(self):
        parameters = {
            k: self.config[k]
            for k in ("backend", "degree", "beam", "alpha", "metric", "seed")
        }
        h = hashlib.sha256(
            b"EXPUNGE-state-v1"
            + canonical(parameters)
            + canonical(self.store.get("shape"))
        )
        h.update(canonical(self.store.get("global")))
        h.update(self.store.root_digest())
        return h.hexdigest()

    def _budget(self, cpu):
        rate = self.config["cpu_fraction"] * self.config["cores"]
        now = time.monotonic()
        self.tokens = min(rate, self.tokens + (now - self.last_refill) * rate)
        self.tokens -= cpu
        self.last_refill = now
        if self.tokens < 0:
            time.sleep(-self.tokens / rate)
            self.tokens = 0
            self.last_refill = time.monotonic()

    def delete(self, target, mode="governor", region=None, budget=True):
        if mode not in (
            "replay",
            "scrub",
            "governor",
            "tombstone",
            "consolidate",
            "inplace",
            "randomwalk",
            "full",
        ):
            raise ValueError("Unknown deletion path")
        with self._mutation:
            node = self.store.node(target)
            if not node.active or (node.tomb and mode == "tombstone"):
                raise ValueError("Target must be present and cannot be masked twice")
            start_wall, start_cpu = time.monotonic(), time.thread_time()
            before_digest = self.digest()
            region = (
                self.region(target)
                if region is None
                else set(map(int, region)) | {target}
            )
            if any(v < 0 or v >= self.capacity for v in region):
                raise ValueError("Region contains invalid stable IDs")
            estimate = self.estimate(target, region)
            path, records, target_seq, details = mode, [], None, {}
            if path == "governor":
                path = (
                    "replay"
                    if estimate["eligible"]
                    and estimate["replay_cpu"] <= estimate["scrub_cpu"]
                    else "scrub"
                )
            try:
                if path == "replay":
                    try:
                        patch, metadata, records, details, target_seq = self._replay(
                            target,
                            estimate["scrub_cpu"] if mode == "governor" else None,
                        )
                    except ReplayAborted:
                        rollback_cpu = time.thread_time() - start_cpu
                        path = "scrub"
                        details = dict(
                            replay_aborted=True,
                            discarded_trial_cpu_seconds=rollback_cpu,
                        )
                        patch, metadata, scrub_details = self._scrub(target, region)
                        details.update(scrub_details)
                elif path == "scrub":
                    patch, metadata, details = self._scrub(target, region)
                elif path == "full":
                    patch, metadata, records = self._full(target)
                    target_seq = self.store.db.execute(
                        "SELECT seq FROM operations WHERE vertex=?", (target,)
                    ).fetchone()[0]
                else:
                    with self._lock:
                        patch, metadata, details = self._repair(target, path)
                with self._lock, self.store.db:
                    for vertex, after in patch.items():
                        self.store.graph_write(vertex, after)
                        self.graph.set(
                            vertex,
                            after,
                            self.store.vector(vertex) if after.active else None,
                        )
                    self.graph.metadata = metadata
                    self.store.put("global", metadata)
                    if path in ("replay", "full"):
                        self.store.db.execute(
                            "DELETE FROM operations WHERE seq=?", (target_seq,)
                        )
                        if path == "full":
                            self.store.db.execute("DELETE FROM operations")
                        for operation, writes, cells in records:
                            self.store.log(operation, writes, cells)
                        self.store.mark_removed(target_seq)
                    else:
                        self.store.checkpoint(metadata)
                    if path != "tombstone":
                        self.store.db.executemany(
                            "UPDATE vectors SET value=NULL WHERE id=?",
                            ((v,) for v, n in patch.items() if not n.active),
                        )
                    costs = self.store.get("costs")
                    costs["replay_cpu"] += details.get("replay_cpu_seconds", 0)
                    costs["replay_count"] += details.get("admitted", 0)
                    self.store.put("costs", costs)
                    record = dict(
                        target=target,
                        requested=mode,
                        path=path,
                        exact=path in ("replay", "full"),
                        epoch=self.store.get("epoch"),
                        before_digest=before_digest,
                        after_digest=self.digest(),
                        estimate=estimate,
                        details=details,
                        cpu_seconds=time.thread_time() - start_cpu,
                        wall_seconds=time.monotonic() - start_wall,
                        timestamp_ns=time.time_ns(),
                        status="committed",
                        timing_scope="through record creation; excludes final commit and pacing",
                    )
                    raw = canonical(record)
                    self.store.db.execute(
                        "INSERT INTO maintenance(record,signature) VALUES (?,?)",
                        (raw, self.key.sign(raw)),
                    )
            except BaseException:
                self.store.db.rollback()
                self.store.reload_removed()
                with self._lock:
                    self.graph.close()
                    self.graph = self._load_graph()
                raise
            # Pacing includes discovery, private replay, rollback, scrub, and persistence work.
            record["total_cpu_seconds"] = time.thread_time() - start_cpu
            record["total_wall_seconds"] = time.monotonic() - start_wall
            if budget:
                self._budget(record["total_cpu_seconds"])
            record["paced_wall_seconds"] = time.monotonic() - start_wall
            return record

    def scrub_matched(self, region):
        """Apply a matched canonical maintenance event to a never-present branch."""
        with self._mutation:
            region = set(map(int, region))
            if any(v < 0 or v >= self.capacity for v in region):
                raise ValueError("Matched region contains invalid IDs")
            absent = next(
                (v for v in range(self.capacity) if not self.store.node(v).active), None
            )
            if absent is None:
                raise ValueError("Matched scrub needs a reserved absent stable ID")
            start, cpu = time.monotonic(), time.thread_time()
            before = self.digest()
            patch, metadata, details = self._scrub(absent, region)
            try:
                with self._lock, self.store.db:
                    for vertex, node in patch.items():
                        self.store.graph_write(vertex, node)
                        self.graph.set(
                            vertex,
                            node,
                            self.store.vector(vertex) if node.active else None,
                        )
                    self.graph.metadata = metadata
                    self.store.put("global", metadata)
                    self.store.checkpoint(metadata)
                    self.store.db.executemany(
                        "UPDATE vectors SET value=NULL WHERE id=?",
                        ((v,) for v, node in patch.items() if not node.active),
                    )
                    record = dict(
                        target=None,
                        requested="matched_scrub",
                        path="scrub",
                        exact=False,
                        epoch=self.store.get("epoch"),
                        before_digest=before,
                        after_digest=self.digest(),
                        details=details,
                        status="committed",
                        cpu_seconds=time.thread_time() - cpu,
                        wall_seconds=time.monotonic() - start,
                        timestamp_ns=time.time_ns(),
                        timing_scope="through record creation; excludes final commit",
                    )
                    raw = canonical(record)
                    self.store.db.execute(
                        "INSERT INTO maintenance(record,signature) VALUES (?,?)",
                        (raw, self.key.sign(raw)),
                    )
            except BaseException:
                self.store.db.rollback()
                self.store.reload_removed()
                with self._lock:
                    self.graph.close()
                    self.graph = self._load_graph()
                raise
            return details

    def validate(self):
        with self._lock:
            return self._validate_unlocked()

    def _validate_unlocked(self):
        nodes = {
            int(v): Node.unpack(raw)
            for v, raw in self.store.db.execute("SELECT id,state FROM graph")
        }
        for owner, node in nodes.items():
            if self.graph.get(owner) != node:
                raise AssertionError("Persistent and native graph differ")
            for level, layer in enumerate(node.edges):
                for other in layer:
                    if other not in nodes or nodes[other].level < level:
                        raise AssertionError(
                            f"Invalid edge {owner}->{other} at level {level}"
                        )
        entry, level, count, deleted = self.graph.metadata
        if count != len(nodes) or deleted != sum(n.tomb for n in nodes.values()):
            raise AssertionError("Graph counts disagree")
        if count and (entry not in nodes or nodes[entry].level != level):
            raise AssertionError("Invalid entry point")
        if not count and entry != NONE:
            raise AssertionError("Nonempty metadata on an empty graph")
        if count and level != max(n.level for n in nodes.values()):
            raise AssertionError("Maximum level metadata differs from live state")
        self.store.verify_root()
        if tuple(self.store.get("global")) != self.graph.metadata:
            raise AssertionError("Persistent and native metadata differ")
        return dict(
            vertices=count,
            tombstones=deleted,
            edges=sum(len(layer) for n in nodes.values() for layer in n.edges),
            digest=self.digest(),
        )

    def export(self, path):
        with self._lock.read():
            self._export_unlocked(path)

    def _export_unlocked(self, path):
        ids, states, payload, offsets = [], [], [], [0]
        for vertex, raw in self.store.db.execute(
            "SELECT id,state FROM graph ORDER BY id"
        ):
            ids.append(vertex)
            states.extend(np.frombuffer(raw, dtype="<u4"))
            offsets.append(len(states))
            payload.append(self.store.vector(vertex))
        np.savez_compressed(
            path,
            ids=np.asarray(ids, np.uint32),
            states=np.asarray(states, np.uint32),
            offsets=np.asarray(offsets, np.uint64),
            vectors=np.asarray(payload, np.float32).reshape(-1, self.dimension),
            metadata=np.asarray(self.graph.metadata, dtype=np.int64),
            config=canonical(self.config).decode(),
        )

    def stats(self):
        with self._lock:
            return self._stats_unlocked()

    def _stats_unlocked(self):
        tables = ("graph", "operations", "versions", "postings", "edges", "merkle")
        counts = {
            name: self.store.db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in tables
        }
        size = sum(
            p.stat().st_size
            for p in (self.store.path, Path(str(self.store.path) + "-wal"))
            if p.exists()
        )
        bitmap = self.store.db.execute(
            "SELECT COALESCE(SUM(length(bitmap)),0),COALESCE(SUM(length(scored)+length(candidates)),0) FROM operations"
        ).fetchone()
        versions = self.store.db.execute(
            "SELECT COALESCE(SUM(length(before_state)+length(after_state)),0) FROM versions"
        ).fetchone()[0]
        return dict(
            counts=counts,
            disk_bytes=size,
            bitmap_bytes=bitmap[0],
            context_bytes=bitmap[1],
            version_bytes=versions,
            config=self.config,
            epoch=self.store.get("epoch"),
            graph=self._validate_unlocked(),
        )
