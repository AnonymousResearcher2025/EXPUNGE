"""Typed interface to the instrumented C++ graph engines."""

from __future__ import annotations
import ctypes as C
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np

NONE = (1 << 32) - 1
U32P, F32P = C.POINTER(C.c_uint32), C.POINTER(C.c_float)
CALLBACK = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint32)


@dataclass(frozen=True)
class Node:
    active: bool = False
    tomb: bool = False
    level: int = 0
    edges: tuple[tuple[int, ...], ...] = ()

    def pack(self) -> bytes:
        values = [int(self.active), int(self.tomb), self.level]
        for layer in self.edges:
            values += [len(layer), *layer]
        return np.asarray(values, dtype="<u4").tobytes()

    @classmethod
    def unpack(cls, raw: bytes) -> Node:
        a = np.frombuffer(raw, dtype="<u4")
        if len(a) < 3 or a[0] > 1 or a[1] > 1 or a[2] > 64:
            raise ValueError("Invalid node encoding")
        cursor, edges = 3, []
        if a[0]:
            for _ in range(int(a[2]) + 1):
                if cursor >= len(a):
                    raise ValueError("Truncated adjacency")
                size = int(a[cursor])
                cursor += 1
                if cursor + size > len(a):
                    raise ValueError("Truncated adjacency")
                edges.append(tuple(map(int, a[cursor : cursor + size])))
                cursor += size
        if cursor != len(a):
            raise ValueError("Trailing node data")
        return cls(bool(a[0]), bool(a[1]), int(a[2]), tuple(edges))


EMPTY = Node().pack()


class Graph:
    @staticmethod
    def library_fingerprint():
        path = Path(__file__).resolve().parents[1] / "build/libexpunge.so"
        if not path.exists():
            raise RuntimeError("Native library missing; run make at the project root")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def __init__(self, capacity: int, dimension: int, config: dict):
        self.capacity, self.dimension = capacity, dimension
        self.config = config
        path = Path(__file__).resolve().parents[1] / "build/libexpunge.so"
        if not path.exists():
            raise RuntimeError("Native library missing; run make at the project root")
        self.lib = C.CDLL(str(path))
        signatures = {
            "ex_create": (
                [
                    C.c_uint32,
                    C.c_uint32,
                    C.c_int,
                    C.c_uint32,
                    C.c_uint32,
                    C.c_float,
                    C.c_int,
                    C.c_uint64,
                ],
                C.c_void_p,
            ),
            "ex_destroy": ([C.c_void_p], None),
            "ex_error": ([], C.c_char_p),
            "ex_callback": ([C.c_void_p, CALLBACK, C.c_void_p], None),
            "ex_insert": ([C.c_void_p, C.c_uint32, F32P], C.c_int),
            "ex_get": ([C.c_void_p, C.c_uint32, U32P, C.c_size_t], C.c_int),
            "ex_set": ([C.c_void_p, C.c_uint32, U32P, C.c_size_t, F32P], C.c_int),
            "ex_global": ([C.c_void_p, C.POINTER(C.c_uint64)], None),
            "ex_metadata": (
                [C.c_void_p, C.c_uint32, C.c_int, C.c_uint64, C.c_uint64],
                C.c_int,
            ),
            "ex_query": (
                [C.c_void_p, F32P, C.c_uint32, C.c_uint32, U32P, F32P],
                C.c_int,
            ),
            "ex_prune": (
                [C.c_void_p, C.c_uint32, U32P, C.c_size_t, C.c_size_t, U32P],
                C.c_int,
            ),
            "ex_warm": ([C.c_void_p, C.c_uint32, C.c_float], None),
            "ex_scored": ([C.c_void_p, U32P, F32P], C.c_int),
            "ex_candidates": ([C.c_void_p, U32P], C.c_int),
            "ex_counters": ([C.c_void_p, C.POINTER(C.c_uint64)], None),
        }
        for name, (args, ret) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = ret
        self.ptr = self.lib.ex_create(
            capacity,
            dimension,
            config["backend"] == "hnsw",
            config["degree"],
            config["beam"],
            config["alpha"],
            config["metric"] == "cosine",
            config["seed"],
        )
        if not self.ptr:
            raise RuntimeError(self.lib.ex_error().decode())
        self._callback = CALLBACK(lambda _, vertex: 1)
        self.callback_error: BaseException | None = None

    def close(self):
        if self.ptr:
            self.lib.ex_destroy(self.ptr)
            self.ptr = None

    def __del__(self):
        if getattr(self, "ptr", None):
            self.close()

    def check(self, status: int) -> int:
        if status < 0:
            if self.callback_error is not None:
                error, self.callback_error = self.callback_error, None
                raise error
            raise RuntimeError(self.lib.ex_error().decode())
        return status

    def on_read(self, fn: Callable[[int], None] | None):
        self.callback_error = None
        if fn is None:
            self._callback = CALLBACK()
            self.lib.ex_callback(self.ptr, self._callback, None)
            return

        def call(_, vertex):
            try:
                if fn:
                    fn(int(vertex))
                return 1
            except BaseException as error:
                self.callback_error = error
                return 0

        self._callback = CALLBACK(call)
        self.lib.ex_callback(self.ptr, self._callback, None)

    def get(self, vertex: int) -> Node:
        size = self.check(self.lib.ex_get(self.ptr, vertex, None, 0))
        raw = np.empty(size, dtype=np.uint32)
        self.check(self.lib.ex_get(self.ptr, vertex, raw.ctypes.data_as(U32P), size))
        return Node.unpack(raw.tobytes())

    def set(self, vertex: int, node: Node, vector: np.ndarray | None):
        raw = np.frombuffer(node.pack(), dtype=np.uint32)
        vector = None if vector is None else self.vector(vector)
        self.check(
            self.lib.ex_set(
                self.ptr,
                vertex,
                raw.ctypes.data_as(U32P),
                len(raw),
                None if vector is None else vector.ctypes.data_as(F32P),
            )
        )

    def vector(self, values) -> np.ndarray:
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            raise ValueError("Vector dimension mismatch or nonfinite values")
        return values

    @property
    def metadata(self) -> tuple[int, int, int, int]:
        raw = (C.c_uint64 * 4)()
        self.lib.ex_global(self.ptr, raw)
        return int(raw[0]), C.c_int64(raw[1]).value, int(raw[2]), int(raw[3])

    @metadata.setter
    def metadata(self, value):
        self.check(self.lib.ex_metadata(self.ptr, *map(int, value)))

    def insert(
        self, vertex: int, vector: np.ndarray, warm: dict[int, float] | None = None
    ):
        for other, distance in (warm or {}).items():
            self.lib.ex_warm(self.ptr, other, distance)
        values = self.vector(vector)
        self.check(self.lib.ex_insert(self.ptr, vertex, values.ctypes.data_as(F32P)))

    def scored(self) -> dict[int, float]:
        size = self.lib.ex_scored(self.ptr, None, None)
        ids = np.empty(size, np.uint32)
        distances = np.empty(size, np.float32)
        self.lib.ex_scored(
            self.ptr, ids.ctypes.data_as(U32P), distances.ctypes.data_as(F32P)
        )
        return dict(zip(map(int, ids), map(float, distances)))

    def candidates(self) -> bytes:
        size = self.lib.ex_candidates(self.ptr, None)
        ids = np.empty(size, np.uint32)
        self.lib.ex_candidates(self.ptr, ids.ctypes.data_as(U32P))
        return ids.astype("<u4").tobytes()

    @property
    def counters(self):
        raw = (C.c_uint64 * 2)()
        self.lib.ex_counters(self.ptr, raw)
        return tuple(map(int, raw))

    def query(self, vector, k: int = 10, width: int = 100):
        if k < 1 or width < 1:
            raise ValueError("k and search width must be positive")
        values = self.vector(vector)
        ids = np.empty(k, np.uint32)
        distances = np.empty(k, np.float32)
        size = self.check(
            self.lib.ex_query(
                self.ptr,
                values.ctypes.data_as(F32P),
                k,
                width,
                ids.ctypes.data_as(U32P),
                distances.ctypes.data_as(F32P),
            )
        )
        return ids[:size], distances[:size]

    def prune(self, owner: int, candidates, limit: int) -> tuple[int, ...]:
        ids = np.asarray(list(candidates), dtype=np.uint32)
        out = np.empty(limit, np.uint32)
        size = self.check(
            self.lib.ex_prune(
                self.ptr,
                owner,
                ids.ctypes.data_as(U32P),
                len(ids),
                limit,
                out.ctypes.data_as(U32P),
            )
        )
        return tuple(map(int, out[:size]))
