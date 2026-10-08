"""Memory-mapped vector formats and a frozen geometric partition."""

from __future__ import annotations
from pathlib import Path
import struct
import numpy as np
from sklearn.cluster import KMeans


def vectors(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        a = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".fbin":
        with path.open("rb") as f:
            head = f.read(8)
        if len(head) != 8:
            raise ValueError("Truncated fbin header")
        n, d = struct.unpack("<II", head)
        if not d or path.stat().st_size != 8 + 4 * n * d:
            raise ValueError("Invalid fbin size")
        a = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(n, d))
    elif suffix in (".fvecs", ".bvecs"):
        with path.open("rb") as f:
            head = f.read(4)
        if len(head) != 4:
            raise ValueError("Truncated vector header")
        d = struct.unpack("<I", head)[0]
        unit = 4 if suffix == ".fvecs" else 1
        row = 4 + unit * d
        if not d or path.stat().st_size % row:
            raise ValueError("Invalid vector file size")
        dtype = np.dtype([("d", "<u4"), ("x", "<f4" if unit == 4 else "u1", (d,))])
        rows = np.memmap(path, dtype=dtype, mode="r")
        if np.any(rows["d"] != d):
            raise ValueError("Inconsistent per-vector dimensions")
        a = rows["x"]
    else:
        raise ValueError("Supported vector formats: .npy, .fbin, .fvecs, .bvecs")
    if a.ndim != 2 or not a.shape[1] or a.dtype.kind not in "fui":
        raise ValueError("Expected a numeric vector matrix")
    return a


def write_fbin(path, values):
    a = np.ascontiguousarray(values, dtype="<f4")
    if a.ndim != 2:
        raise ValueError("Expected a matrix")
    with open(path, "wb") as f:
        f.write(struct.pack("<II", *a.shape))
        f.write(a.tobytes())


class Partition:
    def __init__(self, centers: np.ndarray, metric: str):
        self.centers = np.asarray(centers, np.float32)
        self.metric = metric

    @classmethod
    def fit(cls, values: np.ndarray, cells: int, sample: int, seed: int, metric: str):
        if cells < 1 or cells > len(values) or sample < cells:
            raise ValueError("Partition needs at least one sample per requested cell")
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(len(values), min(sample, len(values)), replace=False))
        x = cls.normalize(np.asarray(values[ids], np.float32), metric)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite partition input")
        model = KMeans(
            n_clusters=cells,
            random_state=seed,
            n_init=1,
            algorithm="lloyd",
            max_iter=100,
        ).fit(x)
        return cls(model.cluster_centers_, metric)

    @staticmethod
    def normalize(x, metric):
        if metric == "cosine":
            x = x / np.maximum(
                np.linalg.norm(x, axis=-1, keepdims=True), np.finfo(np.float32).tiny
            )
        return x

    def assign(self, values, chunk: int = 4096):
        x = np.atleast_2d(np.asarray(values, np.float32))
        ids = np.empty(len(x), np.int32)
        norms = np.sum(self.centers * self.centers, axis=1)
        for start in range(0, len(x), chunk):
            batch = self.normalize(x[start : start + chunk], self.metric)
            d = (
                np.sum(batch * batch, axis=1)[:, None]
                + norms[None, :]
                - 2 * batch @ self.centers.T
            )
            ids[start : start + chunk] = np.argmin(d, axis=1)
        return ids

    def bitmap(self, cells) -> bytes:
        raw = bytearray((len(self.centers) + 7) // 8)
        for cell in cells:
            raw[int(cell) // 8] |= 1 << (int(cell) % 8)
        return bytes(raw)

    def members(self, bitmap: bytes):
        return [
            cell
            for cell in range(len(self.centers))
            if bitmap[cell // 8] & (1 << (cell % 8))
        ]
