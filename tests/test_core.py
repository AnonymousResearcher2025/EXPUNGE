import tempfile
import hashlib
import unittest
from pathlib import Path
import numpy as np
from expunge import Index
from expunge.data import Partition
from expunge.native import Graph, Node


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.x = np.random.default_rng(19).normal(size=(90, 8)).astype(np.float32)
        self.cfg = dict(
            cells=8, partition_sample=90, degree=4, beam=20, cores=1, cpu_fraction=1
        )
        self.partition = Partition.fit(self.x, 8, 90, 42, "l2")

    def tearDown(self):
        self.temp.cleanup()

    def build(self, name, backend, order=None):
        return Index.build(
            self.root / name,
            self.x,
            dict(self.cfg, backend=backend),
            capacity=100,
            order=order,
            partition=self.partition,
        )

    def compare(self, actual, oracle):
        self.assertEqual(actual.graph.metadata, oracle.graph.metadata)
        for vertex in range(100):
            self.assertEqual(
                actual.graph.get(vertex), oracle.graph.get(vertex), f"vertex {vertex}"
            )
        actual.validate()
        self.assertEqual(actual.digest(), oracle.digest())
        expected_ops = {
            vertex: seq
            for seq, vertex in oracle.store.db.execute(
                "SELECT seq,vertex FROM operations"
            )
        }
        for seq, vertex in actual.store.db.execute("SELECT seq,vertex FROM operations"):
            op, expected = (
                actual.store.operation(seq),
                oracle.store.operation(expected_ops[vertex]),
            )
            self.assertEqual(op["bitmap"], expected["bitmap"])
            self.assertEqual(op["candidates"], expected["candidates"])
            self.assertEqual(op["scored"], expected["scored"])
            self.assertEqual(op["before"], expected["before"])
            self.assertEqual(op["after"], expected["after"])

    def test_replay_oracle_both_backends(self):
        for backend in ("hnsw", "vamana"):
            for target in (0, 1, 17, 45, 89):
                with self.subTest(backend=backend, target=target):
                    name = f"{backend}-{target}"
                    with (
                        self.build(name, backend) as index,
                        self.build(
                            name + "-oracle",
                            backend,
                            [v for v in range(90) if v != target],
                        ) as oracle,
                    ):
                        record = index.delete(target, "replay", budget=False)
                        self.assertTrue(record["exact"])
                        self.compare(index, oracle)
                    with (
                        Index(self.root / name) as loaded,
                        Index(self.root / (name + "-oracle")) as oracle,
                    ):
                        self.compare(loaded, oracle)

    def test_sequential_replay_and_insert(self):
        for backend in ("hnsw", "vamana"):
            name = "sequence-" + backend
            removed = {0, 14, 20}
            with self.build(name, backend) as index:
                for vertex in (14, 0, 20):
                    index.delete(vertex, "replay", budget=False)
                with self.build(
                    name + "-oracle",
                    backend,
                    [v for v in range(90) if v not in removed],
                ) as oracle:
                    self.compare(index, oracle)
                    vector = np.arange(8, dtype=np.float32)
                    index.insert(95, vector)
                    oracle.insert(95, vector)
                    self.compare(index, oracle)
                    index.delete(95, "replay", budget=False)
                    oracle.delete(95, "replay", budget=False)
                    self.compare(index, oracle)

    def test_scrub_is_canonical_interior(self):
        for backend in ("hnsw", "vamana"):
            with self.build("scrub-" + backend, backend) as index:
                region = index.region(15)
                order = index.canonical_order(region - {15})
                expected = Graph(100, 8, dict(index.config))
                try:
                    for vertex in order:
                        expected.insert(vertex, self.x[vertex])
                    digest = hashlib.sha256(b"EXPUNGE-canonical-stage-v1")
                    for vertex in sorted(region - {15}):
                        digest.update(vertex.to_bytes(4, "little"))
                        digest.update(expected.get(vertex).pack())
                    record = index.delete(15, "scrub", region=region, budget=False)
                    index.validate()
                    self.assertEqual(
                        record["details"]["canonical_stage_digest"], digest.hexdigest()
                    )
                    self.assertFalse(record["details"]["final_interior_is_canonical"])
                    with self.assertRaises(ValueError):
                        index.delete(16, "replay", budget=False)
                    index.insert(95, np.arange(8, dtype=np.float32))
                    index.delete(95, "replay", budget=False)
                    index.validate()
                finally:
                    expected.close()

    def test_repair_modes(self):
        for backend in ("hnsw", "vamana"):
            for mode in (
                "tombstone",
                "consolidate",
                "inplace",
                "randomwalk",
                "full",
                "governor",
            ):
                with (
                    self.subTest(backend=backend, mode=mode),
                    self.build(backend + mode, backend) as index,
                ):
                    index.delete(18, mode, budget=False)
                    index.validate()
                    result, _ = index.query(self.x[18], 10)
                    self.assertNotIn(18, result)
                    self.assertEqual(
                        index.store.vector(18).shape, (8,)
                    ) if mode == "tombstone" else self.assertIsNone(
                        index.store.db.execute(
                            "SELECT value FROM vectors WHERE id=18"
                        ).fetchone()[0]
                    )

    def test_node_encoding(self):
        node = Node(True, False, 1, ((3, 8), (3,)))
        self.assertEqual(Node.unpack(node.pack()), node)
        with self.assertRaises(ValueError):
            Node.unpack(node.pack()[:-1])

    def test_invalid_inputs(self):
        with self.build("invalid", "hnsw") as index:
            with self.assertRaises(ValueError):
                index.insert(95, np.ones(7))
            with self.assertRaises(ValueError):
                index.delete(99)
            with self.assertRaises(ValueError):
                index.query(np.ones(8), 0)
            index.validate()


class EdgeCaseTests(unittest.TestCase):
    def test_sparse_order_ties_and_cosine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x = (
                np.random.default_rng(90)
                .integers(-2, 3, size=(48, 5))
                .astype(np.float32)
            )
            x[5] = x[6]
            x[7] = 0
            order = list(map(int, np.random.default_rng(7).permutation(len(x))))
            for backend in ("hnsw", "vamana"):
                cfg = dict(
                    cells=12,
                    partition_sample=48,
                    degree=3,
                    beam=12,
                    backend=backend,
                    metric="cosine",
                )
                partition = Partition.fit(x, 12, 48, 42, "cosine")
                with Index.build(
                    root / backend, x, cfg, order=order, partition=partition
                ) as index:
                    removed = set(order[:4]) | {5, 47}
                    for target in removed:
                        index.delete(target, "replay", budget=False)
                    with Index.build(
                        root / (backend + "-oracle"),
                        x,
                        cfg,
                        order=[v for v in order if v not in removed],
                        partition=partition,
                    ) as oracle:
                        self.assertEqual(index.graph.metadata, oracle.graph.metadata)
                        self.assertEqual(index.digest(), oracle.digest())
                        for vertex in range(len(x)):
                            self.assertEqual(
                                index.graph.get(vertex), oracle.graph.get(vertex)
                            )
                        index.validate()

    def test_empty_graph_and_tombstone_consolidation(self):
        for backend in ("hnsw", "vamana"):
            for mode in ("replay", "scrub", "consolidate"):
                with tempfile.TemporaryDirectory() as temporary:
                    x = np.ones((1, 4), np.float32)
                    cfg = dict(
                        cells=1, partition_sample=1, degree=2, beam=4, backend=backend
                    )
                    with Index.build(Path(temporary) / "index", x, cfg) as index:
                        if mode == "consolidate":
                            index.delete(0, "tombstone", budget=False)
                        index.delete(0, mode, budget=False)
                        self.assertEqual(index.validate()["vertices"], 0)
                        self.assertEqual(len(index.query(x[0])[0]), 0)
                        self.assertIsNone(
                            index.store.db.execute(
                                "SELECT value FROM vectors"
                            ).fetchone()[0]
                        )


if __name__ == "__main__":
    unittest.main()
