import concurrent.futures
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from expunge import Index
from expunge.data import vectors, write_fbin
from expunge.index import ReplayAborted
from expunge.metrics import Snapshot, evaluate, ndcg
from experiments.attack import score


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.x = np.random.default_rng(81).normal(size=(55, 6)).astype(np.float32)
        self.config = dict(
            cells=5, partition_sample=55, degree=4, beam=16, cores=1, cpu_fraction=1
        )

    def tearDown(self):
        self.temp.cleanup()

    def build(self, backend="hnsw"):
        return Index.build(
            self.root / "index.sqlite",
            self.x,
            dict(self.config, backend=backend),
            capacity=65,
        )

    def test_failed_build_cleans_owned_files(self):
        from expunge.data import Partition

        partition = Partition.fit(self.x, 5, 55, 42, "l2")
        self.x[-1] = np.nan
        path = self.root / "bad.sqlite"
        with self.assertRaises(ValueError):
            Index.build(path, self.x, self.config, partition=partition)
        self.assertFalse(path.exists())
        self.assertFalse(Path(str(path) + ".key").exists())

    def test_monotonic_sequence_after_last_deletion(self):
        with self.build() as index:
            index.delete(54, "replay", budget=False)
            index.insert(56, self.x[54])
            self.assertEqual(
                index.store.db.execute(
                    "SELECT seq FROM operations WHERE vertex=56"
                ).fetchone()[0],
                55,
            )
            index.delete(56, "replay", budget=False)
            index.insert(57, self.x[54])
            self.assertEqual(index.store.operation(56)["before"][2], 54)
            index.validate()
        with Index(self.root / "index.sqlite") as index:
            index.delete(57, "replay", budget=False)
            self.assertEqual(index.validate()["vertices"], 54)

    def test_failed_commit_rolls_back(self):
        with self.build() as index:
            digest = index.digest()
            original = index.store.graph_write
            calls = [0]

            def fail(vertex, node):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError("Injected storage failure")
                original(vertex, node)

            with (
                patch.object(index.store, "graph_write", fail),
                self.assertRaises(OSError),
            ):
                index.delete(12, "replay", budget=False)
            index.validate()
            self.assertEqual(index.digest(), digest)
            self.assertEqual(
                index.store.db.execute("SELECT COUNT(*) FROM maintenance").fetchone()[
                    0
                ],
                0,
            )
            self.assertTrue(index.store.node(12).active)

    def test_aborted_replay_has_no_partial_commit(self):
        with self.build() as index:
            estimate = dict(
                eligible=True,
                replay_operations=0,
                replay_cpu=0,
                scrub_cpu=100,
                truncated=False,
            )
            with (
                patch.object(index, "estimate", return_value=estimate),
                patch.object(index, "_replay", side_effect=ReplayAborted()),
            ):
                record = index.delete(12, "governor", budget=False)
            self.assertEqual(record["path"], "scrub")
            self.assertFalse(record["exact"])
            self.assertTrue(record["details"]["replay_aborted"])
            index.validate()

    def test_signatures_detect_tampering(self):
        with self.build() as index:
            index.delete(12, "replay", budget=False)
            raw, signature = index.store.db.execute(
                "SELECT record,signature FROM maintenance"
            ).fetchone()
            public = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(index.store.get("public_key"))
            )
            public.verify(signature, raw)
            with self.assertRaises(InvalidSignature):
                public.verify(signature, raw + b" ")

    def test_queries_during_replay(self):
        for backend in ("hnsw", "vamana"):
            path = self.root / "index.sqlite"
            with self.build(backend) as index:

                def queries():
                    for j in range(100):
                        ids, ds = index.query(self.x[j % len(self.x)], 10, 16 + j % 3)
                        self.assertEqual(len(ids), 10)
                        self.assertTrue(np.isfinite(ds).all())

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                    futures = [pool.submit(queries) for _ in range(4)]
                    record = index.delete(12, "replay", budget=False)
                    for future in futures:
                        future.result()
                self.assertTrue(record["exact"])
                index.validate()
            for suffix in ("", ".key", "-wal", "-shm"):
                Path(str(path) + suffix).unlink(missing_ok=True)

    def test_export_and_retrieval(self):
        with self.build() as index:
            index.delete(12, "scrub", budget=False)
            path = self.root / "snapshot.npz"
            index.export(path)
            snapshot = Snapshot(path)
            self.assertNotIn(12, snapshot.nodes)
            features = snapshot.features(self.x[12])
            self.assertEqual(features.shape, (36,))
            self.assertTrue(np.isfinite(features).all())
            metric = evaluate(index, self.x[:3], index, exact=True)
            self.assertEqual(metric["result_disagreement"], 0)
            self.assertGreater(metric["exact_recall"], 0)
            self.assertEqual(ndcg([1, 2], {"1": 2, "2": 1}, 2), 1)

    def test_attack_no_pair_leakage(self):
        X = np.zeros((20, 36))
        y = np.tile([0, 1], 10)
        groups = np.repeat(np.arange(10), 2)
        result, prediction = score(X, y, groups)
        self.assertEqual(result["auc"], 0.5)
        self.assertEqual(len(prediction), 20)

    def test_formats_and_cli(self):
        path = self.root / "vectors.fbin"
        write_fbin(path, self.x)
        np.testing.assert_array_equal(vectors(path), self.x)
        with self.build() as index:
            index.validate()
        output = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "expunge",
                "check",
                "--index",
                str(self.root / "index.sqlite"),
            ],
            text=True,
        )
        self.assertEqual(json.loads(output)["vertices"], 55)

    def command(self, args):
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_small_experiment_drivers(self):
        data = self.root / "data.fbin"
        queries = self.root / "queries.fbin"
        write_fbin(data, self.x)
        write_fbin(queries, self.x[:5])
        with self.build() as index:
            index.validate()
        args = [
            sys.executable,
            "-m",
            "experiments.run",
            "frontier",
            "--data",
            str(data),
            "--index",
            str(self.root / "index.sqlite"),
            "--output",
            str(self.root / "frontier.jsonl"),
            "--targets",
            "2",
            "--check",
            "2",
        ]
        self.command(args)
        rows = [
            json.loads(s)
            for s in (self.root / "frontier.jsonl").read_text().splitlines()
        ]
        self.assertTrue(
            all(r["bitwise_adjacency_equal"] and r["metadata_equal"] for r in rows)
        )
        self.assertTrue(all(r["true_frontier"] <= r["frontier"] for r in rows))
        self.command(
            [
                sys.executable,
                "-m",
                "experiments.run",
                "paired",
                "--data",
                str(data),
                "--queries",
                str(queries),
                "--index",
                str(self.root / "index.sqlite"),
                "--output",
                str(self.root / "paired"),
                "--targets",
                "5",
                "--delete-fraction",
                "0.1",
                "--methods",
                "replay,scrub",
            ]
        )
        self.command(
            [
                sys.executable,
                "-m",
                "experiments.attack",
                "--features",
                str(self.root / "paired/features.npz"),
                "--output",
                str(self.root / "auc.json"),
            ]
        )
        result = json.loads((self.root / "auc.json").read_text())
        self.assertEqual(result["methods"]["replay"]["auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
