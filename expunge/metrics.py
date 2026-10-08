"""Measured retrieval agreement, exact recall, relevance, and graph attack features."""

from __future__ import annotations
import json
import time
import numpy as np
from .native import Node


def distance(values, query, metric):
    x = np.asarray(values, np.float32)
    q = np.asarray(query, np.float32)
    if metric == "cosine":
        denom = np.linalg.norm(x, axis=1) * np.linalg.norm(q)
        return np.maximum(0, 1 - x @ q / np.maximum(denom, np.finfo(np.float32).tiny))
    return np.sum((x - q) ** 2, axis=1)


def exact_topk(index, query, k, chunk=8192):
    best = []
    rows = index.store.db.execute(
        "SELECT vectors.id,vectors.value FROM vectors JOIN graph ON vectors.id=graph.id ORDER BY vectors.id"
    )
    batch = []

    def consume(batch):
        ids = [v for v, _ in batch if not index.store.node(v).tomb]
        values = [
            np.frombuffer(raw, dtype="<f4")
            for v, raw in batch
            if not index.store.node(v).tomb
        ]
        if ids:
            ds = distance(values, query, index.config["metric"])
            best.extend((float(d), v) for d, v in zip(ds, ids))
            best.sort()
            del best[k:]

    for row in rows:
        batch.append(row)
        if len(batch) == chunk:
            consume(batch)
            batch = []
    consume(batch)
    return np.asarray([v for _, v in best], np.uint32)


def ndcg(ids, judgments, k=10):
    gains = np.asarray([judgments.get(str(int(v)), 0) for v in ids[:k]], float)
    ideal = np.asarray(sorted(judgments.values(), reverse=True)[:k], float)
    dcg = np.sum((2**gains - 1) / np.log2(np.arange(len(gains)) + 2))
    idcg = np.sum((2**ideal - 1) / np.log2(np.arange(len(ideal)) + 2))
    return float(dcg / idcg) if idcg else 0.0


def evaluate(
    index, queries, oracle=None, k=10, exact=False, qrels=None, query_ids=None
):
    if k < 1 or not len(queries):
        raise ValueError("Evaluation requires queries and positive k")
    disagreement, recalls, relevance, latency = [], [], [], []
    incomplete = 0
    for j, query in enumerate(queries):
        start = time.perf_counter()
        ids, _ = index.query(query, k)
        latency.append(time.perf_counter() - start)
        if oracle is not None:
            expected, _ = oracle.query(query, k)
            if len(ids) == k and len(expected) == k:
                disagreement.append(
                    1 - len(set(map(int, ids)) & set(map(int, expected))) / k
                )
            else:
                incomplete += 1
        if exact:
            truth = exact_topk(index, query, k)
            if len(truth):
                recalls.append(
                    len(set(map(int, ids)) & set(map(int, truth))) / len(truth)
                )
        if qrels is not None:
            query_id = str(query_ids[j]) if query_ids is not None else str(j)
            if query_id not in qrels:
                raise ValueError(f"No relevance judgments for query {query_id}")
            relevance.append(ndcg(ids, qrels[query_id], k))
    return dict(
        queries=len(queries),
        k=k,
        result_disagreement=float(np.mean(disagreement)) if disagreement else None,
        incomplete_result_pairs=incomplete,
        exact_recall=float(np.mean(recalls)) if recalls else None,
        ndcg=float(np.mean(relevance)) if relevance else None,
        query_median_seconds=float(np.median(latency)),
        query_p99_seconds=float(np.quantile(latency, 0.99)),
    )


class Snapshot:
    """Attack observation: graph state and surviving payloads, excluding maintenance logs."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as raw:
            self.ids, self.values = raw["ids"].copy(), raw["vectors"].copy()
            self.config = json.loads(str(raw["config"]))
            self.nodes = {}
            for j, v in enumerate(self.ids):
                self.nodes[int(v)] = Node.unpack(
                    raw["states"][
                        int(raw["offsets"][j]) : int(raw["offsets"][j + 1])
                    ].tobytes()
                )
        self.positions = {int(v): j for j, v in enumerate(self.ids)}

    def vector(self, vertex):
        return self.values[self.positions[vertex]]

    def features(self, candidate, nearest=32):
        eligible = np.asarray(
            [j for j, v in enumerate(self.ids) if not self.nodes[int(v)].tomb], int
        )
        if not len(eligible):
            raise ValueError("Attack snapshot contains no live survivors")
        distances = distance(self.values[eligible], candidate, self.config["metric"])
        near = eligible[np.lexsort((self.ids[eligible], distances))[:nearest]]
        rows = []
        for position in near:
            owner = int(self.ids[position])
            node = self.nodes[owner]
            neighbors = [
                v
                for v in node.edges[0]
                if v in self.positions and not self.nodes[v].tomb
            ]
            neighbor_set = set(neighbors)
            lengths = (
                distance(
                    [self.vector(v) for v in neighbors],
                    self.vector(owner),
                    self.config["metric"],
                )
                if neighbors
                else np.array([])
            )
            asymmetric = sum(owner not in self.nodes[v].edges[0] for v in neighbors)
            links = sum(
                b in self.nodes[a].edges[0]
                for a in neighbors
                for b in neighbors
                if a != b
            )
            violations = 0
            ordered = sorted(
                neighbors,
                key=lambda v: (
                    float(
                        distance(
                            [self.vector(v)], self.vector(owner), self.config["metric"]
                        )[0]
                    ),
                    v,
                ),
            )
            for j, v in enumerate(ordered):
                d_owner = float(
                    distance(
                        [self.vector(v)], self.vector(owner), self.config["metric"]
                    )[0]
                )
                for u in ordered[:j]:
                    d_pair = float(
                        distance(
                            [self.vector(v)], self.vector(u), self.config["metric"]
                        )[0]
                    )
                    if (
                        d_pair < d_owner
                        if self.config["backend"] == "hnsw"
                        else self.config["alpha"] * d_pair <= d_owner
                    ):
                        violations += 1
                        break
            twohop = {
                v
                for u in neighbors
                for v in self.nodes[u].edges[0]
                if v in self.positions and v != owner and not self.nodes[v].tomb
            } - neighbor_set
            bypass = []
            median_length = float(np.median(lengths)) if len(lengths) else 0.0
            for a in sorted(twohop):
                for b in self.nodes[a].edges[0]:
                    if b in twohop and a < b:
                        bypass.append(
                            float(
                                distance(
                                    [self.vector(b)],
                                    self.vector(a),
                                    self.config["metric"],
                                )[0]
                            )
                        )
            bound = self.config["degree"] * (
                2 if self.config["backend"] == "hnsw" else 1
            )
            rows.append(
                [
                    len(neighbors),
                    len(neighbors) / bound,
                    float(np.mean(lengths)) if len(lengths) else 0,
                    float(np.std(lengths)) if len(lengths) else 0,
                    float(np.max(lengths)) if len(lengths) else 0,
                    violations / max(1, len(neighbors)),
                    links / max(1, len(neighbors) * (len(neighbors) - 1)),
                    asymmetric / max(1, len(neighbors)),
                    sum(length > 2 * median_length for length in bypass)
                    / max(1, len(bypass)),
                ]
            )
        a = np.asarray(rows)
        return np.concatenate(
            (
                np.mean(a, axis=0),
                np.std(a, axis=0),
                np.min(a, axis=0),
                np.max(a, axis=0),
            )
        )
