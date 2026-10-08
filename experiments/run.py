"""Dataset-backed experiment workloads; every result is measured during this run."""

from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from expunge import Index
from expunge.data import vectors
from expunge.index import DEFAULTS
from expunge.metrics import Snapshot, distance, evaluate


def clone(source: Index, path: Path):
    import sqlite3

    destination = sqlite3.connect(path)
    source.store.db.backup(destination)
    destination.close()
    fd = os.open(str(path) + ".key", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(Path(str(source.store.path) + ".key").read_bytes())
    return Index(path)


def choose_targets(index, count, seed, strategy="stratified", labels=None):
    ids = np.asarray(
        [
            v
            for (v,) in index.store.db.execute("SELECT id FROM graph ORDER BY id")
            if not index.store.node(v).tomb
        ],
        np.int64,
    )
    if count < 1 or count > len(ids):
        raise ValueError("Target count must fit the live population")
    rng = np.random.default_rng(seed)
    if strategy == "uniform":
        return list(map(int, rng.choice(ids, count, replace=False)))
    if strategy == "clustered":
        if labels is None or len(labels) <= max(ids):
            raise ValueError(
                "Subject-clustered selection requires one supplied label per stable ID"
            )
        groups = defaultdict(list)
        for vertex in ids:
            groups[str(labels[vertex])].append(int(vertex))
        keys = list(groups)
        rng.shuffle(keys)
        selected = []
        for key in keys:
            group = groups[key]
            rng.shuffle(group)
            selected.extend(group)
            if len(selected) >= count:
                break
        return selected[:count]
    pool = rng.choice(ids, min(len(ids), max(count * 10, count)), replace=False)
    positions = {
        v: position
        for v, position in index.store.db.execute("SELECT id,inserted FROM vectors")
    }
    measurements = []
    for vertex in pool:
        payload = index.store.vector(int(vertex))
        neighbors, _ = index.query(payload, min(33, len(ids)))
        ds = distance(
            [index.store.vector(int(v)) for v in neighbors if int(v) != vertex],
            payload,
            index.config["metric"],
        )
        positive = ds[ds > 0]
        # Hill LID estimate uses Euclidean radius, rather than squared L2 distance.
        radii = np.sqrt(positive) if index.config["metric"] == "l2" else positive
        radii.sort()
        lid = (
            -len(radii) / np.log(np.maximum(radii / max(radii[-1], 1e-30), 1e-30)).sum()
            if len(radii) > 1 and radii[-1] > radii[0]
            else 0
        )
        measurements.append(
            (
                positions.get(int(vertex), int(vertex)),
                len(index.store.node(int(vertex)).edges[0]),
                lid,
            )
        )
    a = np.asarray(measurements)
    bins = np.column_stack(
        [
            np.digitize(a[:, j], np.quantile(a[:, j], [0.25, 0.5, 0.75]))
            for j in range(3)
        ]
    )
    groups = defaultdict(list)
    for vertex, key in zip(pool, bins):
        groups[tuple(key)].append(int(vertex))
    for group in groups.values():
        rng.shuffle(group)
    keys = list(groups)
    rng.shuffle(keys)
    selected = []
    while len(selected) < count:
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == count:
                    break
    return selected


def load_data(args):
    values = vectors(args.data)
    return values[: args.limit] if args.limit else values


def frozen_oracle(base, values, path, excluded):
    order = [
        v
        for _, v in base.store.db.execute(
            "SELECT seq,vertex FROM operations ORDER BY seq"
        )
        if v not in excluded
    ]
    return Index.build(path, values, base.config, base.capacity, order, base.partition)


def equality(a, b):
    differing = 0
    owners = set(v for (v,) in a.store.db.execute("SELECT id FROM graph")) | set(
        v for (v,) in b.store.db.execute("SELECT id FROM graph")
    )
    for vertex in owners:
        differing += a.graph.get(vertex) != b.graph.get(vertex)
    return dict(
        bitwise_adjacency_equal=differing == 0,
        differing_vertices=differing,
        metadata_equal=a.graph.metadata == b.graph.metadata,
    )


def true_frontier(base, oracle, target):
    changed = 0
    for seq, vertex in base.store.db.execute(
        "SELECT seq,vertex FROM operations ORDER BY seq"
    ):
        if vertex == target:
            continue
        other = oracle.store.db.execute(
            "SELECT seq FROM operations WHERE vertex=?", (vertex,)
        ).fetchone()[0]
        old, new = base.store.operation(seq), oracle.store.operation(other)
        before = {v: after for v, (_, after) in base.store.writes(seq).items()}
        after = {v: after for v, (_, after) in oracle.store.writes(other).items()}
        changed += (
            old["candidates"] != new["candidates"]
            or old["scored"] != new["scored"]
            or before != after
            or old["after"][:2] != new["after"][:2]
        )
    return changed


def frontier(args):
    values = load_data(args)
    with Index(args.index) as base:
        if base.store.get("epoch") != 0:
            raise ValueError(
                "Frontier workloads require a retained insertion-history epoch"
            )
        targets = choose_targets(
            base,
            args.targets,
            args.seed,
            args.strategy,
            np.load(args.labels, allow_pickle=False) if args.labels else None,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        records = []
        with out.open("w") as f:
            for j, target in enumerate(targets):
                with tempfile.TemporaryDirectory(dir=args.scratch) as temporary:
                    root = Path(temporary)
                    with clone(base, root / "trial.sqlite") as trial:
                        trial.config["warm_replay"] = not args.cold
                        start, cpu = time.perf_counter(), time.thread_time()
                        record = trial.delete(target, "replay", budget=False)
                        row = dict(
                            target=target,
                            wall_seconds=time.perf_counter() - start,
                            cpu_seconds=time.thread_time() - cpu,
                            frontier=record["details"]["admitted"],
                            changed_admitted=record["details"]["changed"],
                            frontier_fraction=record["details"]["admitted"]
                            / base.graph.metadata[2],
                            estimated=record["estimate"],
                            warm_hits=record["details"]["warm_hits"],
                            warm_replay=not args.cold,
                            storage=trial.stats(),
                        )
                        if j < args.check:
                            with frozen_oracle(
                                base, values, root / "oracle.sqlite", {target}
                            ) as oracle:
                                row.update(equality(trial, oracle))
                                row["true_frontier"] = true_frontier(
                                    base, oracle, target
                                )
                                row["inflation"] = (
                                    row["frontier"] / row["true_frontier"]
                                    if row["true_frontier"]
                                    else None
                                )
                                if (
                                    not row["bitwise_adjacency_equal"]
                                    or not row["metadata_equal"]
                                ):
                                    raise AssertionError(
                                        f"Replay disagrees with independent oracle for target {target}"
                                    )
                        f.write(json.dumps(row, sort_keys=True) + "\n")
                        f.flush()
                        records.append(row)
        sizes = [r["frontier_fraction"] for r in records]
        print(
            json.dumps(
                dict(
                    targets=len(records),
                    frontier_fraction_quantiles={
                        str(q): float(np.quantile(sizes, q))
                        for q in (0.5, 0.95, 0.99, 1)
                    },
                    checks=min(args.check, len(records)),
                ),
                sort_keys=True,
            )
        )


def paired(args):
    values, queries = load_data(args), vectors(args.queries)
    labels = np.load(args.labels, allow_pickle=False) if args.labels else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with Index(args.index) as base:
        if base.store.get("epoch") != 0:
            raise ValueError(
                "Paired workloads need the original retained insertion history"
            )
        targets = choose_targets(base, args.targets, args.seed, "stratified", labels)
        if args.tail_frontiers:
            previous = [
                json.loads(line)
                for line in Path(args.tail_frontiers).read_text().splitlines()
            ]
            threshold = np.quantile(
                [r["frontier_fraction"] for r in previous], args.tail_quantile
            )
            targets = [
                r["target"] for r in previous if r["frontier_fraction"] >= threshold
            ]
        all_ids = [v for (v,) in base.store.db.execute("SELECT id FROM graph")]
        if not 0 <= args.delete_fraction < 1:
            raise ValueError("Deletion fraction must lie in [0,1)")
        background_count = max(0, int(len(all_ids) * args.delete_fraction) - 1)
        background = (
            choose_targets(base, background_count, args.seed + 1, args.strategy, labels)
            if background_count
            else []
        )
        features, outcomes, groups, methods = [], [], [], []
        with (output / "measurements.jsonl").open("w") as measurements:
            for target in targets:
                candidate = base.store.vector(target).copy()
                background_ids = [v for v in background if v != target]
                if len(background_ids) < background_count:
                    background_ids += [
                        v for v in all_ids if v != target and v not in background_ids
                    ][: background_count - len(background_ids)]
                for method in args.methods.split(","):
                    with tempfile.TemporaryDirectory(dir=args.scratch) as temporary:
                        root = Path(temporary)
                        with (
                            clone(base, root / "member.sqlite") as member,
                            frozen_oracle(
                                base, values, root / "null.sqlite", {target}
                            ) as null,
                        ):
                            background_start = time.perf_counter()
                            for deleted in background_ids:
                                common_region = member.region(deleted)
                                result = member.delete(
                                    deleted, method, region=common_region, budget=False
                                )
                                null.delete(
                                    deleted,
                                    result["path"],
                                    region=common_region,
                                    budget=False,
                                )
                            background_seconds = time.perf_counter() - background_start
                            common_region = member.region(target)
                            start, cpu = time.perf_counter(), time.thread_time()
                            result = member.delete(
                                target, method, region=common_region, budget=False
                            )
                            wall, core = (
                                time.perf_counter() - start,
                                time.thread_time() - cpu,
                            )
                            if result["path"] == "scrub":
                                null.scrub_matched(common_region - {target})
                            member.validate()
                            null.validate()
                            for label, index in ((1, member), (0, null)):
                                snapshot = root / f"snapshot-{label}.npz"
                                index.export(snapshot)
                                features.append(Snapshot(snapshot).features(candidate))
                                outcomes.append(label)
                                groups.append(target)
                                methods.append(method)
                            query_scores = distance(
                                queries, candidate, base.config["metric"]
                            )
                            nearby = np.argsort(query_scores, kind="stable")[
                                : min(args.region_queries, len(queries))
                            ]
                            row = dict(
                                target=target,
                                method=method,
                                actual_path=result["path"],
                                deletion_count=len(background_ids) + 1,
                                population="tail"
                                if args.tail_frontiers
                                else "stratified",
                                wall_seconds=wall,
                                cpu_seconds=core,
                                background_seconds=background_seconds,
                                maintenance=result,
                                agreement=equality(member, null),
                                retrieval=None,
                            )
                            with frozen_oracle(
                                base,
                                values,
                                root / "counterfactual.sqlite",
                                set(background_ids) | {target},
                            ) as counterfactual:
                                qrels = (
                                    json.loads(Path(args.qrels).read_text())
                                    if args.qrels
                                    else None
                                )
                                qids = (
                                    Path(args.query_ids).read_text().splitlines()
                                    if args.query_ids
                                    else None
                                )
                                selected_qids = (
                                    [qids[j] for j in nearby]
                                    if qids
                                    else list(map(int, nearby))
                                )
                                row["retrieval"] = evaluate(
                                    member,
                                    queries[nearby],
                                    counterfactual,
                                    exact=args.exact_recall,
                                    qrels=qrels,
                                    query_ids=selected_qids,
                                )
                                if qrels:
                                    row["counterfactual_relevance"] = evaluate(
                                        counterfactual,
                                        queries[nearby],
                                        qrels=qrels,
                                        query_ids=selected_qids,
                                    )
                            measurements.write(json.dumps(row, sort_keys=True) + "\n")
                            measurements.flush()
        np.savez_compressed(
            output / "features.npz",
            X=np.asarray(features),
            y=np.asarray(outcomes),
            groups=np.asarray(groups),
            methods=np.asarray(methods),
            protocol=json.dumps(
                dict(
                    shared_build_history=True,
                    paired_target_disjoint=True,
                    delete_fraction=args.delete_fraction,
                    region_queries=args.region_queries,
                    candidate_count=len(targets),
                    population="tail" if args.tail_frontiers else "stratified",
                    observation="structural_geometric_survivor_features",
                )
            ),
        )
    print(
        json.dumps(
            dict(output=str(output), candidates=len(targets), samples=len(outcomes))
        )
    )


def stream(args):
    labels = np.load(args.labels, allow_pickle=False) if args.labels else None
    with Index(args.index) as index:
        count = int(index.graph.metadata[2] * args.delete_fraction)
        targets = choose_targets(index, count, args.seed, args.strategy, labels)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            for target in targets:
                start, cpu = time.perf_counter(), time.thread_time()
                row = index.delete(target, args.mode, budget=True)
                row["observed_wall_seconds"] = time.perf_counter() - start
                row["observed_cpu_seconds"] = time.thread_time() - cpu
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
        print(json.dumps(index.stats(), sort_keys=True))


def load(args):
    queries = vectors(args.queries)
    if not len(queries) or args.workers < 1:
        raise ValueError("Query load requires queries and positive worker count")
    stop = threading.Event()
    samples, failures = [], []
    count = [0]
    guard = threading.Lock()
    with Index(args.index) as index:
        targets = choose_targets(
            index,
            int(index.graph.metadata[2] * args.delete_fraction),
            args.seed,
            args.strategy,
            np.load(args.labels, allow_pickle=False) if args.labels else None,
        )

        def once(query):
            start = time.perf_counter()
            index.query(query, 10)
            return time.perf_counter() - start

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            baseline = list(pool.map(once, queries))

        def worker(worker_id):
            local, completed = [], 0
            rng = np.random.default_rng(args.seed + worker_id)
            try:
                while not stop.is_set():
                    elapsed = once(queries[int(rng.integers(len(queries)))])
                    completed += 1
                    if len(local) < 100000:
                        local.append(elapsed)
                    else:
                        chosen = int(rng.integers(completed))
                        if chosen < len(local):
                            local[chosen] = elapsed
            except BaseException as error:
                failures.append(error)
                stop.set()
            finally:
                with guard:
                    samples.extend(local)
                    count[0] += completed

        threads = [
            threading.Thread(target=worker, args=(j,)) for j in range(args.workers)
        ]
        for thread in threads:
            thread.start()
        records, start = [], time.perf_counter()
        try:
            for target in targets:
                record = index.delete(target, args.mode, budget=True)
                records.append(record)
        finally:
            stop.set()
            for thread in threads:
                thread.join()
        elapsed = time.perf_counter() - start
        if failures:
            raise failures[0]
        result = dict(
            workers=args.workers,
            maintenance=records,
            query_count=count[0],
            query_sample_count=len(samples),
            elapsed_seconds=elapsed,
            queries_per_second=count[0] / elapsed,
            baseline_p99_seconds=float(np.quantile(baseline, 0.99)),
            loaded_p99_seconds=float(np.quantile(samples, 0.99)) if samples else None,
            measured_cpu_seconds=sum(r["total_cpu_seconds"] for r in records),
            config=index.config,
        )
        result["p99_slowdown"] = (
            result["loaded_p99_seconds"] / result["baseline_p99_seconds"]
            if samples
            else None
        )
        Path(args.output).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "maintenance"}, sort_keys=True
            )
        )


def shards(args):
    from expunge.native import Graph

    values, queries = load_data(args), vectors(args.queries)
    cfg = dict(DEFAULTS, **json.loads(Path(args.config).read_text()))
    if args.backend:
        cfg["backend"] = args.backend
        if args.backend == "vamana":
            cfg.update(degree=64, beam=125)
    if not 1 <= args.shards <= len(values) or not len(queries):
        raise ValueError("Shards must fit the dataset, and queries must be nonempty")
    if args.metric:
        cfg["metric"] = args.metric
    groups = [np.arange(j, len(values), args.shards) for j in range(args.shards)]
    engines = [
        Graph(len(ids), values.shape[1], dict(cfg, seed=cfg["seed"] ^ j))
        for j, ids in enumerate(groups)
    ]
    try:
        start = time.perf_counter()
        for engine, ids in zip(engines, groups):
            for local, vertex in enumerate(ids):
                engine.insert(local, np.asarray(values[vertex], np.float32))
        build_seconds = time.perf_counter() - start
        rng = np.random.default_rng(args.seed)
        targets = rng.choice(len(values), min(args.targets, len(values)), replace=False)
        rows = []
        for target in targets:
            shard = int(target) % args.shards
            start, cpu = time.perf_counter(), time.thread_time()
            engine = Graph(
                len(groups[shard]), values.shape[1], dict(cfg, seed=cfg["seed"] ^ shard)
            )
            for local, vertex in enumerate(groups[shard]):
                if vertex != target:
                    engine.insert(local, np.asarray(values[vertex], np.float32))
            row = dict(
                target=int(target),
                shard=shard,
                rebuild_seconds=time.perf_counter() - start,
                rebuild_cpu_seconds=time.thread_time() - cpu,
            )
            old, engines[shard] = engines[shard], engine
            start = time.perf_counter()
            for query in queries:
                results = []
                for j, graph in enumerate(engines):
                    ids, ds = graph.query(query, 10, cfg["query_width"])
                    results.extend(zip(map(float, ds), map(int, groups[j][ids])))
                sorted(results)[:10]
            row["fanout_query_seconds"] = (time.perf_counter() - start) / len(queries)
            engines[shard] = old
            engine.close()
            rows.append(row)
        Path(args.output).write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
        )
        print(
            json.dumps(
                dict(
                    shards=args.shards,
                    build_seconds=build_seconds,
                    independent_deletions=len(rows),
                )
            )
        )
    finally:
        for graph in engines:
            graph.close()


def sweep(args):
    values = load_data(args)
    queries = vectors(args.queries) if args.queries else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS, **json.loads(Path(args.config).read_text()))
    if args.backend == "vamana":
        cfg.update(backend="vamana", degree=64, beam=125)
    elif args.backend:
        cfg["backend"] = args.backend
    if args.metric:
        cfg["metric"] = args.metric
    for cells in map(int, args.cells.split(",")):
        base_path = output / f"cells-{cells}.sqlite"
        with Index.build(base_path, values, dict(cfg, cells=cells)) as base:
            targets = choose_targets(
                base,
                args.targets,
                args.seed,
                args.strategy,
                np.load(args.labels, allow_pickle=False) if args.labels else None,
            )
        current = argparse.Namespace(**vars(args))
        current.index = str(base_path)
        current.output = str(output / f"frontiers-{cells}.jsonl")
        frontier(current)
        if cells == cfg["cells"]:
            with Index(base_path) as base, (output / "scrub-hops.jsonl").open("w") as f:
                for hops in map(int, args.hops.split(",")):
                    for target in targets:
                        with tempfile.TemporaryDirectory(dir=args.scratch) as temporary:
                            root = Path(temporary)
                            with clone(base, root / "trial.sqlite") as trial:
                                trial.config["hops"] = hops
                                start, cpu = time.perf_counter(), time.thread_time()
                                record = trial.delete(target, "scrub", budget=False)
                                row = dict(
                                    target=target,
                                    hops=hops,
                                    region_size=record["details"]["region_size"],
                                    wall_seconds=time.perf_counter() - start,
                                    cpu_seconds=time.thread_time() - cpu,
                                )
                                if queries is not None:
                                    with frozen_oracle(
                                        base, values, root / "oracle.sqlite", {target}
                                    ) as oracle:
                                        nearby = np.argsort(
                                            distance(
                                                queries,
                                                base.store.vector(target),
                                                cfg["metric"],
                                            ),
                                            kind="stable",
                                        )[: args.region_queries]
                                        row["retrieval"] = evaluate(
                                            trial, queries[nearby], oracle
                                        )
                                f.write(json.dumps(row, sort_keys=True) + "\n")
                                f.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("frontier", "paired", "stream", "shards", "load"):
        p = sub.add_parser(name)
        if name not in ("stream", "load"):
            p.add_argument("--data", required=True)
            p.add_argument("--limit", type=int)
        if name != "shards":
            p.add_argument("--index", required=True)
            p.add_argument("--labels")
            p.add_argument(
                "--strategy",
                choices=["uniform", "clustered", "stratified"],
                default="uniform",
            )
        p.add_argument("--output", required=True)
        p.add_argument("--seed", type=int, default=42)
        if name in ("frontier", "paired"):
            p.add_argument("--scratch")
            p.add_argument("--targets", type=int, default=2000)
        if name == "frontier":
            p.add_argument("--check", type=int, default=200)
            p.add_argument("--cold", action="store_true")
        if name in ("paired", "shards", "load"):
            p.add_argument("--queries", required=True)
        if name == "paired":
            p.add_argument(
                "--methods",
                default="replay,scrub,consolidate,inplace,randomwalk,tombstone",
            )
            p.add_argument("--delete-fraction", type=float, default=0.10)
            p.add_argument("--region-queries", type=int, default=100)
            p.add_argument("--tail-frontiers")
            p.add_argument("--tail-quantile", type=float, default=0.99)
            p.add_argument("--exact-recall", action="store_true")
            p.add_argument("--qrels")
            p.add_argument("--query-ids")
        if name in ("stream", "load"):
            p.add_argument("--delete-fraction", type=float, default=0.10)
            p.add_argument(
                "--mode",
                choices=[
                    "governor",
                    "replay",
                    "scrub",
                    "consolidate",
                    "inplace",
                    "randomwalk",
                    "tombstone",
                ],
                default="governor",
            )
        if name == "load":
            p.add_argument("--workers", type=int, default=16)
        if name == "shards":
            p.add_argument("--shards", type=int, default=32)
            p.add_argument("--targets", type=int, default=200)
            p.add_argument("--config", default="configs/paper.json")
            p.add_argument("--backend", choices=["hnsw", "vamana"])
            p.add_argument("--metric", choices=["l2", "cosine"])
    p = sub.add_parser("sweep")
    p.add_argument("--data", required=True)
    p.add_argument("--queries")
    p.add_argument("--limit", type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/paper.json")
    p.add_argument("--backend", choices=["hnsw", "vamana"])
    p.add_argument("--metric", choices=["l2", "cosine"])
    p.add_argument("--cells", default="512,2048,8192,32768")
    p.add_argument("--hops", default="1,2,3,4")
    p.add_argument("--targets", type=int, default=2000)
    p.add_argument("--check", type=int, default=200)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--strategy",
        choices=["uniform", "clustered", "stratified"],
        default="stratified",
    )
    p.add_argument("--labels")
    p.add_argument("--scratch")
    p.add_argument("--region-queries", type=int, default=100)
    args = parser.parse_args()
    if getattr(args, "scratch", None):
        Path(args.scratch).mkdir(parents=True, exist_ok=True)
    globals()[args.command](args)


if __name__ == "__main__":
    main()
