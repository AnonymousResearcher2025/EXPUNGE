from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .data import vectors
from .index import Index
from .metrics import evaluate
from .storage import Store


def main():
    parser = argparse.ArgumentParser(
        description="EXPUNGE vector-index construction and maintenance"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--data", required=True)
    build.add_argument("--index", required=True)
    build.add_argument("--config")
    build.add_argument("--backend", choices=["hnsw", "vamana"])
    build.add_argument("--metric", choices=["l2", "cosine"])
    for name in (
        "degree",
        "beam",
        "cells",
        "partition-sample",
        "seed",
        "capacity",
        "limit",
    ):
        build.add_argument("--" + name, type=int)
    query = sub.add_parser("query")
    query.add_argument("--index", required=True)
    query.add_argument("--queries", required=True)
    query.add_argument("-k", type=int, default=10)
    query.add_argument("--width", type=int)
    query.add_argument("--output", required=True)
    insert = sub.add_parser("insert")
    insert.add_argument("--index", required=True)
    insert.add_argument("--id", required=True, type=int)
    insert.add_argument("--data", required=True)
    insert.add_argument("--row", type=int, default=0)
    delete = sub.add_parser("delete")
    delete.add_argument("--index", required=True)
    delete.add_argument("--id", type=int, required=True)
    delete.add_argument(
        "--mode",
        choices=[
            "governor",
            "replay",
            "scrub",
            "tombstone",
            "consolidate",
            "inplace",
            "randomwalk",
            "full",
        ],
        default="governor",
    )
    delete.add_argument("--no-pacing", action="store_true")
    for name in ("stats", "check", "verify-records"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--index", required=True)
    export = sub.add_parser("export")
    export.add_argument("--index", required=True)
    export.add_argument("--output", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--index", required=True)
    ev.add_argument("--queries", required=True)
    ev.add_argument("--oracle")
    ev.add_argument("--exact", action="store_true")
    ev.add_argument("--qrels")
    ev.add_argument("--query-ids")
    args = parser.parse_args()
    if args.command == "build":
        cfg = json.loads(Path(args.config).read_text()) if args.config else {}
        if args.backend == "vamana":
            cfg.update(degree=64, beam=125)
        for name in (
            "backend",
            "metric",
            "degree",
            "beam",
            "cells",
            "partition_sample",
            "seed",
        ):
            value = getattr(args, name)
            if value is not None:
                cfg[name] = value
        x = vectors(args.data)
        if args.limit:
            x = x[: args.limit]
        with Index.build(args.index, x, cfg, args.capacity) as index:
            print(json.dumps(index.stats(), sort_keys=True))
    elif args.command == "verify-records":
        store = Store(args.index)
        try:
            key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(store.get("public_key"))
            )
            previous, count = None, 0
            for raw, signature in store.db.execute(
                "SELECT record,signature FROM maintenance ORDER BY seq"
            ):
                key.verify(signature, raw)
                record = json.loads(raw)
                # Inserts can occur between deletion records; each record authenticates its own transition.
                previous = record["after_digest"]
                count += 1
            print(json.dumps(dict(verified=count, last_declared_digest=previous)))
        finally:
            store.close()
    else:
        with Index(args.index) as index:
            if args.command == "query":
                results = [
                    index.query(q, args.k, args.width) for q in vectors(args.queries)
                ]
                lengths = np.asarray([len(ids) for ids, _ in results], np.uint32)
                ids = np.full((len(results), args.k), (1 << 32) - 1, np.uint32)
                ds = np.full((len(results), args.k), np.inf, np.float32)
                for j, (row, distances) in enumerate(results):
                    ids[j, : len(row)] = row
                    ds[j, : len(row)] = distances
                np.savez_compressed(args.output, ids=ids, distances=ds, lengths=lengths)
            elif args.command == "insert":
                index.insert(args.id, vectors(args.data)[args.row])
                print(json.dumps(index.validate()))
            elif args.command == "delete":
                print(
                    json.dumps(
                        index.delete(args.id, args.mode, budget=not args.no_pacing),
                        sort_keys=True,
                    )
                )
            elif args.command == "stats":
                print(json.dumps(index.stats(), sort_keys=True))
            elif args.command == "check":
                print(json.dumps(index.validate(), sort_keys=True))
            elif args.command == "export":
                index.export(args.output)
            elif args.command == "evaluate":
                oracle = Index(args.oracle) if args.oracle else None
                try:
                    qrels = (
                        json.loads(Path(args.qrels).read_text()) if args.qrels else None
                    )
                    qids = (
                        Path(args.query_ids).read_text().splitlines()
                        if args.query_ids
                        else None
                    )
                    print(
                        json.dumps(
                            evaluate(
                                index,
                                vectors(args.queries),
                                oracle,
                                exact=args.exact,
                                qrels=qrels,
                                query_ids=qids,
                            ),
                            sort_keys=True,
                        )
                    )
                finally:
                    if oracle:
                        oracle.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        print(f"EXPUNGE: {error}", file=sys.stderr)
        sys.exit(1)
