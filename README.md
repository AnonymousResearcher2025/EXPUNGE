# EXPUNGE

EXPUNGE maintains HNSW and Vamana proximity graphs with logged insertion dependencies. Exact deletion suppresses an insertion and selectively re-executes dependent operations against corrected historical prefixes. A local scrub rebuilds a region in canonical order. A CPU-work governor chooses a path and can discard an expensive private replay before scrubbing. Queries continue against the committed graph during maintenance.

## Layout and requirements

- `native/`: C++17 graph engines and read instrumentation.
- `expunge/`: typed native interface, frozen partition, SQLite history, maintenance, CLI, and metrics.
- `experiments/`: dataset-backed frontier, attack, sweep, deletion-stream, concurrent-query, and sharding workloads.
- `tests/`: oracle equality, persistence, rollback, concurrent queries, signatures, and end-to-end workflows.
- `configs/paper.json`: paper construction parameters and explicit workload defaults.
- `third_party/hnswlib/`: hnswlib v0.8.0 headers and upstream license. Changes add stable slots, fixed levels including level zero, read hooks, sparse-slot cleanup, and per-query search width.

Requires Python 3.12+, a C++17 compiler, Make, and the packages in `requirements.txt`. Linux and macOS are supported build targets; verification was performed on macOS ARM64. Run commands from this directory. The native library is built locally; no package download is needed for hnswlib.

## Build and run

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make
make test PYTHON=python

python -m expunge build --data /data/sift/base.fvecs --index /data/runs/sift.sqlite --config configs/paper.json
python -m expunge query --index /data/runs/sift.sqlite --queries /data/sift/query.fvecs --output /data/runs/results.npz
python -m expunge delete --index /data/runs/sift.sqlite --id 120 --mode governor
python -m expunge check --index /data/runs/sift.sqlite
python -m expunge verify-records --index /data/runs/sift.sqlite
```

Use `--backend vamana` for R=64, L=125, alpha=1.2; HNSW defaults are M=32 and efConstruction=400. Use `--metric cosine` for normalized E5 embeddings. Vector inputs are `.fvecs`, `.bvecs`, `.fbin`, or numeric `.npy` matrices. A `.fbin` begins with little-endian uint32 row count and dimension, followed by row-major float32 values. Stable IDs are original input row numbers; deleted IDs are never reused.

Reserve capacity when building a growing index, for example `--capacity 1200000`. Insert a new vector with:

```sh
python -m expunge insert --index /data/runs/sift.sqlite --id 1000000 --data /data/new.fbin --row 0
```

Deletion modes are `replay`, `scrub`, `governor`, `full`, `tombstone`, `consolidate`, `inplace`, and `randomwalk`. Tombstones retain their payload and routing vertex. A subsequent `consolidate` or `scrub` can physically remove a masked vertex. `--no-pacing` disables governor CPU pacing for an isolated timing trial. Back up the database together with its adjacent `.key` file; the Ed25519 signing key is created with owner-only permissions. Signed records authenticate declared transitions, not an independent proof of deletion correctness.

`export --index PATH --output SNAPSHOT.npz` exports graph state and current payloads without history or signing keys. Tombstone exports still contain masked payloads. The structural attack feature extractor ignores masked vertices; it does not measure leakage from explicitly retained payloads.

## Experiments

Supply the actual SIFT-1M (128d), Wiki-1M E5-small-v2 (384d), and DEEP-10M (96d) vectors and corresponding queries. Historical raw measurements, the exact Wiki corpus/sample, subject labels, ID mappings, and complete query settings were not supplied with the paper. The commands below execute its workload families and produce new measured outputs; they cannot recover or guarantee its published numbers. No reported result is embedded in the implementation. Full workloads require substantial time, memory, and disk space, especially paired counterfactual builds and version logs.

Build independent indexes for each dataset/backend as above. For Wiki, add `--metric cosine`; for DEEP use its supplied `.fbin` or `.fvecs`. `stats` reports actual bitmap, candidate-context, version, and database footprints.

Replay frontiers on the full dataset, and the paper's 200 equality checks on a 100k prefix:

```sh
python -m experiments.run frontier --data /data/wiki/base.fbin --index /data/runs/wiki.sqlite --targets 2000 --check 0 --strategy stratified --output /data/runs/frontiers.jsonl
python -m expunge build --data /data/wiki/base.fbin --limit 100000 --metric cosine --index /data/runs/wiki100k.sqlite --config configs/paper.json
python -m experiments.run frontier --data /data/wiki/base.fbin --limit 100000 --index /data/runs/wiki100k.sqlite --targets 200 --check 200 --strategy stratified --output /data/runs/equality.jsonl
```

Paired attack and deleted-region retrieval agreement, with 10% total deletion and supplied subject labels:

```sh
python -m experiments.run paired --data /data/wiki/base.fbin --queries /data/wiki/query.fbin --index /data/runs/wiki.sqlite --labels /data/wiki/subjects.npy --strategy clustered --delete-fraction 0.10 --targets 2000 --output /data/runs/paired
python -m experiments.attack --features /data/runs/paired/features.npz --output /data/runs/paired/auc.json
```

For uniform deletion use `--strategy uniform` and omit labels. Labels must be a non-object NumPy array indexed by stable ID. Paired branches share the frozen partition and background deletion schedule; the separately stratified candidate is deleted last, and the total cohort is 10%. Scrubs use a common region. This last-target protocol makes the unspecified event order explicit. Retrieval disagreement is evaluated against a separate full retained-history oracle. The 32 nearest surviving vertices supply degree/fill, edge-length, pruning-violation, clustering, reverse-asymmetry, and two-hop bypass features, summarized by mean, standard deviation, minimum, and maximum. Boosted trees use five target-disjoint folds and target-bootstrap intervals. Targets share a base history; this protocol does not establish generalization to unseen graph histories.

For a matched tail population, add `--tail-frontiers /data/runs/frontiers.jsonl --tail-quantile 0.99 --methods scrub,consolidate,inplace,randomwalk` to the paired command, then score its separate feature file. The quantile is configurable: it selects an empirical frontier tail and is not automatically the governor-routed population. Keep population results separate.

Bitmap and scrub-hop sweeps, 32-shard rebuilding, and paced maintenance under 16 real query threads:

```sh
python -m experiments.run sweep --data /data/wiki/base.fbin --queries /data/wiki/query.fbin --config configs/paper.json --metric cosine --targets 2000 --check 0 --output /data/runs/sweep
python -m experiments.run shards --data /data/wiki/base.fbin --queries /data/wiki/query.fbin --config configs/paper.json --metric cosine --targets 200 --output /data/runs/shards.jsonl
python -m experiments.run load --index /data/runs/wiki.sqlite --queries /data/wiki/query.fbin --workers 16 --delete-fraction 0.10 --strategy uniform --output /data/runs/load.json
```

`load` and `stream` mutate their input index; use a fresh independent build for each run. `stream` has the same deletion arguments and writes per-deletion JSONL without concurrent queries. `frontier`, `paired`, and `sweep` use independent private trials and do not delete from the supplied base index. Their temporary trials are removed automatically. Use `--scratch /nvme/scratch` to place those trials on the intended storage device. Shards use stable row-ID round-robin assignment. Queries fan out to independent native graphs and merge distances; its timings describe that workload, not a distributed service.

For BEIR/NQ relevance, supply JSON qrels mapping query IDs to `{stable_document_id: relevance}` and a text file mapping query rows to query IDs. Add `--qrels PATH --query-ids PATH` to `paired`, or evaluate an index directly:

```sh
python -m expunge evaluate --index /data/runs/wiki.sqlite --queries /data/wiki/query.fbin --qrels /data/wiki/qrels.json --query-ids /data/wiki/query_ids.txt
```

Add `--exact` to `evaluate` or `--exact-recall` to `paired` for brute-force recall. Relevance, exact recall, and ANN result disagreement are separate metrics. Agreement requires both results to contain k items; incomplete pairs are counted explicitly. The qrels ideal uses supplied judgments, including relevant documents deleted from the corpus.

## Configuration and specification choices

`degree`, `beam`, `alpha`, `cells`, and `hops` control construction, provenance granularity, and scrub region size. `cores` should equal the allocated hardware core count. `cpu_fraction * cores` is the maintenance CPU refill rate, with a one-second burst; pacing includes discovery, replay, rollback, boundary repair, and persistence. `probe` bounds the initial posting estimate; actual transitive replay can exceed it. Costs are learned from observed insertion/replay CPU time. The estimator is a heuristic, not a conservative transitive-work bound. Record-creation timings exclude the final transaction commit; returned total timings and experiment timings include it.

Important choices where the draft does not provide an executable specification:

- Mutations have a serialized deterministic order; native queries run concurrently. HNSW levels derive from seed and stable ID. Indexes bind the native binary fingerprint; rebuild an index when its binary changes. Vamana uses the first retained insertion as its initial entry point and squared L2 in robust pruning. No claim is made to reproduce an undocumented parallel build schedule.
- Warm replay reuses immutable query-to-candidate distances along the normal corrected traversal. It never seeds a stale search beam. Use `frontier --cold` for a paired cold timing run with the same targets. No fivefold speedup is assumed.
- Scrubs preserve old outgoing exterior connections as candidates, and re-prune incoming exterior lists against regional candidates. Boundary pruning can alter internal choices. Only the isolated rebuild stage is canonical; its digest is recorded separately.
- A scrub or repair establishes a checkpoint and starts a new insertion epoch. Exact replay is available for insertions after that checkpoint; older vertices require scrub. `full` requires an insertion-history epoch. This avoids asserting exact replay of the original history after an unlogged repair.
- Consolidation uses neighbor bypass and pruning. In-place repair implements candidate-based edge copies and immediate dangling-edge cleanup. `randomwalk` implements SPatch's deterministic star-mesh top-weight sparsification per layer, with degree-bound pruning. These are adaptations to these backends, not the original baseline executables. `inplace_width/candidates/copies` and `spatch_r/alpha` are explicit parameters. See [in-place algorithms](https://arxiv.org/html/2502.13826v1) and [SPatch](https://arxiv.org/html/2512.18060v1).
- Seed 42, a 65,536-vector partition sample, query width 100, 100 nearby evaluation queries, feature aggregation, and repair-specific settings are explicit defaults where the draft omits details. Subject clustering requires supplied labels; geometric cells are not substituted for subjects.
- SQLite WAL stores complete before/after adjacency, version chains, bitmaps, ordered scored candidates, distance context, and cell postings. Read hooks conservatively include complete touched adjacency and can admit extra operations. Historical nodes and retained-prefix counts are materialized lazily, avoiding a suffix metadata rewrite. An incremental Merkle digest covers current graph/payload state. Python handles journal and maintenance orchestration around native C++ graph operations; timings and storage overhead must be measured on the target hardware.

Physical removal clears live payload rows and native slots. History remains operator-visible and can identify prior members. Frozen centroids are data-dependent. Input files, SQLite journals, storage media, and backups are outside secure-erasure claims. The test fixtures are explicitly synthetic correctness inputs, not paper experiments.
