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

## Project Files

- **Required:** `expunge/`, `native/graph.cpp`, `third_party/hnswlib/`, `Makefile`, `requirements.txt`.
- **Configuration:** `configs/paper.json`.
- **Optional for runtime:** `experiments/`, `tests/`, `README.md`, `.gitignore`.
- **Generated:** `build/`, `.venv/`, and Python caches; exclude these from repository uploads.

Keep index databases and their adjacent `.key` files together.

## Configuration and Limits

Configure graph degree, construction beam, cells, scrub hops, seed, and CPU budget in `configs/paper.json`.

Exact replay applies to insertions after the latest repair checkpoint. Physical deletion clears live payloads; historical journals and backups are outside secure-erasure guarantees. Rebuild indexes when the native binary changes.
