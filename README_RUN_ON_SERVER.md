# Run On Server Later

This repository only prepares patches, scripts, and documentation. Run the actual experiment later on a server that is allowed to build and run Milvus.

## Preparation

1. Clone Milvus on the server.
2. Check out the Milvus branch or commit you want to profile.
3. Apply the patches from this scaffold:

```bash
git apply patches/milvus_filter_generation_profile.patch
git apply patches/milvus_expr_cache_profile.patch
git apply patches/milvus_search_profile.patch
```

If a patch does not apply cleanly, use the TODO notes in the patch and README to adjust the nearby source manually.

If the unified diff context has drifted on the checked-out Milvus revision, use
the fallback source-editing script from the root of the Milvus checkout:

```bash
python3 ../FILTERRED-ANNS/scripts/apply_milvus_profile_scaffold.py --milvus-root .
```

Then inspect the resulting `git diff` before building.

## Build

Use the official Milvus build instructions for the chosen checkout. This may require Docker, Go, CMake, Conan, compiler toolchains, or other dependencies depending on the Milvus version and build path.

This Codex task did not run a build.

## Experiment Shape

Create or load a production-like collection with:

- IVF_PQ vector index
- scalar metadata fields such as `camera`, `country`, `year`, and `tag`
- scalar indexes where appropriate
- representative filtered search predicates

The goal is to measure scalar predicate evaluation and filter bitset/allowlist materialization overhead, not a no-filter vs with-filter latency comparison.

## Enable Profiling

Set:

```bash
export MILVUS_FILTER_PROFILE=1
```

Then run the filtered search workload on the server.

Expected logs:

- `/tmp/milvus_filter_profile.csv`
- `/tmp/milvus_expr_cache_profile.csv`
- `/tmp/milvus_search_profile.csv`

## Analyze Logs

Example:

```bash
python scripts/analyze_milvus_filter_profile.py \
  --filter-profile /tmp/milvus_filter_profile.csv \
  --search-profile /tmp/milvus_search_profile.csv \
  --expr-cache-profile /tmp/milvus_expr_cache_profile.csv \
  --summary-csv /tmp/milvus_filter_profile_summary.csv
```

For IVF-list-local hot filter analysis:

```bash
python scripts/collect_cluster_hot_filters.py \
  --trace-csv query_trace.csv \
  --top-k 5 \
  --output-dir /tmp/milvus_cluster_hot_filters
```

Trace input columns:

```text
query_id,predicate_type,predicate_value,probed_list_ids
```

For global filter/cache waste simulation:

```bash
python scripts/simulate_filter_cache_waste.py \
  --total-database-size 100000000 \
  --ivf-list-sizes ivf_list_sizes.csv \
  --trace-csv query_trace.csv \
  --output-dir /tmp/milvus_filter_cache_waste
```

## Permissions That May Be Needed Later

Depending on how Milvus is built and run on the server, the later experiment may require:

- Docker permission for the official Milvus build or deployment path
- sudo permission for system packages or service setup
- write access to `/tmp`
- access to the dataset storage path
- perf permission only if you separately choose to run perf profiling

None of those actions were performed in this scaffold task.
