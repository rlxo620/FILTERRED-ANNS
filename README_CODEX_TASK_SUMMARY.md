# Codex Task Summary

This scaffold was prepared for `rlxo620/FILTERRED-ANNS` under the requested constraints:

- Do not push to `main`.
- Work should target branch `milvus-filter-profile-scaffold`.
- Do not run Milvus.
- Do not run Docker.
- Do not download YFCC or any large dataset.
- Do not run long build or test jobs.
- Do not start a server.
- Do not run perf or connect to a server.
- Only create static patches, Python scripts, and README documentation.

## GitHub Status

Best-effort GitHub writes were attempted only against branch `milvus-filter-profile-scaffold`.

The GitHub connector reported write failures:

- creating a branch ref returned `403 Resource not accessible by integration`
- updating a branch ref returned `403 Resource not accessible by integration`
- creating a file on `milvus-filter-profile-scaffold` returned `403 Resource not accessible by integration`

Because of that, no commit hash or PR URL could be created from this environment. `main` was not modified by these attempts.

The scaffold is therefore saved locally as a diff/artifact folder under:

```text
outputs/milvus-filter-profile-scaffold
```

## Source Inspection Basis

Public Milvus repository inspected:

- `milvus-io/milvus`
- default branch observed: `master`

Relevant paths found:

- `internal/core/src/exec/operator/FilterBitsNode.cpp`
- `internal/core/src/exec/operator/FilterBitsNode.h`
- `internal/core/src/exec/operator/VectorSearchNode.cpp`
- `internal/core/src/query/SearchOnSealed.cpp`
- `internal/core/src/exec/expression/ExprCache.h`
- `internal/core/src/exec/expression/ExprCache.cpp`

`ExprCacheManager` was not found by exact name. The closest equivalent found was `ExprResCacheManager`.

## Generated Files

- `patches/milvus_filter_generation_profile.patch`
- `patches/milvus_expr_cache_profile.patch`
- `patches/milvus_search_profile.patch`
- `scripts/analyze_milvus_filter_profile.py`
- `scripts/collect_cluster_hot_filters.py`
- `scripts/simulate_filter_cache_waste.py`
- `scripts/apply_milvus_profile_scaffold.py`
- `README_MILVUS_FILTER_PROFILE.md`
- `README_RUN_ON_SERVER.md`
- `README_CODEX_TASK_SUMMARY.md`

## Remaining TODOs

- Apply these files to branch `milvus-filter-profile-scaffold` from an environment with working GitHub write credentials.
- Verify patch application against the exact Milvus server checkout.
- If `git apply` fails because Milvus context drifted, run `scripts/apply_milvus_profile_scaffold.py` from the Milvus checkout and inspect `git diff`.
- Confirm bitmap cardinality APIs such as `count()` on `TargetBitmap` and `TargetBitmapView`.
- Decide whether `total_search_us` should be vector node time or full query plan time.
- If required, propagate predicate text from `FilterBitsNode` to `VectorSearchNode` through `QueryContext`.
- Replace `search_params_.dump()` with a precise Milvus index type accessor if available.
