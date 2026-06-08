# Milvus Filter Generation Profile

This scaffold prepares static patches and analysis scripts for measuring query-time scalar filter materialization overhead in Milvus filtered IVF-PQ search. It does not compare no-filter latency against with-filter latency. The target signal is the time spent converting a scalar predicate into a bitset or allowlist used by vector search.

No Milvus server, Docker container, dataset download, long build, server access, or perf command was run while creating this scaffold.

## Inspected Milvus Paths

Public source inspected on 2026-06-06:

- `milvus-io/milvus`, default branch `master`
- `internal/core/src/exec/operator/FilterBitsNode.cpp`
- `internal/core/src/exec/operator/FilterBitsNode.h`
- `internal/core/src/exec/operator/VectorSearchNode.cpp`
- `internal/core/src/query/SearchOnSealed.cpp`
- `internal/core/src/exec/expression/ExprCache.h`
- `internal/core/src/exec/expression/ExprCache.cpp`

The most likely filter generation path is `PhyFilterBitsNode::GetOutput()`. The vector search handoff path is `PhyVectorSearchNode::GetOutput()`, where a `TargetBitmapView` becomes a `milvus::BitsetView` and is passed to `segment_->vector_search(...)`.

`ExprCacheManager` was not found by exact name. The equivalent path found in the inspected source is `ExprResCacheManager`, used by `FilterBitsNode` for cached expression result bitsets.

## Patch Files

- `patches/milvus_filter_generation_profile.patch`
  - Adds optional filter bitset generation logging.
  - Output: `/tmp/milvus_filter_profile.csv`
  - Columns: `timestamp,profile_id,segment_id,predicate_string,segment_rows,filter_gen_us,bitset_bytes,matched_count`

- `patches/milvus_expr_cache_profile.patch`
  - Instruments `ExprResCacheManager` lookup, hit, miss, insert, and evict events.
  - Output: `/tmp/milvus_expr_cache_profile.csv`
  - Columns: `timestamp,profile_id,segment_id,predicate_string,cache_key,event_type,cache_lookup_us,filter_build_us,bitset_bytes,matched_count,segment_rows`

- `patches/milvus_search_profile.patch`
  - Instruments `PhyVectorSearchNode` vector-search timing after the upstream bitset is materialized.
  - Output: `/tmp/milvus_search_profile.csv`
  - Columns: `timestamp,profile_id,total_search_us,nq,topk,index_type,predicate_string`

## Runtime Guard

All C++ profiling hooks are guarded by:

```bash
MILVUS_FILTER_PROFILE=1
```

When the variable is absent or not equal to `1`, the profiling hooks should return immediately.

## Joining Logs

The patches use `QueryContext::query_id()` as `profile_id` when it is available. If it is empty, they fall back to a monotonic atomic id local to each compilation unit.

If query ids are populated, join:

- filter rows to search rows by `profile_id`
- filter rows to cache rows by `profile_id` when available, or by `segment_id` and `predicate_string/cache_key`

If atomic fallbacks are used, the ids may not align across files. Use timestamp windows, segment id, and predicate string as secondary join keys.

## Known TODOs

- Verify `TargetBitmap::count()` and `TargetBitmapView::count()` on the final Milvus checkout.
- Decide whether `total_search_us` should mean vector node time or full plan/driver time. The scaffold records vector node time because that path has `nq`, `topk`, and the bitset handoff.
- If search CSV rows must include predicate text directly, add a `QueryContext` field set by `FilterBitsNode` and read by `VectorSearchNode`.
- Replace `search_params_.dump()` with an exact index type accessor if the final checkout exposes one.
- Populate nonzero `filter_build_us` in cache insert rows by passing miss build duration from `FilterBitsNode` to the cache layer.
