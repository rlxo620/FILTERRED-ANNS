#!/usr/bin/env python3
"""Apply the Milvus filter-profile scaffold by editing known source anchors.

This is a fallback for Milvus checkouts where the unified diff patches do not
apply cleanly because context lines drifted. It does not build or run Milvus.
Run from a Milvus checkout:

    python3 ../FILTERRED-ANNS/scripts/apply_milvus_profile_scaffold.py --milvus-root .
"""

from __future__ import annotations

import argparse
from pathlib import Path


FILTER_HELPERS = r'''std::atomic<uint64_t> g_filter_profile_id{0};
std::mutex g_filter_profile_mu;

bool
FilterProfileEnabled() {
    const char* value = std::getenv("MILVUS_FILTER_PROFILE");
    return value != nullptr && std::string(value) == "1";
}

int64_t
UnixMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string
CsvEscape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 2);
    out.push_back('"');
    for (char c : input) {
        if (c == '"') {
            out.push_back('"');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::string
BuildProfileId(QueryContext* query_context) {
    if (query_context != nullptr) {
        auto query_id = query_context->query_id();
        if (!query_id.empty()) {
            return query_id;
        }
    }
    return std::to_string(g_filter_profile_id.fetch_add(1));
}

int64_t
SegmentId(QueryContext* query_context) {
    auto* segment = query_context != nullptr ? query_context->get_segment()
                                             : nullptr;
    return segment != nullptr ? segment->get_segment_id() : -1;
}

int64_t
MatchedCountFromExclusionBitmap(const TargetBitmap& bitset,
                                int64_t segment_rows) {
    return segment_rows - static_cast<int64_t>(bitset.count());
}

int64_t
MatchedCountFromExclusionView(const TargetBitmapView& view,
                              int64_t segment_rows) {
    return segment_rows - static_cast<int64_t>(view.count());
}

void
AppendFilterProfileRow(QueryContext* query_context,
                       const std::string& predicate_string,
                       int64_t segment_rows,
                       double filter_gen_us,
                       size_t bitset_bytes,
                       int64_t matched_count) {
    if (!FilterProfileEnabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_filter_profile_mu);
    std::ofstream out("/tmp/milvus_filter_profile.csv", std::ios::app);
    out << UnixMicros() << ','
        << CsvEscape(BuildProfileId(query_context)) << ','
        << SegmentId(query_context) << ','
        << CsvEscape(predicate_string) << ','
        << segment_rows << ','
        << static_cast<int64_t>(filter_gen_us) << ','
        << bitset_bytes << ','
        << matched_count << '\n';
}

'''


EXPR_CACHE_HELPERS = r'''namespace {

std::mutex g_expr_cache_profile_mu;

bool
ExprCacheProfileEnabled() {
    const char* value = std::getenv("MILVUS_FILTER_PROFILE");
    return value != nullptr && std::string(value) == "1";
}

int64_t
ExprCacheUnixMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string
ExprCacheCsvEscape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 2);
    out.push_back('"');
    for (char c : input) {
        if (c == '"') {
            out.push_back('"');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

size_t
ExprCacheBitsetBytes(const ExprResCacheManager::Value& value) {
    size_t bytes = 0;
    if (value.result) {
        bytes += (value.result->size() + 7) / 8;
    }
    if (value.valid_result) {
        bytes += (value.valid_result->size() + 7) / 8;
    }
    return bytes;
}

int64_t
ExprCacheMatchedCount(const ExprResCacheManager::Value& value) {
    if (!value.result) {
        return 0;
    }
    return value.active_count -
           static_cast<int64_t>(value.result->count());
}

void
AppendExprCacheProfileRow(const ExprResCacheManager::Key& key,
                          const std::string& event_type,
                          double cache_lookup_us,
                          double filter_build_us,
                          const ExprResCacheManager::Value* value) {
    if (!ExprCacheProfileEnabled()) {
        return;
    }
    const int64_t rows = value != nullptr ? value->active_count : 0;
    const size_t bitset_bytes =
        value != nullptr ? ExprCacheBitsetBytes(*value) : 0;
    const int64_t matched =
        value != nullptr ? ExprCacheMatchedCount(*value) : 0;

    std::lock_guard<std::mutex> lock(g_expr_cache_profile_mu);
    std::ofstream out("/tmp/milvus_expr_cache_profile.csv", std::ios::app);
    out << ExprCacheUnixMicros() << ','
        << ExprCacheCsvEscape("") << ','
        << key.segment_id << ','
        << ExprCacheCsvEscape(key.signature) << ','
        << ExprCacheCsvEscape(key.signature) << ','
        << event_type << ','
        << static_cast<int64_t>(cache_lookup_us) << ','
        << static_cast<int64_t>(filter_build_us) << ','
        << bitset_bytes << ','
        << matched << ','
        << rows << '\n';
}

}  // namespace

'''


SEARCH_HELPERS = r'''namespace {

std::atomic<uint64_t> g_search_profile_id{0};
std::mutex g_search_profile_mu;

bool
SearchProfileEnabled() {
    const char* value = std::getenv("MILVUS_FILTER_PROFILE");
    return value != nullptr && std::string(value) == "1";
}

int64_t
SearchUnixMicros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string
SearchCsvEscape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 2);
    out.push_back('"');
    for (char c : input) {
        if (c == '"') {
            out.push_back('"');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::string
SearchProfileId(QueryContext* query_context) {
    if (query_context != nullptr) {
        auto query_id = query_context->query_id();
        if (!query_id.empty()) {
            return query_id;
        }
    }
    return std::to_string(g_search_profile_id.fetch_add(1));
}

void
AppendSearchProfileRow(QueryContext* query_context,
                       double total_search_us,
                       int64_t nq,
                       int64_t topk,
                       const std::string& index_type,
                       const std::string& predicate_string) {
    if (!SearchProfileEnabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_search_profile_mu);
    std::ofstream out("/tmp/milvus_search_profile.csv", std::ios::app);
    out << SearchUnixMicros() << ','
        << SearchCsvEscape(SearchProfileId(query_context)) << ','
        << static_cast<int64_t>(total_search_us) << ','
        << nq << ','
        << topk << ','
        << SearchCsvEscape(index_type) << ','
        << SearchCsvEscape(predicate_string) << '\n';
}

}  // namespace

'''


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def add_includes(text: str, includes: list[str]) -> str:
    for include in includes:
        line = f"#include <{include}>"
        if line not in text:
            anchor = '#include <algorithm>\n'
            if anchor in text:
                text = text.replace(anchor, anchor + line + "\n", 1)
            else:
                text = text.replace("\n\n", f"\n{line}\n\n", 1)
    return text


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"anchor not found for {label}")
    return text.replace(old, new, 1)


def patch_filter_header(path: Path) -> None:
    text = read(path)
    if "filter_profile_predicate_" in text:
        return
    text = replace_once(
        text,
        "    std::string expr_cache_key_;\n",
        "    std::string expr_cache_key_;\n"
        "    // Human-readable predicate string for optional profiling logs.\n"
        "    std::string filter_profile_predicate_;\n",
        "FilterBitsNode.h predicate member",
    )
    write(path, text)


def patch_filter_cpp(path: Path) -> None:
    text = read(path)
    if "AppendFilterProfileRow" in text:
        return
    text = add_includes(text, ["atomic", "cstdlib", "fstream", "mutex", "sstream"])
    text = replace_once(text, "namespace {\n\n", "namespace {\n\n" + FILTER_HELPERS, "filter helpers")
    text = replace_once(
        text,
        "    if (enable_expr_cache_) {\n"
        "        expr_cache_key_ = BuildExprCacheKey(*filter, query_context_);\n"
        "    }\n",
        "    if (enable_expr_cache_) {\n"
        "        expr_cache_key_ = BuildExprCacheKey(*filter, query_context_);\n"
        "    }\n"
        "    filter_profile_predicate_ = filter->ToString();\n",
        "filter predicate assignment",
    )
    text = replace_once(
        text,
        "        milvus::monitor::internal_core_search_latency_scalar.Observe(\n"
        "            scalar_cost / 1000);\n\n"
        "        return std::make_shared<RowVector>(col_res);\n",
        "        milvus::monitor::internal_core_search_latency_scalar.Observe(\n"
        "            scalar_cost / 1000);\n\n"
        "        AppendFilterProfileRow(\n"
        "            query_context_,\n"
        "            filter_profile_predicate_,\n"
        "            need_process_rows_,\n"
        "            scalar_cost,\n"
        "            (col_vec_size + 7) / 8,\n"
        "            MatchedCountFromExclusionView(view, need_process_rows_));\n"
        "        return std::make_shared<RowVector>(col_res);\n",
        "filter fast path log",
    )
    text = replace_once(
        text,
        "    // num_processed_rows_ = need_process_rows_;\n"
        "    std::vector<VectorPtr> col_res;\n",
        "    const size_t profile_bitset_bytes = (bitset.size() + 7) / 8;\n"
        "    const int64_t profile_matched_count =\n"
        "        MatchedCountFromExclusionBitmap(bitset, need_process_rows_);\n\n"
        "    // num_processed_rows_ = need_process_rows_;\n"
        "    std::vector<VectorPtr> col_res;\n",
        "filter non-fast path counters",
    )
    text = replace_once(
        text,
        "    milvus::monitor::internal_core_search_latency_scalar.Observe(scalar_cost /\n"
        "                                                                 1000);\n\n"
        "    return std::make_shared<RowVector>(col_res);\n",
        "    milvus::monitor::internal_core_search_latency_scalar.Observe(scalar_cost /\n"
        "                                                                 1000);\n\n"
        "    AppendFilterProfileRow(\n"
        "        query_context_,\n"
        "        filter_profile_predicate_,\n"
        "        need_process_rows_,\n"
        "        scalar_cost,\n"
        "        profile_bitset_bytes,\n"
        "        profile_matched_count);\n"
        "    return std::make_shared<RowVector>(col_res);\n",
        "filter non-fast path log",
    )
    write(path, text)


def patch_expr_cache_cpp(path: Path) -> None:
    text = read(path)
    if "AppendExprCacheProfileRow" in text:
        return
    text = add_includes(text, ["chrono", "cstdlib", "fstream", "mutex", "sstream"])
    text = replace_once(
        text,
        "std::atomic<bool> ExprResCacheManager::enabled_{false};\n\n",
        "std::atomic<bool> ExprResCacheManager::enabled_{false};\n\n"
        + EXPR_CACHE_HELPERS,
        "expr cache helpers",
    )
    text = replace_once(
        text,
        "    auto it = concurrent_map_.find(key);\n"
        "    if (it == concurrent_map_.end()) {\n"
        "        return false;\n"
        "    }\n",
        "    auto lookup_start = std::chrono::high_resolution_clock::now();\n"
        "    auto it = concurrent_map_.find(key);\n"
        "    if (it == concurrent_map_.end()) {\n"
        "        auto lookup_end = std::chrono::high_resolution_clock::now();\n"
        "        auto lookup_us =\n"
        "            std::chrono::duration<double, std::micro>(lookup_end - lookup_start)\n"
        "                .count();\n"
        "        AppendExprCacheProfileRow(key, \"lookup\", lookup_us, 0, nullptr);\n"
        "        AppendExprCacheProfileRow(key, \"miss\", lookup_us, 0, nullptr);\n"
        "        return false;\n"
        "    }\n",
        "expr cache miss",
    )
    text = replace_once(
        text,
        "    LOG_DEBUG(\"get expr res cache, segment_id: {}, key: {}\",\n"
        "              key.segment_id,\n"
        "              key.signature);\n"
        "    return true;\n",
        "    auto lookup_end = std::chrono::high_resolution_clock::now();\n"
        "    auto lookup_us =\n"
        "        std::chrono::duration<double, std::micro>(lookup_end - lookup_start)\n"
        "            .count();\n"
        "    AppendExprCacheProfileRow(key, \"lookup\", lookup_us, 0, &out_value);\n"
        "    AppendExprCacheProfileRow(key, \"hit\", lookup_us, 0, &out_value);\n\n"
        "    LOG_DEBUG(\"get expr res cache, segment_id: {}, key: {}\",\n"
        "              key.segment_id,\n"
        "              key.signature);\n"
        "    return true;\n",
        "expr cache hit",
    )
    text = replace_once(
        text,
        "    if (current_bytes_.load() > capacity_bytes_.load()) {\n"
        "        EnsureCapacity();\n"
        "    }\n",
        "    AppendExprCacheProfileRow(key, \"insert\", 0, 0, &stored_value);\n\n"
        "    if (current_bytes_.load() > capacity_bytes_.load()) {\n"
        "        EnsureCapacity();\n"
        "    }\n",
        "expr cache insert",
    )
    text = replace_once(
        text,
        "            current_bytes_.fetch_sub(it->second.value.bytes);\n"
        "            concurrent_map_.unsafe_erase(it);\n",
        "            current_bytes_.fetch_sub(it->second.value.bytes);\n"
        "            AppendExprCacheProfileRow(back_key, \"evict\", 0, 0, &it->second.value);\n"
        "            concurrent_map_.unsafe_erase(it);\n",
        "expr cache evict",
    )
    write(path, text)


def patch_vector_search_cpp(path: Path) -> None:
    text = read(path)
    if "AppendSearchProfileRow" in text:
        return
    text = add_includes(text, ["atomic", "cstdlib", "fstream", "mutex", "sstream"])
    text = replace_once(
        text,
        "namespace milvus {\nnamespace exec {\n\n",
        "namespace milvus {\nnamespace exec {\n\n" + SEARCH_HELPERS,
        "search helpers",
    )
    text = replace_once(
        text,
        "    milvus::monitor::internal_core_search_latency_vector.Observe(vector_cost /\n"
        "                                                                 1000);\n"
        "    // vector search stores result in query_context;\n",
        "    milvus::monitor::internal_core_search_latency_vector.Observe(vector_cost /\n"
        "                                                                 1000);\n"
        "    AppendSearchProfileRow(query_context_,\n"
        "                           vector_cost,\n"
        "                           num_queries,\n"
        "                           search_info_.topk_,\n"
        "                           search_info_.search_params_.dump(),\n"
        "                           \"\");\n"
        "    // vector search stores result in query_context;\n",
        "search profile row",
    )
    write(path, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milvus-root", default=".")
    args = parser.parse_args()

    root = Path(args.milvus_root).resolve()
    edits = [
        (patch_filter_header, root / "internal/core/src/exec/operator/FilterBitsNode.h"),
        (patch_filter_cpp, root / "internal/core/src/exec/operator/FilterBitsNode.cpp"),
        (patch_expr_cache_cpp, root / "internal/core/src/exec/expression/ExprCache.cpp"),
        (patch_vector_search_cpp, root / "internal/core/src/exec/operator/VectorSearchNode.cpp"),
    ]
    for func, path in edits:
        if not path.exists():
            raise FileNotFoundError(path)
        func(path)
        print(f"updated {path.relative_to(root)}")


if __name__ == "__main__":
    main()
