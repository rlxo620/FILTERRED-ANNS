#!/usr/bin/env python3
"""Analyze Milvus filter generation, search, and expression-cache profile CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_csv(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def predicate_of(row: Dict[str, str]) -> str:
    return row.get("predicate_string") or row.get("cache_key") or "<unknown>"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(
    filter_rows: List[Dict[str, str]],
    search_rows: List[Dict[str, str]],
    cache_rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    search_by_profile: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in search_rows:
        search_by_profile[row.get("profile_id", "")].append(row)

    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    cache_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in filter_rows:
        predicate = predicate_of(row)
        filter_gen_us = as_float(row.get("filter_gen_us"))
        stats[predicate]["filter_gen_us"].append(filter_gen_us)
        stats[predicate]["bitset_bytes"].append(as_float(row.get("bitset_bytes")))

        profile_id = row.get("profile_id", "")
        total_search_values = [
            as_float(search_row.get("total_search_us"))
            for search_row in search_by_profile.get(profile_id, [])
            if as_float(search_row.get("total_search_us")) > 0
        ]
        if total_search_values:
            stats[predicate]["filter_gen_ratio"].append(
                filter_gen_us / mean(total_search_values)
            )

    for row in cache_rows:
        predicate = predicate_of(row)
        event = (row.get("event_type") or "").strip().lower()
        if event in {"hit", "miss"}:
            cache_stats[predicate]["lookups"] += 1
        if event == "hit":
            cache_stats[predicate]["hits"] += 1
        if event == "miss":
            cache_stats[predicate]["misses"] += 1
            build_us = as_float(row.get("filter_build_us"))
            if build_us > 0:
                cache_stats[predicate]["miss_build_sum"] += build_us
                cache_stats[predicate]["miss_build_count"] += 1

    predicates = sorted(set(stats) | set(cache_stats))
    summaries: List[Dict[str, Any]] = []
    for predicate in predicates:
        lookups = cache_stats[predicate]["lookups"]
        hits = cache_stats[predicate]["hits"]
        miss_build_count = cache_stats[predicate]["miss_build_count"]
        summaries.append(
            {
                "predicate_string": predicate,
                "samples": len(stats[predicate]["filter_gen_us"]),
                "avg_filter_gen_ms": mean(stats[predicate]["filter_gen_us"]) / 1000.0,
                "avg_filter_gen_ratio": mean(stats[predicate]["filter_gen_ratio"]),
                "avg_bitset_bytes": mean(stats[predicate]["bitset_bytes"]),
                "cache_lookups": int(lookups),
                "cache_hits": int(hits),
                "cache_hit_rate": hits / lookups if lookups else 0.0,
                "avg_cache_miss_filter_build_us": (
                    cache_stats[predicate]["miss_build_sum"] / miss_build_count
                    if miss_build_count
                    else 0.0
                ),
            }
        )
    return summaries


def write_summary(path: Optional[str], rows: List[Dict[str, Any]]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "predicate_string",
        "samples",
        "avg_filter_gen_ms",
        "avg_filter_gen_ratio",
        "avg_bitset_bytes",
        "cache_lookups",
        "cache_hits",
        "cache_hit_rate",
        "avg_cache_miss_filter_build_us",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No profile rows found.")
        return
    for row in rows:
        print(
            f"{row['predicate_string']}\t"
            f"samples={row['samples']}\t"
            f"avg_filter_gen_ms={row['avg_filter_gen_ms']:.3f}\t"
            f"avg_filter_gen_ratio={row['avg_filter_gen_ratio']:.4f}\t"
            f"avg_bitset_bytes={row['avg_bitset_bytes']:.1f}\t"
            f"cache_hit_rate={row['cache_hit_rate']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-profile", required=True)
    parser.add_argument("--search-profile", default=None)
    parser.add_argument("--expr-cache-profile", default=None)
    parser.add_argument("--summary-csv", default=None)
    args = parser.parse_args()

    rows = summarize(
        read_csv(args.filter_profile),
        read_csv(args.search_profile),
        read_csv(args.expr_cache_profile),
    )
    print_summary(rows)
    write_summary(args.summary_csv, rows)


if __name__ == "__main__":
    main()
