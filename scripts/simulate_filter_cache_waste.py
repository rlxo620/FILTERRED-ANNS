#!/usr/bin/env python3
"""Estimate global filter/cache waste for probed IVF lists."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


SPLIT_RE = re.compile(r"[\s,;|]+")


def ceil_bytes(rows: int) -> int:
    return int(math.ceil(rows / 8.0))


def parse_list_ids(value: str) -> List[str]:
    cleaned = (value or "").strip().strip("[](){}")
    if not cleaned:
        return []
    return [item for item in SPLIT_RE.split(cleaned) if item]


def predicate_key(row: Dict[str, str]) -> str:
    if row.get("predicate"):
        return row["predicate"]
    return f"{row.get('predicate_type', '')}={row.get('predicate_value', '')}"


def probed_lists(row: Dict[str, str]) -> List[str]:
    value = row.get("probed_list_ids", "")
    extras = row.get(None)  # type: ignore[arg-type]
    if extras:
        value = ",".join([value] + extras)
    return parse_list_ids(value)


def load_list_sizes(path: str) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        reader = csv.reader(handle)
        header = next(reader, None) if has_header else None
        for row in reader:
            if not row:
                continue
            if header:
                values = dict(zip(header, row))
                list_id = values.get("list_id") or values.get("ivf_list_id") or row[0]
                rows_value = (
                    values.get("rows")
                    or values.get("size")
                    or values.get("list_size")
                    or row[1]
                )
            else:
                list_id, rows_value = row[0], row[1]
            sizes[str(list_id)] = int(float(rows_value))
    return sizes


def analyze(total_database_size: int, list_sizes: Dict[str, int], trace_csv: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    global_filter_bytes = ceil_bytes(total_database_size)
    per_query_rows: List[Dict[str, object]] = []
    unique_predicates = set()
    predicate_list_pairs = set()
    totals = defaultdict(float)

    with Path(trace_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            predicate = predicate_key(row)
            lists = probed_lists(row)
            unique_predicates.add(predicate)
            needed_rows = sum(list_sizes.get(list_id, 0) for list_id in lists)
            needed_bytes = ceil_bytes(needed_rows)
            for list_id in lists:
                predicate_list_pairs.add((predicate, list_id))
            useful_ratio = needed_bytes / global_filter_bytes if global_filter_bytes else 0.0
            wasted_ratio = 1.0 - useful_ratio
            wasted_bytes = max(global_filter_bytes - needed_bytes, 0)
            totals["queries"] += 1
            totals["global_filter_bytes"] += global_filter_bytes
            totals["probed_list_needed_bytes"] += needed_bytes
            totals["global_wasted_bytes"] += wasted_bytes
            per_query_rows.append(
                {
                    "query_id": row.get("query_id", ""),
                    "predicate": predicate,
                    "num_probed_lists": len(lists),
                    "global_filter_size_bytes": global_filter_bytes,
                    "probed_list_needed_filter_size_bytes": needed_bytes,
                    "useful_ratio": useful_ratio,
                    "wasted_ratio": wasted_ratio,
                    "estimated_global_cache_wasted_bytes": wasted_bytes,
                    "estimated_list_local_cache_bytes": needed_bytes,
                }
            )

    distinct_global_cache_bytes = len(unique_predicates) * global_filter_bytes
    distinct_list_local_cache_bytes = sum(
        ceil_bytes(list_sizes.get(list_id, 0))
        for _predicate, list_id in predicate_list_pairs
    )
    summary = {
        "queries": int(totals["queries"]),
        "unique_predicates": len(unique_predicates),
        "unique_predicate_list_pairs": len(predicate_list_pairs),
        "avg_global_filter_size_bytes": (
            totals["global_filter_bytes"] / totals["queries"] if totals["queries"] else 0
        ),
        "avg_probed_list_needed_filter_size_bytes": (
            totals["probed_list_needed_bytes"] / totals["queries"]
            if totals["queries"]
            else 0
        ),
        "avg_useful_ratio": (
            totals["probed_list_needed_bytes"] / totals["global_filter_bytes"]
            if totals["global_filter_bytes"]
            else 0
        ),
        "avg_wasted_ratio": (
            totals["global_wasted_bytes"] / totals["global_filter_bytes"]
            if totals["global_filter_bytes"]
            else 0
        ),
        "estimated_global_cache_wasted_bytes": int(totals["global_wasted_bytes"]),
        "estimated_list_local_cache_bytes": int(totals["probed_list_needed_bytes"]),
        "distinct_predicate_global_cache_bytes": distinct_global_cache_bytes,
        "distinct_predicate_list_local_cache_bytes": distinct_list_local_cache_bytes,
        "distinct_predicate_cache_wasted_bytes": max(
            distinct_global_cache_bytes - distinct_list_local_cache_bytes, 0
        ),
    }
    return per_query_rows, summary


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-database-size", type=int, required=True)
    parser.add_argument("--ivf-list-sizes", required=True)
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    per_query, summary = analyze(
        args.total_database_size,
        load_list_sizes(args.ivf_list_sizes),
        args.trace_csv,
    )
    print("Summary:")
    for key, value in summary.items():
        print(f"{key}={value}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "filter_cache_waste_per_query.csv", per_query, list(per_query[0].keys()) if per_query else [])
        write_csv(out_dir / "filter_cache_waste_summary.csv", [summary], list(summary.keys()))


if __name__ == "__main__":
    main()
