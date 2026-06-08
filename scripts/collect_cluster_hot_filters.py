#!/usr/bin/env python3
"""Compare global hot filters with IVF-list-local hot filters."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPLIT_RE = re.compile(r"[\s,;|]+")


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


def top_keys(counter: Counter[str], k: int) -> List[str]:
    return [key for key, _count in counter.most_common(k)]


def coverage(counter: Counter[str], selected: Iterable[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    selected = set(selected)
    return sum(count for key, count in counter.items() if key in selected) / total


def analyze(trace_csv: str, top_k: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    global_counts: Counter[str] = Counter()
    per_list_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    with Path(trace_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            predicate = predicate_key(row)
            lists = probed_lists(row)
            global_counts[predicate] += 1
            for list_id in lists:
                per_list_counts[list_id][predicate] += 1

    global_top = top_keys(global_counts, top_k)
    global_rows = [
        {"rank": rank, "predicate": pred, "count": global_counts[pred]}
        for rank, pred in enumerate(global_top, start=1)
    ]

    local_rows: List[Dict[str, object]] = []
    overlap_values: List[float] = []
    local_covered = 0
    local_total = 0
    for list_id in sorted(per_list_counts, key=lambda x: int(x) if x.isdigit() else x):
        counter = per_list_counts[list_id]
        local_top = top_keys(counter, top_k)
        overlap = len(set(local_top) & set(global_top))
        overlap_values.append(overlap / top_k if top_k else 0.0)
        local_covered += sum(counter[pred] for pred in local_top)
        local_total += sum(counter.values())
        for rank, pred in enumerate(local_top, start=1):
            local_rows.append(
                {
                    "list_id": list_id,
                    "rank": rank,
                    "predicate": pred,
                    "count": counter[pred],
                    "in_global_top_k": pred in global_top,
                }
            )

    global_cov = coverage(global_counts, global_top)
    local_cov = local_covered / local_total if local_total else 0.0
    summary = {
        "top_k": top_k,
        "global_total_queries": sum(global_counts.values()),
        "local_total_list_hits": local_total,
        "global_top_k_coverage": global_cov,
        "local_top_k_coverage": local_cov,
        "coverage_lift": local_cov / global_cov if global_cov else 0.0,
        "avg_overlap_fraction": sum(overlap_values) / len(overlap_values)
        if overlap_values
        else 0.0,
        "num_ivf_lists_seen": len(per_list_counts),
    }
    return global_rows, local_rows, summary


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    global_rows, local_rows, summary = analyze(args.trace_csv, args.top_k)
    print("Global hot filters:")
    for row in global_rows:
        print(f"{row['rank']}\t{row['predicate']}\t{row['count']}")
    print("Summary:")
    for key, value in summary.items():
        print(f"{key}={value}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "global_hot_filters.csv", global_rows, ["rank", "predicate", "count"])
        write_csv(
            out_dir / "per_list_hot_filters.csv",
            local_rows,
            ["list_id", "rank", "predicate", "count", "in_global_top_k"],
        )
        write_csv(
            out_dir / "cluster_hot_filter_summary.csv",
            [summary],
            list(summary.keys()),
        )


if __name__ == "__main__":
    main()
