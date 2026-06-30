#!/usr/bin/env python3
"""Measure cluster-level hot filters using real vectors and FAISS IVF centroids.

This script is for the motivation experiment:

    Do different IVF clusters see different hot filters in a real filtered
    ANNS benchmark such as YFCC?

It does not synthesize filters. It uses:
  * real base embeddings to train/load FAISS IVF coarse centroids
  * real query embeddings to find the nprobe clusters per query
  * real query filter metadata/predicates to count hot filters per cluster

Typical output:
  * query_probed_clusters.csv
  * global_query_filter_hotness.csv
  * cluster_query_filter_hotness.csv
  * cluster_base_counts.csv, with --assign-base-clusters
  * cluster_hot_filter_summary.csv

Supported vector formats:
  * .npy  : numpy array [N, D], float32-compatible
  * .fbin : int32 N, int32 D header followed by float32 data
  * .fvecs: repeated int32 D followed by D float32 values
  * .u8bin: int32 N, int32 D header followed by uint8 data

Supported query-filter formats:
  * .csv  : predicate column or filter columns
  * .spmat: CSR sparse metadata matrix from big-ann-benchmarks
            (each query row's nonzero metadata ids become one predicate)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SPLIT_RE = re.compile(r"[\s,;|]+")


def require_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "numpy is required. Install with: python3 -m pip install --user numpy"
        ) from exc
    return np


def require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "faiss is required. Install with: python3 -m pip install --user faiss-cpu"
        ) from exc
    return faiss


class MatrixSource:
    path: Path
    shape: Tuple[int, int]

    def read_rows(self, indices: Sequence[int]):
        raise NotImplementedError

    def iter_chunks(self, chunk_rows: int):
        raise NotImplementedError


class NpyMatrix(MatrixSource):
    def __init__(self, path: Path):
        np = require_numpy()
        self.path = path
        self.array = np.load(path, mmap_mode="r")
        if self.array.ndim != 2:
            raise SystemExit(f"{path} must have shape [N, D]")
        self.shape = (int(self.array.shape[0]), int(self.array.shape[1]))

    def read_rows(self, indices: Sequence[int]):
        np = require_numpy()
        return np.asarray(self.array[list(indices)], dtype=np.float32)

    def iter_chunks(self, chunk_rows: int):
        np = require_numpy()
        nrows = self.shape[0]
        for start in range(0, nrows, chunk_rows):
            end = min(start + chunk_rows, nrows)
            yield start, np.asarray(self.array[start:end], dtype=np.float32)


class FbinMatrix(MatrixSource):
    def __init__(self, path: Path):
        np = require_numpy()
        self.path = path
        with path.open("rb") as handle:
            header = handle.read(8)
        if len(header) != 8:
            raise SystemExit(f"{path} is too small for .fbin header")
        nrows, dim = struct.unpack("<ii", header)
        if nrows <= 0 or dim <= 0:
            raise SystemExit(f"{path} has invalid .fbin shape: {nrows} x {dim}")
        self.shape = (int(nrows), int(dim))
        self.array = np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=self.shape)

    def read_rows(self, indices: Sequence[int]):
        np = require_numpy()
        return np.asarray(self.array[list(indices)], dtype=np.float32)

    def iter_chunks(self, chunk_rows: int):
        np = require_numpy()
        nrows = self.shape[0]
        for start in range(0, nrows, chunk_rows):
            end = min(start + chunk_rows, nrows)
            yield start, np.asarray(self.array[start:end], dtype=np.float32)


class U8binMatrix(MatrixSource):
    def __init__(self, path: Path):
        np = require_numpy()
        self.path = path
        with path.open("rb") as handle:
            header = handle.read(8)
        if len(header) != 8:
            raise SystemExit(f"{path} is too small for .u8bin header")
        nrows, dim = struct.unpack("<ii", header)
        if nrows <= 0 or dim <= 0:
            raise SystemExit(f"{path} has invalid .u8bin shape: {nrows} x {dim}")
        self.shape = (int(nrows), int(dim))
        self.array = np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=self.shape)

    def read_rows(self, indices: Sequence[int]):
        np = require_numpy()
        return np.asarray(self.array[list(indices)], dtype=np.float32)

    def iter_chunks(self, chunk_rows: int):
        np = require_numpy()
        nrows = self.shape[0]
        for start in range(0, nrows, chunk_rows):
            end = min(start + chunk_rows, nrows)
            yield start, np.asarray(self.array[start:end], dtype=np.float32)


class FvecsMatrix(MatrixSource):
    def __init__(self, path: Path):
        np = require_numpy()
        self.path = path
        with path.open("rb") as handle:
            raw = handle.read(4)
        if len(raw) != 4:
            raise SystemExit(f"{path} is too small for .fvecs")
        (dim,) = struct.unpack("<i", raw)
        if dim <= 0:
            raise SystemExit(f"{path} has invalid fvec dimension {dim}")
        record_bytes = 4 + 4 * dim
        size = os.path.getsize(path)
        if size % record_bytes != 0:
            raise SystemExit(f"{path} size is not divisible by fvec record size")
        self.dim = dim
        self.record_bytes = record_bytes
        self.shape = (size // record_bytes, dim)

    def read_rows(self, indices: Sequence[int]):
        np = require_numpy()
        out = np.empty((len(indices), self.dim), dtype=np.float32)
        with self.path.open("rb") as handle:
            for out_idx, row_idx in enumerate(indices):
                handle.seek(int(row_idx) * self.record_bytes + 4)
                out[out_idx] = np.frombuffer(handle.read(4 * self.dim), dtype="<f4")
        return out

    def iter_chunks(self, chunk_rows: int):
        np = require_numpy()
        nrows = self.shape[0]
        with self.path.open("rb") as handle:
            for start in range(0, nrows, chunk_rows):
                end = min(start + chunk_rows, nrows)
                rows = end - start
                out = np.empty((rows, self.dim), dtype=np.float32)
                handle.seek(start * self.record_bytes)
                for idx in range(rows):
                    dim_raw = handle.read(4)
                    if len(dim_raw) != 4:
                        raise SystemExit(f"Unexpected EOF in {self.path}")
                    out[idx] = np.frombuffer(handle.read(4 * self.dim), dtype="<f4")
                yield start, out


def open_matrix(path: Path) -> MatrixSource:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return NpyMatrix(path)
    if suffix == ".fbin":
        return FbinMatrix(path)
    if suffix == ".u8bin":
        return U8binMatrix(path)
    if suffix == ".fvecs":
        return FvecsMatrix(path)
    raise SystemExit(f"Unsupported vector format: {path}. Use .npy, .fbin, .u8bin, or .fvecs")


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def csv_header(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader)


def iter_csv(path: Path) -> Iterator[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def sample_rows(matrix: MatrixSource, sample_rows_count: int, seed: int):
    np = require_numpy()
    nrows = matrix.shape[0]
    sample_n = min(sample_rows_count, nrows)
    rng = np.random.default_rng(seed)
    indices = rng.choice(nrows, size=sample_n, replace=False)
    indices.sort()
    return matrix.read_rows(indices)


def train_or_load_quantizer(args, base: MatrixSource):
    np = require_numpy()
    faiss = require_faiss()

    if args.faiss_index:
        index = faiss.read_index(args.faiss_index)
        if not hasattr(index, "quantizer") or not hasattr(index, "nlist"):
            raise SystemExit("--faiss-index must be an IVF index with quantizer and nlist")
        quantizer = index.quantizer
        nlist = int(index.nlist)
        dim = int(index.d)
        if dim != base.shape[1]:
            raise SystemExit(f"FAISS index dim {dim} != base vector dim {base.shape[1]}")
        return quantizer, nlist

    centroids_path = Path(args.centroids_npy) if args.centroids_npy else None
    if centroids_path and centroids_path.exists():
        centroids = np.load(centroids_path)
        if centroids.ndim != 2:
            raise SystemExit(f"{centroids_path} must have shape [nlist, dim]")
    else:
        train = sample_rows(base, args.train_sample_rows, args.seed)
        if train.shape[0] < args.nlist:
            raise SystemExit(
                f"Training sample {train.shape[0]} is smaller than nlist {args.nlist}"
            )
        kmeans = faiss.Kmeans(
            base.shape[1],
            args.nlist,
            niter=args.kmeans_iterations,
            verbose=True,
            seed=args.seed,
            gpu=False,
        )
        kmeans.train(train)
        centroids = kmeans.centroids
        if centroids_path:
            centroids_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(centroids_path, centroids)

    quantizer = faiss.IndexFlatL2(base.shape[1])
    quantizer.add(np.asarray(centroids, dtype=np.float32))
    return quantizer, int(centroids.shape[0])


def spmat_query_predicates(filters_spmat: Path, prefix: str, vocabulary: Optional[Path]) -> List[str]:
    np = require_numpy()
    with filters_spmat.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24:
        raise SystemExit(f"{filters_spmat} is too small for .spmat header")
    nrow, ncol, nnz = struct.unpack("<qqq", header)
    if nrow <= 0 or ncol <= 0 or nnz < 0:
        raise SystemExit(f"{filters_spmat} has invalid .spmat shape")

    indptr_offset = 24
    indices_offset = indptr_offset + 8 * (nrow + 1)
    data_offset = indices_offset + 4 * nnz
    expected_min_size = data_offset + 4 * nnz
    actual_size = filters_spmat.stat().st_size
    if actual_size < expected_min_size:
        raise SystemExit(
            f"{filters_spmat} is too small for expected CSR payload: "
            f"actual={actual_size} expected>={expected_min_size}"
        )

    indptr = np.memmap(filters_spmat, dtype=np.int64, mode="r", offset=indptr_offset, shape=(nrow + 1,))
    indices = np.memmap(filters_spmat, dtype=np.int32, mode="r", offset=indices_offset, shape=(nnz,))

    vocab: Optional[List[str]] = None
    if vocabulary is not None:
        with vocabulary.open(encoding="utf-8") as handle:
            vocab = [line.rstrip("\n") for line in handle]

    predicates: List[str] = []
    for row_id in range(nrow):
        start = int(indptr[row_id])
        end = int(indptr[row_id + 1])
        terms = [int(x) for x in indices[start:end]]
        if not terms:
            predicates.append("<empty>")
            continue
        labels = []
        for term in sorted(terms):
            if vocab is not None and 0 <= term < len(vocab):
                labels.append(f"{prefix}=={vocab[term]}")
            else:
                labels.append(f"{prefix}=={term}")
        predicates.append(" AND ".join(labels))
    return predicates


def query_predicates(
    filters_path: Path,
    predicate_column: str,
    filter_columns: Sequence[str],
    spmat_prefix: str,
    spmat_vocabulary: Optional[Path],
) -> List[str]:
    if filters_path.suffix.lower() == ".spmat":
        return spmat_query_predicates(filters_path, spmat_prefix, spmat_vocabulary)

    header = csv_header(filters_path)
    if predicate_column not in header and not filter_columns:
        raise SystemExit(
            f"{filters_path} needs column {predicate_column!r} or --query-filter-columns"
        )

    predicates: List[str] = []
    for row in iter_csv(filters_path):
        if predicate_column in row and row[predicate_column].strip():
            predicates.append(row[predicate_column].strip())
            continue
        parts = []
        for col in filter_columns:
            value = (row.get(col) or "").strip()
            if value:
                parts.append(f"{col}=={value}")
        predicates.append(" AND ".join(parts) if parts else "<empty>")
    return predicates


def update_hotness(
    quantizer,
    queries: MatrixSource,
    predicates: Sequence[str],
    nprobe: int,
    chunk_rows: int,
) -> Tuple[Counter[str], Dict[int, Counter[str]], List[Dict[str, object]]]:
    if queries.shape[0] < len(predicates):
        raise SystemExit(
            f"query vectors ({queries.shape[0]}) fewer than query filters ({len(predicates)})"
        )

    global_counts: Counter[str] = Counter()
    per_cluster: Dict[int, Counter[str]] = defaultdict(Counter)
    trace_rows: List[Dict[str, object]] = []

    processed = 0
    for start, chunk in queries.iter_chunks(chunk_rows):
        if start >= len(predicates):
            break
        keep = min(chunk.shape[0], len(predicates) - start)
        chunk = chunk[:keep]
        _distances, cluster_ids = quantizer.search(chunk, nprobe)
        for local_idx in range(keep):
            query_id = start + local_idx
            predicate = predicates[query_id]
            clusters = [int(x) for x in cluster_ids[local_idx] if int(x) >= 0]
            global_counts[predicate] += 1
            for cluster_id in clusters:
                per_cluster[cluster_id][predicate] += 1
            trace_rows.append(
                {
                    "query_id": query_id,
                    "predicate": predicate,
                    "probed_cluster_ids": " ".join(str(x) for x in clusters),
                }
            )
        processed += keep

    if processed != len(predicates):
        raise SystemExit(f"Only processed {processed} predicates out of {len(predicates)}")
    return global_counts, per_cluster, trace_rows


def assign_base_cluster_counts(quantizer, base: MatrixSource, chunk_rows: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for _start, chunk in base.iter_chunks(chunk_rows):
        _distances, cluster_ids = quantizer.search(chunk, 1)
        for cluster_id in cluster_ids[:, 0]:
            if int(cluster_id) >= 0:
                counts[int(cluster_id)] += 1
    return counts


def top_keys(counter: Counter[str], top_k: int) -> List[str]:
    return [key for key, _count in counter.most_common(top_k)]


def coverage(counter: Counter[str], selected: Iterable[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    selected_set = set(selected)
    return sum(count for key, count in counter.items() if key in selected_set) / total


def build_rows(
    global_counts: Counter[str],
    per_cluster: Dict[int, Counter[str]],
    top_k: int,
    base_cluster_counts: Optional[Counter[int]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    total_queries = sum(global_counts.values())
    global_top = top_keys(global_counts, top_k)
    global_top_set = set(global_top)
    global_rows = [
        {
            "rank": rank,
            "predicate": pred,
            "query_count": global_counts[pred],
            "query_fraction": global_counts[pred] / total_queries if total_queries else 0.0,
        }
        for rank, pred in enumerate(global_top, start=1)
    ]

    local_rows: List[Dict[str, object]] = []
    local_total = 0
    local_covered = 0
    overlap_sum = 0.0
    non_global_top_entries = 0
    unique_local_top: set[str] = set()
    entropy_values: List[float] = []

    for cluster_id in sorted(per_cluster):
        counter = per_cluster[cluster_id]
        cluster_total = sum(counter.values())
        if cluster_total == 0:
            continue
        local_total += cluster_total
        local_top = top_keys(counter, top_k)
        unique_local_top.update(local_top)
        local_covered += sum(counter[pred] for pred in local_top)
        overlap_sum += len(set(local_top) & global_top_set) / top_k if top_k else 0.0

        probs = [count / cluster_total for count in counter.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        entropy_values.append(entropy)

        for rank, pred in enumerate(local_top, start=1):
            if pred not in global_top_set:
                non_global_top_entries += 1
            local_rows.append(
                {
                    "cluster_id": cluster_id,
                    "rank": rank,
                    "predicate": pred,
                    "base_vectors_in_cluster": base_cluster_counts.get(cluster_id, 0)
                    if base_cluster_counts
                    else "",
                    "cluster_query_hits": counter[pred],
                    "cluster_total_hits": cluster_total,
                    "cluster_hit_fraction": counter[pred] / cluster_total,
                    "global_query_count": global_counts[pred],
                    "global_query_fraction": global_counts[pred] / total_queries
                    if total_queries
                    else 0.0,
                    "enrichment_vs_global": (counter[pred] / cluster_total)
                    / (global_counts[pred] / total_queries)
                    if total_queries and global_counts[pred]
                    else math.inf,
                    "in_global_top_k": pred in global_top_set,
                }
            )

    global_cov = coverage(global_counts, global_top)
    local_cov = local_covered / local_total if local_total else 0.0
    summary: Dict[str, object] = {
        "queries": total_queries,
        "cluster_query_hits": local_total,
        "clusters_seen": len(per_cluster),
        "unique_predicates": len(global_counts),
        "top_k": top_k,
        "global_top_k_query_coverage": global_cov,
        "cluster_local_top_k_hit_coverage": local_cov,
        "coverage_lift_local_over_global": local_cov / global_cov if global_cov else 0.0,
        "avg_cluster_top_k_overlap_with_global": overlap_sum / len(per_cluster)
        if per_cluster
        else 0.0,
        "local_top_entries_not_in_global_top_k": non_global_top_entries,
        "unique_predicates_in_cluster_top_k": len(unique_local_top),
        "avg_cluster_filter_entropy": sum(entropy_values) / len(entropy_values)
        if entropy_values
        else 0.0,
    }
    if base_cluster_counts:
        populated = [count for count in base_cluster_counts.values() if count > 0]
        summary.update(
            {
                "base_vectors_assigned": sum(base_cluster_counts.values()),
                "base_populated_clusters": len(populated),
                "base_cluster_min_size": min(populated) if populated else 0,
                "base_cluster_max_size": max(populated) if populated else 0,
                "base_cluster_avg_size": sum(populated) / len(populated)
                if populated
                else 0.0,
            }
        )
    return global_rows, local_rows, summary


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-vectors", required=True, help=".npy, .fbin, or .fvecs base embeddings")
    parser.add_argument("--query-vectors", required=True, help=".npy, .fbin, or .fvecs query embeddings")
    parser.add_argument("--query-filters-csv", required=True, help="CSV or big-ann .spmat query metadata")
    parser.add_argument("--query-predicate-column", default="predicate")
    parser.add_argument("--query-filter-columns", default="", help="Comma-separated columns if no predicate column")
    parser.add_argument("--spmat-prefix", default="tag", help="Predicate prefix for .spmat metadata ids")
    parser.add_argument("--spmat-vocabulary", default=None, help="Optional words.txt for .spmat metadata ids")
    parser.add_argument("--faiss-index", default=None, help="Optional existing FAISS IVF index")
    parser.add_argument("--centroids-npy", default=None, help="Optional path to load/save IVF centroids")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--train-sample-rows", type=int, default=200000)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--assign-base-clusters",
        action="store_true",
        help="Assign every base vector to its nearest IVF centroid and write cluster_base_counts.csv",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = open_matrix(Path(args.base_vectors))
    queries = open_matrix(Path(args.query_vectors))
    predicates = query_predicates(
        Path(args.query_filters_csv),
        args.query_predicate_column,
        parse_csv_list(args.query_filter_columns),
        args.spmat_prefix,
        Path(args.spmat_vocabulary) if args.spmat_vocabulary else None,
    )

    quantizer, nlist = train_or_load_quantizer(args, base)
    base_cluster_counts: Optional[Counter[int]] = None
    if args.assign_base_clusters:
        base_cluster_counts = assign_base_cluster_counts(quantizer, base, args.chunk_rows)

    global_counts, per_cluster, trace_rows = update_hotness(
        quantizer, queries, predicates, args.nprobe, args.chunk_rows
    )
    global_rows, local_rows, summary = build_rows(
        global_counts, per_cluster, args.top_k, base_cluster_counts
    )
    summary.update(
        {
            "backend": "faiss_ivf_quantizer",
            "nlist": nlist,
            "nprobe": args.nprobe,
            "base_vectors": str(args.base_vectors),
            "query_vectors": str(args.query_vectors),
            "query_filters_csv": str(args.query_filters_csv),
        }
    )

    write_csv(
        output_dir / "query_probed_clusters.csv",
        trace_rows,
        ["query_id", "predicate", "probed_cluster_ids"],
    )
    write_csv(
        output_dir / "global_query_filter_hotness.csv",
        global_rows,
        ["rank", "predicate", "query_count", "query_fraction"],
    )
    write_csv(
        output_dir / "cluster_query_filter_hotness.csv",
        local_rows,
        [
            "cluster_id",
            "rank",
            "predicate",
            "base_vectors_in_cluster",
            "cluster_query_hits",
            "cluster_total_hits",
            "cluster_hit_fraction",
            "global_query_count",
            "global_query_fraction",
            "enrichment_vs_global",
            "in_global_top_k",
        ],
    )
    if base_cluster_counts is not None:
        write_csv(
            output_dir / "cluster_base_counts.csv",
            [
                {"cluster_id": cluster_id, "base_vectors_in_cluster": count}
                for cluster_id, count in sorted(base_cluster_counts.items())
            ],
            ["cluster_id", "base_vectors_in_cluster"],
        )
    write_csv(output_dir / "cluster_hot_filter_summary.csv", [summary], list(summary.keys()))

    print(f"backend=faiss_ivf_quantizer")
    print(f"base_shape={base.shape}")
    print(f"query_shape={queries.shape}")
    print(f"queries_with_filters={len(predicates)}")
    print(f"nlist={nlist}")
    print(f"nprobe={args.nprobe}")
    print(f"global_top_k_query_coverage={summary['global_top_k_query_coverage']}")
    print(f"cluster_local_top_k_hit_coverage={summary['cluster_local_top_k_hit_coverage']}")
    print(f"coverage_lift_local_over_global={summary['coverage_lift_local_over_global']}")
    print(f"avg_cluster_top_k_overlap_with_global={summary['avg_cluster_top_k_overlap_with_global']}")
    print(f"local_top_entries_not_in_global_top_k={summary['local_top_entries_not_in_global_top_k']}")
    if base_cluster_counts is not None:
        print(f"base_vectors_assigned={summary['base_vectors_assigned']}")
        print(f"base_populated_clusters={summary['base_populated_clusters']}")
        print(f"base_cluster_avg_size={summary['base_cluster_avg_size']}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
