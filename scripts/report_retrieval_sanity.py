from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report feature and retrieval sanity checks for smoke/full retrieval experiments.",
    )
    parser.add_argument("--features", type=Path, required=True, help="Numpy feature matrix path.")
    parser.add_argument("--feature-metadata-csv", type=Path, default=None, help="Optional feature metadata CSV.")
    parser.add_argument("--retrieval-results-csv", type=Path, default=None, help="Optional retrieval results CSV.")
    parser.add_argument("--report-json", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument(
        "--collapse-cosine-threshold",
        type=float,
        default=0.9999,
        help="Warn when mean off-diagonal cosine exceeds this threshold.",
    )
    parser.add_argument(
        "--top1-concentration-threshold",
        type=float,
        default=0.20,
        help="Warn when a single top-1 retrieved id dominates more than this fraction of queries.",
    )
    parser.add_argument(
        "--unique-ratio-threshold",
        type=float,
        default=0.50,
        help="Warn when rounded unique feature ratio drops below this threshold.",
    )
    return parser.parse_args()


def load_features(path: Path) -> np.ndarray:
    features = np.load(path).astype(np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape={features.shape}")
    return features


def normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return features / norms


def compute_feature_report(features: np.ndarray, unique_ratio_threshold: float, collapse_cosine_threshold: float) -> tuple[dict[str, object], list[str]]:
    warnings: list[str] = []
    normalized = normalize_rows(features)
    norms = np.linalg.norm(features, axis=1)

    unique_rows_6 = int(np.unique(np.round(normalized, decimals=6), axis=0).shape[0])
    unique_ratio = float(unique_rows_6 / max(len(normalized), 1))

    similarity = normalized @ normalized.T
    if len(normalized) > 1:
        off_diag_mask = ~np.eye(len(normalized), dtype=bool)
        off_diag = similarity[off_diag_mask]
        off_diag_mean = float(off_diag.mean())
        off_diag_std = float(off_diag.std())
        off_diag_max = float(off_diag.max())
        off_diag_min = float(off_diag.min())
    else:
        off_diag_mean = 0.0
        off_diag_std = 0.0
        off_diag_max = 0.0
        off_diag_min = 0.0

    if unique_ratio < unique_ratio_threshold:
        warnings.append(
            f"Feature unique ratio is low ({unique_ratio:.4f} < {unique_ratio_threshold:.4f}); embeddings may be collapsing."
        )
    if off_diag_mean >= collapse_cosine_threshold:
        warnings.append(
            f"Mean off-diagonal cosine is extremely high ({off_diag_mean:.6f} >= {collapse_cosine_threshold:.6f}); embeddings may be near-constant."
        )

    report = {
        "feature_rows": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "norm_min": float(norms.min()) if len(norms) else 0.0,
        "norm_max": float(norms.max()) if len(norms) else 0.0,
        "norm_std": float(norms.std()) if len(norms) else 0.0,
        "unique_rows_rounded_6": unique_rows_6,
        "unique_ratio_rounded_6": unique_ratio,
        "off_diagonal_cosine_mean": off_diag_mean,
        "off_diagonal_cosine_std": off_diag_std,
        "off_diagonal_cosine_min": off_diag_min,
        "off_diagonal_cosine_max": off_diag_max,
    }
    return report, warnings


def load_optional_metadata(path: Path | None) -> tuple[dict[str, object], list[str]]:
    if path is None or not path.exists():
        return {}, []

    frame = pd.read_csv(path)
    report: dict[str, object] = {"metadata_rows": int(len(frame))}
    warnings: list[str] = []
    for column in ("split", "disaster_type", "damage_label", "disaster", "tile_id"):
        if column in frame.columns:
            counts = {str(key): int(value) for key, value in frame[column].value_counts().items()}
            report[f"{column}_counts"] = counts
            if len(counts) <= 1 and column in {"split", "disaster_type", "disaster"}:
                warnings.append(f"Metadata only covers one {column}; the evaluation slice is narrow.")
            if column == "tile_id" and len(counts) <= 2:
                warnings.append(f"Metadata covers only {len(counts)} tiles; tile-level shortcuts may dominate retrieval.")
    return report, warnings


def compute_retrieval_report(results_csv: Path, top1_concentration_threshold: float) -> tuple[dict[str, object], list[str]]:
    warnings: list[str] = []
    frame = pd.read_csv(results_csv)
    if frame.empty:
        return {"rows": 0, "queries": 0}, warnings

    query_groups = frame.groupby("query_positive_id", sort=False)
    top1 = frame.loc[frame["rank"] == 1].copy() if "rank" in frame.columns else pd.DataFrame()
    report: dict[str, object] = {
        "rows": int(len(frame)),
        "queries": int(query_groups.ngroups),
        "rows_per_query": float(len(frame) / max(query_groups.ngroups, 1)),
    }

    if not top1.empty and "retrieved_positive_id" in top1.columns:
        concentration = top1["retrieved_positive_id"].value_counts(normalize=True).max()
        report["top1_concentration"] = float(concentration)
        report["top1_unique_retrieved_ids"] = int(top1["retrieved_positive_id"].nunique())
        if concentration > top1_concentration_threshold:
            warnings.append(
                f"Top-1 retrieved ids are overly concentrated ({concentration:.4f} > {top1_concentration_threshold:.4f})."
            )

    if not top1.empty and "is_match" in top1.columns:
        top1_matches = top1["is_match"].astype(str).str.lower().eq("true")
        report["top1_accuracy"] = float(top1_matches.mean())

    if "is_match" in frame.columns and "rank" in frame.columns:
        first_hit_ranks: list[int | None] = []
        for _, query_frame in query_groups:
            hits = query_frame.loc[query_frame["is_match"].astype(str).str.lower().eq("true"), "rank"].astype(int)
            first_hit_ranks.append(int(hits.min()) if not hits.empty else None)
        for k in (1, 5, 10):
            hit_rate = sum(rank is not None and rank <= k for rank in first_hit_ranks) / max(len(first_hit_ranks), 1)
            report[f"recall@{k}_from_results"] = float(hit_rate)

    for query_column, retrieved_column, report_key in (
        ("query_tile_id", "retrieved_tile_id", "same_tile_rate"),
        ("query_building_id", "retrieved_building_id", "same_building_rate"),
        ("query_damage_label", "retrieved_damage_label", "same_damage_rate"),
    ):
        if not top1.empty and query_column in top1.columns and retrieved_column in top1.columns:
            report[report_key] = float((top1[query_column].astype(str) == top1[retrieved_column].astype(str)).mean())

    same_tile_rate = float(report.get("same_tile_rate", 0.0))
    same_building_rate = float(report.get("same_building_rate", 0.0))
    if same_tile_rate >= 0.75 and (same_tile_rate - same_building_rate) >= 0.50:
        warnings.append(
            "Top-1 retrieval stays on the same tile far more often than on the same building; the model may be learning location context instead of building identity."
        )

    return report, warnings


def main() -> None:
    args = parse_args()

    features = load_features(args.features)
    feature_report, warnings = compute_feature_report(
        features=features,
        unique_ratio_threshold=args.unique_ratio_threshold,
        collapse_cosine_threshold=args.collapse_cosine_threshold,
    )
    metadata_report, metadata_warnings = load_optional_metadata(args.feature_metadata_csv)

    report: dict[str, object] = {
        "features": str(args.features),
        "feature_report": feature_report,
        "metadata_report": metadata_report,
        "warnings": [*warnings, *metadata_warnings],
    }

    if args.retrieval_results_csv is not None and args.retrieval_results_csv.exists():
        retrieval_report, retrieval_warnings = compute_retrieval_report(
            results_csv=args.retrieval_results_csv,
            top1_concentration_threshold=args.top1_concentration_threshold,
        )
        report["retrieval_results_csv"] = str(args.retrieval_results_csv)
        report["retrieval_report"] = retrieval_report
        report["warnings"] = [*report["warnings"], *retrieval_warnings]

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Sanity report saved to: {args.report_json}")


if __name__ == "__main__":
    main()
