from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize retrieval results into thesis-ready evidence for pure-image limitations.",
    )
    parser.add_argument("--metrics-json", type=Path, required=True, help="Metrics JSON from eval_clip_retrieval.py.")
    parser.add_argument("--results-csv", type=Path, required=True, help="Per-rank retrieval results CSV.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Structured summary JSON output.")
    parser.add_argument("--summary-md", type=Path, required=True, help="Markdown summary output.")
    parser.add_argument("--failed-cases-csv", type=Path, default=None, help="Optional failed-case summary CSV.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k if needed.")
    parser.add_argument("--example-count", type=int, default=10, help="How many failed examples to embed in markdown.")
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_query_summary(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_id, query_frame in frame.groupby("query_positive_id", sort=False):
        ordered = query_frame.sort_values("rank", kind="stable").reset_index(drop=True)
        top1 = ordered.iloc[0]
        matches = ordered.loc[bool_series(ordered["is_match"])]
        first_hit_rank = int(matches["rank"].min()) if not matches.empty else None

        rows.append(
            {
                "query_positive_id": str(query_id),
                "query_building_id": str(top1["query_building_id"]),
                "query_tile_id": str(top1["query_tile_id"]),
                "query_disaster": str(top1["query_disaster"]),
                "query_disaster_type": str(top1["query_disaster_type"]),
                "query_damage_label": str(top1["query_damage_label"]),
                "top1_retrieved_positive_id": str(top1["retrieved_positive_id"]),
                "top1_retrieved_building_id": str(top1["retrieved_building_id"]),
                "top1_retrieved_tile_id": str(top1["retrieved_tile_id"]),
                "top1_retrieved_disaster_type": str(top1["retrieved_disaster_type"]),
                "top1_retrieved_damage_label": str(top1["retrieved_damage_label"]),
                "top1_score": float(top1["score"]),
                "top1_match": bool(str(top1["is_match"]).lower() == "true"),
                "same_tile_top1": bool(str(top1["query_tile_id"]) == str(top1["retrieved_tile_id"])),
                "same_building_top1": bool(str(top1["query_building_id"]) == str(top1["retrieved_building_id"])),
                "same_damage_top1": bool(str(top1["query_damage_label"]) == str(top1["retrieved_damage_label"])),
                "first_hit_rank": first_hit_rank,
                "hit@1": bool(first_hit_rank == 1),
                "hit@5": bool(first_hit_rank is not None and first_hit_rank <= min(5, top_k)),
                f"hit@{top_k}": bool(first_hit_rank is not None and first_hit_rank <= top_k),
            }
        )
    return pd.DataFrame(rows)


def summarize_groups(query_summary: pd.DataFrame, group_column: str, top_k: int) -> list[dict[str, object]]:
    if group_column not in query_summary.columns:
        return []

    rows: list[dict[str, object]] = []
    for group_value, group_frame in query_summary.groupby(group_column, sort=True):
        rows.append(
            {
                group_column: str(group_value),
                "count": int(len(group_frame)),
                "top1_accuracy": float(group_frame["top1_match"].mean()),
                "recall@5": float(group_frame["hit@5"].mean()),
                f"recall@{top_k}": float(group_frame[f"hit@{top_k}"].mean()),
                "same_tile_rate": float(group_frame["same_tile_top1"].mean()),
                "same_building_rate": float(group_frame["same_building_top1"].mean()),
                "same_damage_rate": float(group_frame["same_damage_top1"].mean()),
                "avg_top1_score": float(group_frame["top1_score"].mean()),
            }
        )
    return rows


def hit_rank_distribution(query_summary: pd.DataFrame, top_k: int) -> list[dict[str, object]]:
    counts = query_summary["first_hit_rank"].fillna(f"miss@{top_k}").value_counts(dropna=False)
    total = max(len(query_summary), 1)
    rows: list[dict[str, object]] = []
    for key, value in counts.items():
        rows.append({"bucket": str(key), "count": int(value), "rate": float(value / total)})
    return rows


def build_findings(metrics: dict[str, object], query_summary: pd.DataFrame, top_k: int) -> list[str]:
    findings: list[str] = []
    overall = metrics.get("overall", {}) if isinstance(metrics.get("overall"), dict) else {}

    top1_accuracy = float(query_summary["top1_match"].mean()) if not query_summary.empty else 0.0
    same_tile_rate = float(query_summary["same_tile_top1"].mean()) if not query_summary.empty else 0.0
    same_building_rate = float(query_summary["same_building_top1"].mean()) if not query_summary.empty else 0.0
    same_damage_rate = float(query_summary["same_damage_top1"].mean()) if not query_summary.empty else 0.0

    if top1_accuracy < 0.25:
        findings.append(
            f"Top-1 exact-match accuracy is only {top1_accuracy:.3f}, so pure image retrieval still misses the correct building in most queries."
        )
    if same_tile_rate - same_building_rate >= 0.30:
        findings.append(
            f"Top-1 same-tile rate ({same_tile_rate:.3f}) is far higher than same-building rate ({same_building_rate:.3f}), which indicates the model relies on location context more than building identity."
        )
    if same_damage_rate - same_building_rate >= 0.20:
        findings.append(
            f"Top-1 same-damage rate ({same_damage_rate:.3f}) is much higher than same-building rate ({same_building_rate:.3f}), so the model captures coarse damage semantics better than instance-level matching."
        )

    if "query_damage_label" in query_summary.columns:
        by_damage = summarize_groups(query_summary, "query_damage_label", top_k=top_k)
        by_damage_map = {str(item["query_damage_label"]): item for item in by_damage}
        if "destroyed" in by_damage_map and "no-damage" in by_damage_map:
            destroyed_top1 = float(by_damage_map["destroyed"]["top1_accuracy"])
            no_damage_top1 = float(by_damage_map["no-damage"]["top1_accuracy"])
            if no_damage_top1 - destroyed_top1 >= 0.10:
                findings.append(
                    f"Destroyed queries underperform no-damage queries on top-1 accuracy ({destroyed_top1:.3f} vs {no_damage_top1:.3f}), which is consistent with severe post-disaster appearance change hurting pure visual matching."
                )

    recall_at_top_k = float(overall.get(f"recall@{top_k}", 0.0))
    if recall_at_top_k < 0.50:
        findings.append(
            f"Even at top-{top_k}, recall is only {recall_at_top_k:.3f}, so pure image retrieval still fails to recover the correct building for a large fraction of queries."
        )

    if not findings:
        findings.append("The current run did not cross the configured thresholds for an explicit limitation claim.")
    return findings


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No data_"
    rendered = frame.fillna("").copy()
    headers = [str(column) for column in rendered.columns]
    rows = [[str(value) for value in row] for row in rendered.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), max((len(row[index]) for row in rows), default=0))
        for index in range(len(headers))
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def write_markdown(
    summary_md: Path,
    metrics: dict[str, object],
    query_summary: pd.DataFrame,
    damage_rows: list[dict[str, object]],
    disaster_rows: list[dict[str, object]],
    hit_rows: list[dict[str, object]],
    failed_examples: pd.DataFrame,
    findings: list[str],
    top_k: int,
) -> None:
    overall = metrics.get("overall", {}) if isinstance(metrics.get("overall"), dict) else {}
    top1_accuracy = float(query_summary["top1_match"].mean()) if not query_summary.empty else 0.0
    same_tile_rate = float(query_summary["same_tile_top1"].mean()) if not query_summary.empty else 0.0
    same_building_rate = float(query_summary["same_building_top1"].mean()) if not query_summary.empty else 0.0
    same_damage_rate = float(query_summary["same_damage_top1"].mean()) if not query_summary.empty else 0.0

    lines = [
        "# Pure Image Retrieval Limitation Summary",
        "",
        "## Overall",
        "",
        f"- Queries: {len(query_summary)}",
        f"- recall@1: {float(overall.get('recall@1', 0.0)):.4f}",
        f"- recall@5: {float(overall.get('recall@5', 0.0)):.4f}",
        f"- recall@{top_k}: {float(overall.get(f'recall@{top_k}', 0.0)):.4f}",
        f"- mrr: {float(overall.get('mrr', 0.0)):.4f}",
        f"- top1_accuracy: {top1_accuracy:.4f}",
        f"- same_tile_rate: {same_tile_rate:.4f}",
        f"- same_building_rate: {same_building_rate:.4f}",
        f"- same_damage_rate: {same_damage_rate:.4f}",
        "",
        "## Evidence",
        "",
    ]
    lines.extend([f"- {finding}" for finding in findings])
    lines.extend(
        [
            "",
            "## Hit Rank Distribution",
            "",
            markdown_table(pd.DataFrame(hit_rows)),
            "",
            "## By Damage Label",
            "",
            markdown_table(pd.DataFrame(damage_rows)),
            "",
            "## By Disaster Type",
            "",
            markdown_table(pd.DataFrame(disaster_rows)),
            "",
            "## Failed Query Examples",
            "",
            markdown_table(failed_examples),
            "",
        ]
    )

    summary_md.parent.mkdir(parents=True, exist_ok=True)
    with summary_md.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    args = parse_args()

    metrics = load_metrics(args.metrics_json)
    frame = pd.read_csv(args.results_csv)
    if frame.empty:
        raise ValueError("Retrieval results CSV is empty.")

    top_k = args.top_k or int(frame["rank"].max())
    query_summary = build_query_summary(frame, top_k=top_k)

    damage_rows = summarize_groups(query_summary, "query_damage_label", top_k=top_k)
    disaster_rows = summarize_groups(query_summary, "query_disaster_type", top_k=top_k)
    hit_rows = hit_rank_distribution(query_summary, top_k=top_k)
    findings = build_findings(metrics=metrics, query_summary=query_summary, top_k=top_k)

    failed_cases = query_summary.loc[~query_summary["top1_match"]].copy()
    failed_cases = failed_cases.sort_values(
        by=["same_tile_top1", "same_damage_top1", "top1_score"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    if args.failed_cases_csv is not None:
        args.failed_cases_csv.parent.mkdir(parents=True, exist_ok=True)
        failed_cases.to_csv(args.failed_cases_csv, index=False)

    summary = {
        "metrics_json": str(args.metrics_json),
        "results_csv": str(args.results_csv),
        "query_count": int(len(query_summary)),
        "top_k": top_k,
        "overall": metrics.get("overall", {}),
        "macro": metrics.get("macro", {}),
        "by_disaster_type": metrics.get("by_disaster_type", {}),
        "by_damage_label": metrics.get("by_damage_label", {}),
        "top1_behavior": {
            "top1_accuracy": float(query_summary["top1_match"].mean()),
            "same_tile_rate": float(query_summary["same_tile_top1"].mean()),
            "same_building_rate": float(query_summary["same_building_top1"].mean()),
            "same_damage_rate": float(query_summary["same_damage_top1"].mean()),
            "location_context_gap": float(query_summary["same_tile_top1"].mean() - query_summary["same_building_top1"].mean()),
            "damage_semantics_gap": float(query_summary["same_damage_top1"].mean() - query_summary["same_building_top1"].mean()),
        },
        "hit_rank_distribution": hit_rows,
        "group_summaries": {
            "by_damage_label": damage_rows,
            "by_disaster_type": disaster_rows,
        },
        "findings": findings,
        "failed_case_count": int(len(failed_cases)),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    example_columns = [
        "query_positive_id",
        "query_disaster_type",
        "query_damage_label",
        "top1_retrieved_positive_id",
        "top1_retrieved_disaster_type",
        "top1_retrieved_damage_label",
        "top1_score",
        "same_tile_top1",
        "same_building_top1",
        "same_damage_top1",
        "first_hit_rank",
    ]
    failed_examples = failed_cases.loc[:, example_columns].head(args.example_count)
    write_markdown(
        summary_md=args.summary_md,
        metrics=metrics,
        query_summary=query_summary,
        damage_rows=damage_rows,
        disaster_rows=disaster_rows,
        hit_rows=hit_rows,
        failed_examples=failed_examples,
        findings=findings,
        top_k=top_k,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Limitation summary saved to: {args.summary_json}")
    print(f"Markdown summary saved to: {args.summary_md}")
    if args.failed_cases_csv is not None:
        print(f"Failed cases saved to: {args.failed_cases_csv}")


if __name__ == "__main__":
    main()
