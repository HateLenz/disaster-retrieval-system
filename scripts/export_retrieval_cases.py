from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "outputs" / "predictions" / "clip_retrieval_results.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "figures" / "retrieval_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export retrieval case contact sheets.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-cases", type=int, default=50)
    parser.add_argument("--incorrect-only", action="store_true")
    parser.add_argument("--correct-only", action="store_true")
    parser.add_argument("--disaster-type", type=str, default=None)
    return parser.parse_args()


def frame_image(image: Image.Image, color: str) -> Image.Image:
    return ImageOps.expand(image.convert("RGB"), border=4, fill=color)


def make_contact_sheet(query_path: Path, rows: pd.DataFrame, output_path: Path) -> None:
    query_image = frame_image(Image.open(query_path), color="blue")
    gallery_images: list[Image.Image] = []

    for row in rows.itertuples(index=False):
        border = "green" if bool(row.is_match) else "red"
        gallery_images.append(frame_image(Image.open(Path(row.retrieved_patch_path)), color=border))

    width = query_image.width + sum(image.width for image in gallery_images)
    height = max([query_image.height, *[image.height for image in gallery_images]]) + 36
    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)

    offset_x = 0
    canvas.paste(query_image, (offset_x, 36))
    draw.text((offset_x + 8, 8), "Query post", fill="black")
    offset_x += query_image.width

    for row, gallery_image in zip(rows.itertuples(index=False), gallery_images, strict=True):
        canvas.paste(gallery_image, (offset_x, 36))
        caption = f"R{row.rank} {row.score:.3f}"
        draw.text((offset_x + 8, 8), caption, fill="black")
        offset_x += gallery_image.width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results_csv)
    results = results.loc[results["rank"] <= args.top_k].copy()
    if args.disaster_type is not None:
        results = results.loc[results["query_disaster_type"] == args.disaster_type]

    top1 = results.loc[results["rank"] == 1, ["query_positive_id", "is_match"]].copy()
    if args.incorrect_only:
        keep_ids = top1.loc[~top1["is_match"], "query_positive_id"]
        results = results.loc[results["query_positive_id"].isin(keep_ids)]
    if args.correct_only:
        keep_ids = top1.loc[top1["is_match"], "query_positive_id"]
        results = results.loc[results["query_positive_id"].isin(keep_ids)]

    selected_ids = results["query_positive_id"].drop_duplicates().head(args.num_cases)
    summary_rows: list[dict[str, str | int | float | bool]] = []
    for query_positive_id in selected_ids:
        query_rows = results.loc[results["query_positive_id"] == query_positive_id].sort_values("rank")
        if query_rows.empty:
            continue

        first_row = query_rows.iloc[0]
        image_name = f"{query_positive_id}.png"
        output_path = args.output_dir / image_name
        make_contact_sheet(Path(first_row["query_patch_path"]), query_rows, output_path)

        summary_rows.append(
            {
                "query_positive_id": query_positive_id,
                "query_tile_id": first_row["query_tile_id"],
                "query_disaster_type": first_row["query_disaster_type"],
                "top1_match": bool(first_row["is_match"]),
                "output_path": str(output_path),
            }
        )

    summary_path = args.output_dir / "cases.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Exported {len(summary_rows)} retrieval cases to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
