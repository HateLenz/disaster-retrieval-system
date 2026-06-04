from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DIR = REPO_ROOT / "outputs" / "predictions"
LOG_DIR = REPO_ROOT / "outputs" / "logs"
DATA_DIR = REPO_ROOT / "data" / "processed"
FIGURE_DIR = REPO_ROOT / "docs" / "figures" / "algorithm_design"

COLORS = {
    "black": "#111111",
    "dark": "#222222",
    "mid": "#666666",
    "light": "#D9D9D9",
    "lighter": "#F3F3F3",
    "white": "#FFFFFF",
    "grid": "#E5E5E5",
}

HATCHES = ["", "///", "\\\\\\", "xx", "..", "--", "++", "oo"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<"]
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 10.5


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(REPO_ROOT).as_posix())


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#BDBDBD",
            "axes.labelcolor": COLORS["dark"],
            "axes.titleweight": "bold",
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "font.family": FONT_FAMILY,
            "font.serif": [FONT_FAMILY],
            "font.size": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "legend.title_fontsize": FONT_SIZE,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "hatch.linewidth": 0.8,
        }
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    hatch: str = "",
    fontsize: float = FONT_SIZE,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        hatch=hatch,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["dark"],
        linespacing=1.18,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = COLORS["black"]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def plot_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.965,
        "Cross-temporal Building Retrieval Pipeline",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=COLORS["dark"],
    )

    rows = [
        (
            0.70,
            "1  Sample construction",
            [
                ("xBD pre/post tiles\n+ annotations", COLORS["white"], COLORS["black"], ""),
                ("Match building UID\npre <-> post", COLORS["lighter"], COLORS["black"], "///"),
                ("Union bbox\n+ 16 px context", COLORS["white"], COLORS["black"], "\\\\\\"),
                ("224 x 224 paired\nbuilding patches", COLORS["lighter"], COLORS["black"], "xx"),
                ("Semantic labels\n(type, damage)", COLORS["white"], COLORS["black"], ".."),
            ],
        ),
        (
            0.41,
            "2  Stage-1 visual retrieval",
            [
                ("Pre/Post patches", COLORS["white"], COLORS["black"], ""),
                ("Shared CLIP ViT-B/32\nvisual encoder", COLORS["lighter"], COLORS["black"], "///"),
                ("Projection head\n256-d vector", COLORS["white"], COLORS["black"], "\\\\\\"),
                ("L2 normalization", COLORS["lighter"], COLORS["black"], "xx"),
                ("Symmetric\nInfoNCE loss", COLORS["white"], COLORS["black"], ".."),
            ],
        ),
        (
            0.12,
            "3  Stage-2 semantic query",
            [
                ("Post image query", COLORS["white"], COLORS["black"], ""),
                ("CLIP image\nembedding", COLORS["lighter"], COLORS["black"], "///"),
                ("Text prompt\nfrom labels", COLORS["white"], COLORS["black"], "\\\\\\"),
                ("Residual gated\nfusion", COLORS["lighter"], COLORS["black"], "xx"),
                ("FAISS Top-K\npre-disaster results", COLORS["white"], COLORS["black"], ".."),
            ],
        ),
    ]

    for y, label, boxes in rows:
        stage_id, stage_name = label.split("  ", maxsplit=1)
        ax.text(
            0.035,
            y + 0.065,
            f"{stage_id}\n{stage_name}",
            ha="left",
            va="center",
            fontsize=FONT_SIZE,
            fontweight="bold",
            color=COLORS["black"],
            linespacing=1.15,
        )
        x_positions = np.linspace(0.24, 0.82, len(boxes))
        box_width = 0.12
        for index, (text, face, edge, hatch) in enumerate(boxes):
            x = float(x_positions[index])
            rounded_box(ax, (x, y), box_width, 0.13, text, face, edge, hatch=hatch)
            if index < len(boxes) - 1:
                arrow(ax, (x + box_width, y + 0.065), (float(x_positions[index + 1]), y + 0.065))

    ax.text(
        0.5,
        0.035,
        "Stage-1 establishes the visual metric space; Stage-2 injects disaster semantics into the query side while keeping gallery retrieval cosine-compatible.",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color=COLORS["mid"],
    )
    save_figure(fig, "pipeline_overview.png")


def plot_dataset_distribution() -> None:
    tile_csv = DATA_DIR / "xbd_tile_dataset.csv"
    patch_csv = DATA_DIR / "xbd_building_patches.csv"
    tiles = pd.read_csv(tile_csv)
    patches = pd.read_csv(patch_csv)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    disaster_order = ["fire", "flooding", "wind"]
    disaster_faces = {"fire": COLORS["white"], "flooding": COLORS["light"], "wind": COLORS["lighter"]}
    disaster_hatches = {"fire": "", "flooding": "///", "wind": "xx"}
    split_order = ["hold", "test", "tier1", "tier3"]
    damage_order = ["no-damage", "minor-damage", "major-damage", "destroyed"]

    tile_counts = tiles.pivot_table(index="split", columns="disaster_type", values="tile_id", aggfunc="count", fill_value=0)
    tile_counts = tile_counts.reindex(index=split_order, columns=disaster_order).fillna(0)
    x0 = np.arange(len(tile_counts.index))
    bottom = np.zeros(len(tile_counts.index))
    for disaster_type in disaster_order:
        values = tile_counts[disaster_type].to_numpy()
        axes[0].bar(
            x0,
            values,
            bottom=bottom,
            width=0.72,
            label=disaster_type,
            facecolor=disaster_faces[disaster_type],
            edgecolor=COLORS["black"],
            hatch=disaster_hatches[disaster_type],
            linewidth=0.8,
        )
        bottom += values
    axes[0].set_title("Tile-level records")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(x0)
    axes[0].set_xticklabels(tile_counts.index, rotation=0)

    patch_counts = patches["damage_label"].value_counts().reindex(damage_order)
    bars = axes[1].bar(
        patch_counts.index,
        patch_counts.values,
        facecolor=COLORS["white"],
        edgecolor=COLORS["black"],
        width=0.68,
        linewidth=0.9,
    )
    for bar, hatch in zip(bars, ["", "///", "\\\\\\", "xx"], strict=True):
        bar.set_hatch(hatch)
    axes[1].set_title("Building-patch damage distribution")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(title="Disaster type", loc="upper left")
    fig.tight_layout(w_pad=2.4)
    save_figure(fig, "dataset_distribution.png")


def plot_smoke_subset_balance() -> None:
    smoke_csv = DATA_DIR / "xbd_building_patches_smoke_stage1.csv"
    smoke = pd.read_csv(smoke_csv)

    disaster_order = ["fire", "flooding", "wind"]
    disaster_faces = {"fire": COLORS["white"], "flooding": COLORS["light"], "wind": COLORS["lighter"]}
    disaster_hatches = {"fire": "", "flooding": "///", "wind": "xx"}
    damage_order = ["no-damage", "minor-damage", "major-damage", "destroyed"]

    smoke_counts = smoke.pivot_table(index="damage_label", columns="disaster_type", values="positive_id", aggfunc="count", fill_value=0)
    smoke_counts = smoke_counts.reindex(index=damage_order, columns=disaster_order).fillna(0)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(smoke_counts.index))
    bottom = np.zeros(len(smoke_counts.index))
    for disaster_type in disaster_order:
        values = smoke_counts[disaster_type].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.72,
            label=disaster_type,
            facecolor=disaster_faces[disaster_type],
            edgecolor=COLORS["black"],
            hatch=disaster_hatches[disaster_type],
            linewidth=0.8,
        )
        bottom += values

    ax.set_title("Smoke subset balance")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(smoke_counts.index, rotation=20, ha="right")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Disaster type", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    save_figure(fig, "smoke_subset_balance.png")


def metric_row(name: str, path: str) -> dict[str, float | str]:
    overall = load_json(PREDICTION_DIR / path)["overall"]
    return {
        "model": name,
        "Recall@1": overall["recall@1"],
        "Recall@5": overall["recall@5"],
        "Recall@10": overall["recall@10"],
        "MRR": overall["mrr"],
    }


def plot_stage_comparison() -> None:
    rows = [
        metric_row("Stage-1", "clip_retrieval_metrics_smoke_stage1.json"),
        metric_row("Stage-1 v2", "clip_retrieval_metrics_smoke_stage1_v2.json"),
        metric_row("Stage-2 v1", "clip_retrieval_metrics_smoke_stage2_semantic.json"),
        metric_row("Stage-2 v2", "clip_retrieval_metrics_smoke_stage2_semantic_v2.json"),
        metric_row("Stage-2 v3", "clip_retrieval_metrics_smoke_stage2_semantic_v3.json"),
        metric_row("Stage-2 v4", "clip_retrieval_metrics_smoke_stage2_semantic_v4_stage1init.json"),
        metric_row("Stage-2 v5", "clip_retrieval_metrics_smoke_stage2_semantic_v5_stage1init_type_damage.json"),
    ]
    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.4, 1.0]})
    axes[0].plot(x, frame["Recall@1"], marker="o", linestyle="-", linewidth=2, color=COLORS["black"], label="Recall@1")
    axes[0].plot(x, frame["Recall@5"], marker="s", linestyle="--", linewidth=2, color=COLORS["black"], label="Recall@5")
    axes[0].plot(x, frame["Recall@10"], marker="^", linestyle="-.", linewidth=2, color=COLORS["black"], label="Recall@10")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(frame["model"], rotation=25, ha="right")
    axes[0].set_ylim(0, 0.70)
    axes[0].set_title("Model variants on hold retrieval")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncols=3)

    bars = axes[1].bar(
        frame["model"],
        frame["MRR"],
        facecolor=COLORS["white"],
        edgecolor=COLORS["black"],
        width=0.68,
        linewidth=0.9,
    )
    for index, bar in enumerate(bars):
        bar.set_hatch(HATCHES[index % len(HATCHES)])
    axes[1].set_ylim(0, 0.45)
    axes[1].set_title("MRR comparison")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=2.4)
    save_figure(fig, "stage_model_comparison.png")


def plot_group_metrics() -> None:
    data = load_json(PREDICTION_DIR / "clip_retrieval_metrics_smoke_stage1_v2.json")
    disaster = pd.DataFrame(data["by_disaster_type"]).T[["recall@1", "recall@5", "recall@10", "mrr"]]
    damage = pd.DataFrame(data["by_damage_label"]).T[["recall@1", "recall@5", "recall@10", "mrr"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.11, right=0.90, bottom=0.18, top=0.86, wspace=0.34)
    column_labels = ["R@1", "R@5", "R@10", "MRR"]
    for ax, frame, title in [
        (axes[0], disaster, "Stage-1 v2 by disaster type"),
        (axes[1], damage, "Stage-1 v2 by damage label"),
    ]:
        image = ax.imshow(frame.values, cmap="Greys", vmin=0.15, vmax=0.80, aspect="auto")
        ax.set_xticks(np.arange(len(frame.columns)))
        ax.set_xticklabels(column_labels, rotation=0, ha="center")
        ax.set_yticks(np.arange(len(frame.index)))
        ax.set_yticklabels(frame.index)
        ax.set_title(title, pad=12)
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", pad=6)
        ax.set_xlim(-0.5, len(frame.columns) - 0.5)
        ax.set_ylim(len(frame.index) - 0.5, -0.5)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                value = frame.iloc[i, j]
                text_color = COLORS["white"] if value >= 0.55 else COLORS["black"]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=FONT_SIZE, color=text_color)
    cax = fig.add_axes([0.925, 0.22, 0.015, 0.56])
    fig.colorbar(image, cax=cax, label="Metric value")
    save_figure(fig, "group_metric_heatmaps.png")


def plot_filtering_and_tile_aggregation() -> None:
    full = load_json(PREDICTION_DIR / "clip_retrieval_metrics_smoke_stage1_v2.json")["overall"]
    prior = load_json(PREDICTION_DIR / "clip_retrieval_metrics_smoke_stage1_v2_disaster_prior.json")
    patch_prior = prior["patch"]["overall"]
    tile_plain = load_json(PREDICTION_DIR / "clip_tile_retrieval_metrics_smoke_stage1_v2.json")["overall"]
    tile_prior = prior["tile"]["overall"]

    metrics = ["recall@1", "recall@5", "recall@10", "mrr"]
    labels = ["R@1", "R@5", "R@10", "MRR"]
    x = np.arange(len(metrics))
    width = 0.20

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    series = [
        ("Patch full", full, COLORS["white"], ""),
        ("Patch + disaster prior", patch_prior, COLORS["light"], "///"),
        ("Tile top_m", tile_plain, COLORS["white"], "\\\\\\"),
        ("Tile top_m + prior", tile_prior, COLORS["lighter"], "xx"),
    ]
    for index, (name, values, facecolor, hatch) in enumerate(series):
        offsets = x + (index - 1.5) * width
        ax.bar(
            offsets,
            [values[m] for m in metrics],
            width=width,
            label=name,
            facecolor=facecolor,
            edgecolor=COLORS["black"],
            hatch=hatch,
            linewidth=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Candidate prior and tile aggregation")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncols=2, loc="upper left")
    save_figure(fig, "filtering_tile_aggregation.png")


def plot_training_curves() -> None:
    histories = [
        ("Stage-1", LOG_DIR / "smoke_stage1_visual" / "train_history.jsonl"),
        ("Stage-1 v2", LOG_DIR / "smoke_stage1_visual_v2" / "train_history.jsonl"),
        ("Stage-2 v1", LOG_DIR / "smoke_stage2_semantic" / "train_history.jsonl"),
        ("Stage-2 v2", LOG_DIR / "smoke_stage2_semantic_v2" / "train_history.jsonl"),
        ("Stage-2 v3", LOG_DIR / "smoke_stage2_semantic_v3" / "train_history.jsonl"),
        ("Stage-2 v4", LOG_DIR / "smoke_stage2_semantic_v4_stage1init" / "train_history.jsonl"),
        ("Stage-2 v5", LOG_DIR / "smoke_stage2_semantic_v5_stage1init_type_damage" / "train_history.jsonl"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=False)
    stage_axes = {
        "Stage-1": (axes[0, 0], axes[0, 1]),
        "Stage-2": (axes[1, 0], axes[1, 1]),
    }
    for index, (name, path) in enumerate(histories):
        if not path.exists():
            continue
        rows = load_jsonl(path)
        epochs = [row["epoch"] for row in rows]
        recall1 = [row.get("eval", {}).get("recall@1", np.nan) for row in rows]
        loss = [row.get("train", {}).get("loss", np.nan) for row in rows]
        stage = "Stage-1" if name.startswith("Stage-1") else "Stage-2"
        recall_ax, loss_ax = stage_axes[stage]
        style_index = index if stage == "Stage-1" else index - 2
        line_kwargs = {
            "marker": MARKERS[style_index % len(MARKERS)],
            "linestyle": LINE_STYLES[style_index % len(LINE_STYLES)],
            "linewidth": 1.6,
            "markersize": 4.5,
            "color": COLORS["black"],
            "label": name,
        }
        recall_ax.plot(epochs, recall1, **line_kwargs)
        loss_ax.plot(epochs, loss, **line_kwargs)

    axes[0, 0].set_title("Stage-1 validation Recall@1")
    axes[0, 1].set_title("Stage-1 training loss")
    axes[1, 0].set_title("Stage-2 validation Recall@1")
    axes[1, 1].set_title("Stage-2 training loss")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recall@1")
        ax.set_ylim(0, 0.36)
    for ax in axes[:, 1]:
        ax.set_ylabel("Loss")
    for ax in axes.ravel():
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", ncols=2 if len(ax.lines) > 3 else 1)
    fig.tight_layout(h_pad=3.0, w_pad=2.4)
    save_figure(fig, "training_curves.png")


def main() -> None:
    apply_style()
    plot_pipeline_overview()
    plot_dataset_distribution()
    plot_smoke_subset_balance()
    plot_stage_comparison()
    plot_group_metrics()
    plot_filtering_and_tile_aggregation()
    plot_training_curves()


if __name__ == "__main__":
    main()
