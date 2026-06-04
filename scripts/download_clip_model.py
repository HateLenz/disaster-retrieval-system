from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "openai/clip-vit-base-patch32"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pretrained" / "openai-clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional helper to mirror the CLIP pretrained visual backbone into the local pretrained directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face model repo id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the model snapshot will be stored.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional revision, tag, or commit hash.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        revision=args.revision,
    )
    print(f"Model snapshot saved to: {output_dir}")
    print(f"Hugging Face snapshot path: {snapshot_path}")


if __name__ == "__main__":
    main()
