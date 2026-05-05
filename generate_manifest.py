"""
Generate a flat dataset manifest for training and large-scale pipeline reuse.

Examples:
    python generate_manifest.py
    python generate_manifest.py --format json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import IMAGES_DIR, LABELS_DIR, MANIFESTS_DIR
from src.data import DatasetBuilder, DatasetManifestBuilder, summarize_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset manifest from labels and images")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to manifests/dataset_manifest.<format>",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_builder = DatasetBuilder(LABELS_DIR, IMAGES_DIR)
    manifest_builder = DatasetManifestBuilder(dataset_builder)
    rows = manifest_builder.build_rows()

    if not rows:
        print("No manifest rows were generated. Check dataset paths and labels.")
        return 1

    output_path = args.output or (MANIFESTS_DIR / f"dataset_manifest.{args.format}")
    manifest_builder.save(output_path, rows)

    summary = summarize_manifest(rows)
    print(f"Manifest saved to: {output_path}")
    print(f"Rows: {summary['total_rows']}")
    print(f"Videos: {summary['video_count']}")
    print(f"Class distribution: {summary['class_distribution']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
