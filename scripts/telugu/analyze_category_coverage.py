#!/usr/bin/env python3
"""Analyze Telugu category coverage after MINDlarge refined classification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository directory.",
    )
    parser.add_argument(
        "--telugu-dir",
        type=Path,
        default=Path("../telugu_mindlarge_refined"),
        help="Directory containing Telugu classification outputs.",
    )
    parser.add_argument(
        "--english-counts",
        type=Path,
        default=Path("results/telugu_mindlarge_refined/mindlarge_refined_category_counts.csv"),
        help="Optional MINDlarge English refined training category counts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/telugu_category_coverage"),
        help="Directory for coverage outputs.",
    )
    return parser.parse_args()


def resolve_path(repo_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_dir / path


def distribution_metrics(counts: pd.Series, total_categories: int) -> dict[str, float | int]:
    total = int(counts.sum())
    nonzero = counts[counts > 0]
    probabilities = nonzero / total if total else nonzero
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if total else 0.0
    max_entropy = math.log2(total_categories) if total_categories > 1 else 0.0
    top_1_share = float((counts.max() / total) if total else 0.0)
    top_5_share = float((counts.sort_values(ascending=False).head(5).sum() / total) if total else 0.0)
    return {
        "total_items": total,
        "total_categories": total_categories,
        "used_categories": int((counts > 0).sum()),
        "unused_categories": int((counts == 0).sum()),
        "coverage_rate": float((counts > 0).sum() / total_categories) if total_categories else 0.0,
        "entropy_bits": entropy,
        "normalized_entropy": float(entropy / max_entropy) if max_entropy else 0.0,
        "effective_category_count": float(2**entropy),
        "top_1_share": top_1_share,
        "top_5_share": top_5_share,
    }


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    telugu_dir = resolve_path(repo_dir, args.telugu_dir)
    english_counts = resolve_path(repo_dir, args.english_counts)
    output_dir = resolve_path(repo_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = json.loads((telugu_dir / "label_mapping.json").read_text())
    categories = label_mapping["categories"]
    predictions = pd.read_csv(telugu_dir / "telugu_category_predictions.csv", dtype={"story_id": str})

    counts = predictions["predicted_category"].value_counts().reindex(categories, fill_value=0)
    coverage = pd.DataFrame(
        {
            "category": counts.index,
            "telugu_count": counts.values,
            "telugu_share": counts.values / counts.sum(),
        }
    )

    if english_counts.exists():
        english = pd.read_csv(english_counts)
        english = english.rename(columns={english.columns[0]: "category", english.columns[1]: "english_train_count"})
        coverage = coverage.merge(english, on="category", how="left")
        coverage["english_train_count"] = coverage["english_train_count"].fillna(0).astype(int)
        coverage["english_train_share"] = coverage["english_train_count"] / coverage["english_train_count"].sum()

    summary = distribution_metrics(counts, len(categories))
    summary["unused_category_names"] = counts[counts == 0].index.tolist()
    summary["top_categories"] = (
        coverage.sort_values("telugu_count", ascending=False)
        .head(10)[["category", "telugu_count", "telugu_share"]]
        .to_dict(orient="records")
    )

    coverage.to_csv(output_dir / "telugu_category_coverage.csv", index=False)
    (output_dir / "telugu_category_coverage_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
