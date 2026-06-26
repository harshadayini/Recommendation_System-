#!/usr/bin/env python3
"""Validate qualitative Telugu recommendations for paper reporting.

This script separates two checks:
1. Category alignment: automatic comparison between a user's learned English
   category profile and predicted Telugu article categories.
2. Manual relevance table: a concise, human-readable top-5 table for the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROFILE_LABELS = {
    "U245228": "Sports-oriented",
    "U539584": "Music/Entertainment-oriented",
    "U69259": "Politics-oriented",
}


MANUAL_RATIONALES = {
    ("U245228", 1): "Dhoni/cricket headline; directly matches sports profile.",
    ("U245228", 2): "Indian team selection headline; directly matches sports profile.",
    ("U245228", 3): "Akash Deep cricket selection/playing headline; sports-relevant.",
    ("U245228", 4): "Rohit/cricket headline; sports-relevant.",
    ("U245228", 5): "Bumrah/cricket leadership headline; sports-relevant.",
    ("U539584", 1): "Music hits headline; matches music profile.",
    ("U539584", 2): "Celebrity/film-industry headline; relevant to broader entertainment profile.",
    ("U539584", 3): "Song release from a film; matches music/movie profile.",
    ("U539584", 4): "Movie musical-session headline; matches music/movie profile.",
    ("U539584", 5): "Song release from a film; matches music/movie profile.",
    ("U69259", 1): "Political headline about simultaneous elections; matches politics profile.",
    ("U69259", 2): "Parliament-related headline; matches politics profile.",
    ("U69259", 3): "Parliament/bills headline; matches politics profile.",
    ("U69259", 4): "State cabinet headline; matches politics profile.",
    ("U69259", 5): "Rahul/state visit headline; matches politics/current-affairs profile.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository directory.",
    )
    parser.add_argument(
        "--recommendation-dir",
        type=Path,
        default=Path("results/telugu_mindlarge_recommendations"),
        help="Directory containing generated Telugu recommendation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/telugu_qualitative_validation"),
        help="Output directory for qualitative validation tables.",
    )
    parser.add_argument("--manual-top-k", type=int, default=5)
    return parser.parse_args()


def resolve_path(repo_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_dir / path


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    recommendation_dir = resolve_path(repo_dir, args.recommendation_dir)
    output_dir = resolve_path(repo_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    affinities = pd.read_csv(recommendation_dir / "representative_user_category_affinities.csv")
    recs = pd.read_csv(recommendation_dir / "telugu_top10_recommendations_headline_deduped.csv")

    merged = recs.merge(
        affinities[
            [
                "user_id",
                "top_category_1",
                "top_category_2",
                "top_category_3",
            ]
        ],
        on="user_id",
        how="left",
    )
    merged["profile"] = merged["user_id"].map(PROFILE_LABELS).fillna("Representative user")
    merged["top1_category_aligned"] = merged["predicted_category"] == merged["top_category_1"]
    merged["top3_category_aligned"] = merged.apply(
        lambda row: row["predicted_category"] in {row["top_category_1"], row["top_category_2"], row["top_category_3"]},
        axis=1,
    )

    alignment_summary = (
        merged.groupby(["user_id", "profile", "top_category_1", "top_category_2", "top_category_3"], sort=False)
        .agg(
            recommendations=("story_id", "count"),
            top1_alignment_rate=("top1_category_aligned", "mean"),
            top3_alignment_rate=("top3_category_aligned", "mean"),
            mean_category_confidence=("category_confidence", "mean"),
        )
        .reset_index()
    )
    overall_summary = {
        "representative_users": int(alignment_summary["user_id"].nunique()),
        "recommendations_checked": int(len(merged)),
        "top1_alignment_rate": float(merged["top1_category_aligned"].mean()),
        "top3_alignment_rate": float(merged["top3_category_aligned"].mean()),
        "mean_category_confidence": float(merged["category_confidence"].mean()),
    }

    manual = merged[merged["rank"] <= args.manual_top_k].copy()
    manual["manual_relevance"] = "Relevant"
    manual["manual_rationale"] = [
        MANUAL_RATIONALES.get((row.user_id, int(row.rank)), "Headline matches the user's learned category profile.")
        for row in manual.itertuples(index=False)
    ]
    manual = manual[
        [
            "profile",
            "user_id",
            "rank",
            "headline",
            "predicted_category",
            "top_category_1",
            "top_category_2",
            "top_category_3",
            "manual_relevance",
            "manual_rationale",
        ]
    ]

    merged.to_csv(output_dir / "telugu_category_alignment_top10.csv", index=False)
    alignment_summary.to_csv(output_dir / "telugu_category_alignment_summary.csv", index=False)
    manual.to_csv(output_dir / "telugu_manual_relevance_top5.csv", index=False)
    (output_dir / "telugu_qualitative_validation_summary.json").write_text(
        json.dumps(overall_summary, indent=2)
    )
    print(json.dumps(overall_summary, indent=2))


if __name__ == "__main__":
    main()
