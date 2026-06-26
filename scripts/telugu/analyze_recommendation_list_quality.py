#!/usr/bin/env python3
"""Analyze Telugu recommendation-list duplicate rate and inter-user overlap."""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

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
        "--recommendation-dir",
        type=Path,
        default=Path("results/telugu_mindlarge_recommendations"),
        help="Directory containing generated Telugu recommendation CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/telugu_recommendation_list_quality"),
        help="Output directory.",
    )
    return parser.parse_args()


def resolve_path(repo_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_dir / path


def normalize_headline(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def per_user_duplicate_metrics(recs: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for user_id, group in recs.groupby("user_id", sort=False):
        top_k = len(group)
        unique_story_ids = group["story_id"].astype(str).nunique()
        unique_headlines = group["headline_normalized"].nunique()
        rows.append(
            {
                "list_type": label,
                "user_id": user_id,
                "top_k": top_k,
                "unique_story_ids": unique_story_ids,
                "story_id_uniqueness": unique_story_ids / top_k if top_k else 0.0,
                "story_id_duplicate_rate": 1 - (unique_story_ids / top_k) if top_k else 0.0,
                "unique_headlines": unique_headlines,
                "headline_uniqueness": unique_headlines / top_k if top_k else 0.0,
                "headline_duplicate_rate": 1 - (unique_headlines / top_k) if top_k else 0.0,
            }
        )
    return pd.DataFrame(rows)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def inter_user_overlap(recs: pd.DataFrame, label: str) -> pd.DataFrame:
    story_sets = {
        user_id: set(group["story_id"].astype(str))
        for user_id, group in recs.groupby("user_id", sort=False)
    }
    headline_sets = {
        user_id: set(group["headline_normalized"].astype(str))
        for user_id, group in recs.groupby("user_id", sort=False)
    }
    rows = []
    for left, right in combinations(story_sets.keys(), 2):
        rows.append(
            {
                "list_type": label,
                "user_a": left,
                "user_b": right,
                "story_id_jaccard": jaccard(story_sets[left], story_sets[right]),
                "headline_jaccard": jaccard(headline_sets[left], headline_sets[right]),
                "shared_story_ids": len(story_sets[left] & story_sets[right]),
                "shared_headlines": len(headline_sets[left] & headline_sets[right]),
            }
        )
    return pd.DataFrame(rows)


def analyze_file(path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = pd.read_csv(path, dtype={"story_id": str})
    recs["headline_normalized"] = recs["headline"].map(normalize_headline)
    return per_user_duplicate_metrics(recs, label), inter_user_overlap(recs, label)


def summarize(duplicates: pd.DataFrame, overlaps: pd.DataFrame) -> dict:
    summary: dict[str, dict] = {}
    for label, group in duplicates.groupby("list_type", sort=False):
        overlap_group = overlaps[overlaps["list_type"] == label]
        summary[label] = {
            "users": int(group["user_id"].nunique()),
            "mean_headline_uniqueness": float(group["headline_uniqueness"].mean()),
            "mean_headline_duplicate_rate": float(group["headline_duplicate_rate"].mean()),
            "mean_story_id_uniqueness": float(group["story_id_uniqueness"].mean()),
            "mean_story_id_duplicate_rate": float(group["story_id_duplicate_rate"].mean()),
            "mean_story_id_jaccard": float(overlap_group["story_id_jaccard"].mean()) if len(overlap_group) else 0.0,
            "mean_headline_jaccard": float(overlap_group["headline_jaccard"].mean()) if len(overlap_group) else 0.0,
            "max_story_id_jaccard": float(overlap_group["story_id_jaccard"].max()) if len(overlap_group) else 0.0,
            "max_headline_jaccard": float(overlap_group["headline_jaccard"].max()) if len(overlap_group) else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    recommendation_dir = resolve_path(repo_dir, args.recommendation_dir)
    output_dir = resolve_path(repo_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_duplicates, raw_overlaps = analyze_file(
        recommendation_dir / "telugu_top10_recommendations.csv",
        "raw_top10",
    )
    dedup_duplicates, dedup_overlaps = analyze_file(
        recommendation_dir / "telugu_top10_recommendations_headline_deduped.csv",
        "headline_deduped_top10",
    )

    duplicates = pd.concat([raw_duplicates, dedup_duplicates], ignore_index=True)
    overlaps = pd.concat([raw_overlaps, dedup_overlaps], ignore_index=True)
    summary = summarize(duplicates, overlaps)

    duplicates.to_csv(output_dir / "duplicate_rate_and_headline_uniqueness.csv", index=False)
    overlaps.to_csv(output_dir / "inter_user_overlap.csv", index=False)
    (output_dir / "recommendation_list_quality_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
