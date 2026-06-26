#!/usr/bin/env python3
"""Sampled Telugu recommendation-list quality analysis.

Unlike the representative qualitative examples, this script samples users at
random and computes duplicate/headline uniqueness and inter-user overlap over
their Telugu top-k recommendations.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


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
        help="Directory containing Telugu classification and feature outputs.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("submissions/refined_mindlarge_test_train_only"),
        help="Directory containing the saved tuned refined LightFM model and dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/telugu_sampled_recommendation_list_quality"),
        help="Output directory.",
    )
    parser.add_argument("--sample-users", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=107)
    return parser.parse_args()


def resolve_path(repo_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_dir / path


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_headline(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def build_full_telugu_item_features(
    telugu_features_23: sparse.csr_matrix,
    categories: list[str],
    item_feature_map: dict[str, int],
) -> sparse.csr_matrix:
    coo = telugu_features_23.tocoo()
    mapped_cols = np.array([item_feature_map[categories[int(col)]] for col in coo.col], dtype=np.int32)
    return sparse.csr_matrix(
        (coo.data.astype(np.float32), (coo.row.astype(np.int32), mapped_cols)),
        shape=(telugu_features_23.shape[0], len(item_feature_map)),
        dtype=np.float32,
    )


def sample_users(user_id_map: dict[str, int], sample_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = np.array(list(user_id_map.items()), dtype=object)
    actual_size = min(sample_size, len(users))
    selected_positions = rng.choice(np.arange(len(users)), size=actual_size, replace=False)
    sampled = users[selected_positions]
    return pd.DataFrame(
        {
            "user_id": sampled[:, 0].astype(str),
            "user_index": sampled[:, 1].astype(int),
        }
    ).sort_values("user_id").reset_index(drop=True)


def topk_recommendations(
    model,
    sampled_users: pd.DataFrame,
    telugu_features_full: sparse.csr_matrix,
    telugu_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    item_ids = np.arange(telugu_features_full.shape[0], dtype=np.int32)
    rows = []
    for position, user in enumerate(sampled_users.itertuples(index=False), start=1):
        scores = model.predict(
            int(user.user_index),
            item_ids,
            item_features=telugu_features_full,
            num_threads=1,
        )
        top_indices = np.argsort(-scores)[:top_k]
        for rank, item_idx in enumerate(top_indices, start=1):
            item_idx = int(item_idx)
            headline = str(telugu_df.iloc[item_idx]["headline"])
            rows.append(
                {
                    "user_id": user.user_id,
                    "user_index": int(user.user_index),
                    "rank": rank,
                    "story_id": str(telugu_df.iloc[item_idx]["story_id"]),
                    "headline": headline,
                    "headline_normalized": normalize_headline(headline),
                    "score": float(scores[item_idx]),
                }
            )
        if position % 50 == 0:
            print(f"scored {position:,}/{len(sampled_users):,} users", flush=True)
    return pd.DataFrame(rows)


def duplicate_metrics(recs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for user_id, group in recs.groupby("user_id", sort=False):
        top_k = len(group)
        unique_story_ids = group["story_id"].nunique()
        unique_headlines = group["headline_normalized"].nunique()
        rows.append(
            {
                "user_id": user_id,
                "top_k": top_k,
                "unique_story_ids": unique_story_ids,
                "story_id_uniqueness": unique_story_ids / top_k,
                "story_id_duplicate_rate": 1 - unique_story_ids / top_k,
                "unique_headlines": unique_headlines,
                "headline_uniqueness": unique_headlines / top_k,
                "headline_duplicate_rate": 1 - unique_headlines / top_k,
            }
        )
    return pd.DataFrame(rows)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def inter_user_overlap(recs: pd.DataFrame) -> pd.DataFrame:
    story_sets = {
        user_id: set(group["story_id"])
        for user_id, group in recs.groupby("user_id", sort=False)
    }
    headline_sets = {
        user_id: set(group["headline_normalized"])
        for user_id, group in recs.groupby("user_id", sort=False)
    }
    rows = []
    for left, right in combinations(story_sets.keys(), 2):
        rows.append(
            {
                "user_a": left,
                "user_b": right,
                "story_id_jaccard": jaccard(story_sets[left], story_sets[right]),
                "headline_jaccard": jaccard(headline_sets[left], headline_sets[right]),
                "shared_story_ids": len(story_sets[left] & story_sets[right]),
                "shared_headlines": len(headline_sets[left] & headline_sets[right]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    telugu_dir = resolve_path(repo_dir, args.telugu_dir)
    model_dir = resolve_path(repo_dir, args.model_dir)
    output_dir = resolve_path(repo_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_pickle(model_dir / "refined_submission_model.pkl")
    dataset = load_pickle(model_dir / "refined_submission_dataset.pkl")
    user_id_map, _, _, item_feature_map = dataset.mapping()

    label_mapping = json.loads((telugu_dir / "label_mapping.json").read_text())
    categories = label_mapping["categories"]
    telugu_features_23 = sparse.load_npz(telugu_dir / "telugu_item_features_23cat.npz").tocsr()
    telugu_features_full = build_full_telugu_item_features(telugu_features_23, categories, item_feature_map)
    telugu_df = pd.read_pickle(telugu_dir / "telugu_df_full.pkl")
    telugu_df["story_id"] = telugu_df["story_id"].astype(str)

    sampled_users = sample_users(user_id_map, args.sample_users, args.seed)
    recs = topk_recommendations(model, sampled_users, telugu_features_full, telugu_df, args.top_k)
    duplicates = duplicate_metrics(recs)
    overlaps = inter_user_overlap(recs)

    summary = {
        "sample_users": int(len(sampled_users)),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "recommendations": int(len(recs)),
        "mean_story_id_uniqueness": float(duplicates["story_id_uniqueness"].mean()),
        "mean_story_id_duplicate_rate": float(duplicates["story_id_duplicate_rate"].mean()),
        "mean_headline_uniqueness": float(duplicates["headline_uniqueness"].mean()),
        "mean_headline_duplicate_rate": float(duplicates["headline_duplicate_rate"].mean()),
        "median_headline_uniqueness": float(duplicates["headline_uniqueness"].median()),
        "mean_story_id_jaccard": float(overlaps["story_id_jaccard"].mean()) if len(overlaps) else 0.0,
        "mean_headline_jaccard": float(overlaps["headline_jaccard"].mean()) if len(overlaps) else 0.0,
        "max_story_id_jaccard": float(overlaps["story_id_jaccard"].max()) if len(overlaps) else 0.0,
        "max_headline_jaccard": float(overlaps["headline_jaccard"].max()) if len(overlaps) else 0.0,
        "mean_shared_story_ids": float(overlaps["shared_story_ids"].mean()) if len(overlaps) else 0.0,
        "mean_shared_headlines": float(overlaps["shared_headlines"].mean()) if len(overlaps) else 0.0,
    }

    sampled_users.to_csv(output_dir / "sampled_users.csv", index=False)
    recs.to_csv(output_dir / "sampled_telugu_top10_recommendations.csv", index=False)
    duplicates.to_csv(output_dir / "sampled_duplicate_rate_and_headline_uniqueness.csv", index=False)
    overlaps.to_csv(output_dir / "sampled_inter_user_overlap.csv", index=False)
    (output_dir / "sampled_recommendation_list_quality_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
