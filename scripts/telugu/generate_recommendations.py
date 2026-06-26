#!/usr/bin/env python3
"""Generate qualitative Telugu recommendations for the MINDlarge refined model.

This reproduces the paper's Telugu recommendation stage after the MINDlarge
switch: Telugu articles are represented by XLM-R category probabilities in the
same refined category space used by the English-trained LightFM model.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
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
        help="Directory containing Colab Telugu outputs.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("submissions/refined_mindlarge_test_train_only"),
        help="Directory containing the saved tuned refined LightFM model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/telugu_mindlarge_recommendations"),
        help="Output directory for generated tables.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--target-categories",
        nargs="+",
        default=["sports", "music", "newspolitics"],
        help="Representative user categories to mirror the paper's qualitative study.",
    )
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def build_full_telugu_item_features(
    telugu_features_23: sparse.csr_matrix,
    categories: list[str],
    item_feature_map: dict[str, int],
) -> sparse.csr_matrix:
    rows = []
    cols = []
    data = []
    coo = telugu_features_23.tocoo()
    for row, col, value in zip(coo.row, coo.col, coo.data):
        category = categories[int(col)]
        mapped_col = item_feature_map.get(category)
        if mapped_col is None:
            continue
        rows.append(int(row))
        cols.append(mapped_col)
        data.append(float(value))

    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(telugu_features_23.shape[0], len(item_feature_map)),
        dtype=np.float32,
    )


def category_score_matrix(model, user_indices: np.ndarray, category_feature_matrix: sparse.csr_matrix) -> np.ndarray:
    scores = np.empty((len(user_indices), category_feature_matrix.shape[0]), dtype=np.float32)
    item_ids = np.arange(category_feature_matrix.shape[0], dtype=np.int32)
    for out_row, user_idx in enumerate(user_indices):
        scores[out_row] = model.predict(
            int(user_idx),
            item_ids,
            item_features=category_feature_matrix,
            num_threads=1,
        )
    return scores


def choose_representative_users(
    model,
    user_id_map: dict[str, int],
    categories: list[str],
    item_feature_map: dict[str, int],
    target_categories: list[str],
    sample_size: int = 50000,
) -> pd.DataFrame:
    rng = np.random.default_rng(107)
    all_user_indices = np.array(list(user_id_map.values()), dtype=np.int32)
    if len(all_user_indices) > sample_size:
        sampled = np.sort(rng.choice(all_user_indices, size=sample_size, replace=False))
    else:
        sampled = all_user_indices

    category_rows = np.arange(len(categories), dtype=np.int32)
    category_cols = [item_feature_map[category] for category in categories]
    category_features = sparse.csr_matrix(
        (np.ones(len(categories), dtype=np.float32), (category_rows, category_cols)),
        shape=(len(categories), len(item_feature_map)),
    )
    scores = category_score_matrix(model, sampled, category_features)
    reverse_user_map = {idx: user_id for user_id, idx in user_id_map.items()}

    rows = []
    used_users: set[int] = set()
    for target in target_categories:
        target_col = categories.index(target)
        order = np.argsort(-scores[:, target_col])
        chosen_sample_pos = None
        for pos in order:
            user_idx = int(sampled[pos])
            if user_idx not in used_users:
                chosen_sample_pos = int(pos)
                used_users.add(user_idx)
                break
        if chosen_sample_pos is None:
            continue

        user_idx = int(sampled[chosen_sample_pos])
        user_scores = scores[chosen_sample_pos]
        top_cols = np.argsort(-user_scores)[:3]
        rows.append(
            {
                "target_category": target,
                "user_id": reverse_user_map[user_idx],
                "user_index": user_idx,
                "top_category_1": categories[int(top_cols[0])],
                "top_score_1": float(user_scores[int(top_cols[0])]),
                "top_category_2": categories[int(top_cols[1])],
                "top_score_2": float(user_scores[int(top_cols[1])]),
                "top_category_3": categories[int(top_cols[2])],
                "top_score_3": float(user_scores[int(top_cols[2])]),
            }
        )
    return pd.DataFrame(rows)


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def normalize_headline(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def generate_recommendations(
    model,
    representative_users: pd.DataFrame,
    telugu_features_full: sparse.csr_matrix,
    telugu_df: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_ids = np.arange(telugu_features_full.shape[0], dtype=np.int32)
    pred_lookup = predictions.set_index("story_id")
    rec_rows = []
    dedup_rows = []
    user_recs: dict[str, set[str]] = {}
    user_recs_dedup: dict[str, set[str]] = {}

    for user in representative_users.itertuples(index=False):
        scores = model.predict(
            int(user.user_index),
            item_ids,
            item_features=telugu_features_full,
            num_threads=1,
        )
        top_indices = np.argsort(-scores)[:top_k]
        story_ids = []
        for rank, item_idx in enumerate(top_indices, start=1):
            story_id = str(telugu_df.iloc[int(item_idx)]["story_id"])
            story_ids.append(story_id)
            pred_row = pred_lookup.loc[story_id] if story_id in pred_lookup.index else None
            rec_rows.append(
                {
                    "user_id": user.user_id,
                    "user_index": int(user.user_index),
                    "rank": rank,
                    "story_id": story_id,
                    "headline": telugu_df.iloc[int(item_idx)]["headline"],
                    "source_file": telugu_df.iloc[int(item_idx)]["source_file"],
                    "predicted_category": pred_row["predicted_category"] if pred_row is not None else "",
                    "category_confidence": float(pred_row["confidence"]) if pred_row is not None else np.nan,
                    "score": float(scores[int(item_idx)]),
                }
            )
        user_recs[str(user.user_id)] = set(story_ids)

        seen_headlines: set[str] = set()
        dedup_rank = 1
        dedup_story_ids = []
        for item_idx in np.argsort(-scores):
            headline = telugu_df.iloc[int(item_idx)]["headline"]
            normalized = normalize_headline(headline)
            if normalized in seen_headlines:
                continue
            seen_headlines.add(normalized)
            story_id = str(telugu_df.iloc[int(item_idx)]["story_id"])
            dedup_story_ids.append(story_id)
            pred_row = pred_lookup.loc[story_id] if story_id in pred_lookup.index else None
            dedup_rows.append(
                {
                    "user_id": user.user_id,
                    "user_index": int(user.user_index),
                    "rank": dedup_rank,
                    "story_id": story_id,
                    "headline": headline,
                    "source_file": telugu_df.iloc[int(item_idx)]["source_file"],
                    "predicted_category": pred_row["predicted_category"] if pred_row is not None else "",
                    "category_confidence": float(pred_row["confidence"]) if pred_row is not None else np.nan,
                    "score": float(scores[int(item_idx)]),
                    "postprocess": "headline_dedup",
                }
            )
            dedup_rank += 1
            if dedup_rank > top_k:
                break
        user_recs_dedup[str(user.user_id)] = set(dedup_story_ids)

    recs = pd.DataFrame(rec_rows)
    dedup_recs = pd.DataFrame(dedup_rows)

    def build_diversity(frame: pd.DataFrame) -> pd.DataFrame:
        diversity_rows = []
        for user_id, group in frame.groupby("user_id", sort=False):
            diversity_rows.append(
                {
                    "user_id": user_id,
                    "top_k": int(len(group)),
                    "headline_uniqueness": float(group["headline"].nunique() / len(group)),
                    "category_entropy_bits": entropy(group["predicted_category"].astype(str).tolist()),
                    "unique_categories": int(group["predicted_category"].nunique()),
                }
            )
        return pd.DataFrame(diversity_rows)

    diversity = build_diversity(recs)
    dedup_diversity = build_diversity(dedup_recs)

    def build_jaccard(user_recs_by_id: dict[str, set[str]]) -> pd.DataFrame:
        user_ids = list(user_recs_by_id)
        matrix = pd.DataFrame(index=user_ids, columns=user_ids, dtype=float)
        for left in user_ids:
            for right in user_ids:
                union = user_recs_by_id[left] | user_recs_by_id[right]
                matrix.loc[left, right] = (
                    len(user_recs_by_id[left] & user_recs_by_id[right]) / len(union) if union else 0.0
                )
        return matrix

    jaccard = build_jaccard(user_recs)
    dedup_jaccard = build_jaccard(user_recs_dedup)
    return recs, diversity, jaccard, dedup_recs, dedup_diversity, dedup_jaccard


def save_jaccard_heatmap(jaccard: pd.DataFrame, path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(4, 3.5))
    image = ax.imshow(jaccard.astype(float).to_numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(jaccard.columns)))
    ax.set_xticklabels(jaccard.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(jaccard.index)))
    ax.set_yticklabels(jaccard.index)
    ax.set_title(title)
    for i in range(jaccard.shape[0]):
        for j in range(jaccard.shape[1]):
            ax.text(j, i, f"{jaccard.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def mean_off_diagonal(jaccard: pd.DataFrame) -> float:
    if jaccard.shape[0] <= 1:
        return math.nan
    return float(
        np.mean(
            [
                jaccard.iloc[i, j]
                for i in range(jaccard.shape[0])
                for j in range(jaccard.shape[1])
                if i != j
            ]
        )
    )


def mean_dict(frame: pd.DataFrame) -> dict[str, float]:
    return frame.mean(numeric_only=True).replace({np.nan: None}).to_dict()


def main() -> None:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    telugu_dir = (repo_dir / args.telugu_dir).resolve()
    model_dir = (repo_dir / args.model_dir).resolve()
    output_dir = (repo_dir / args.output_dir).resolve()
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
    predictions = pd.read_csv(telugu_dir / "telugu_category_predictions.csv", dtype={"story_id": str})

    representative_users = choose_representative_users(
        model,
        user_id_map,
        categories,
        item_feature_map,
        args.target_categories,
    )
    recs, diversity, jaccard, dedup_recs, dedup_diversity, dedup_jaccard = generate_recommendations(
        model,
        representative_users,
        telugu_features_full,
        telugu_df,
        predictions,
        args.top_k,
    )

    representative_users.to_csv(output_dir / "representative_user_category_affinities.csv", index=False)
    recs.to_csv(output_dir / "telugu_top10_recommendations.csv", index=False)
    diversity.to_csv(output_dir / "telugu_recommendation_diversity.csv", index=False)
    jaccard.to_csv(output_dir / "telugu_recommendation_jaccard.csv")
    dedup_recs.to_csv(output_dir / "telugu_top10_recommendations_headline_deduped.csv", index=False)
    dedup_diversity.to_csv(output_dir / "telugu_recommendation_diversity_headline_deduped.csv", index=False)
    dedup_jaccard.to_csv(output_dir / "telugu_recommendation_jaccard_headline_deduped.csv")
    save_jaccard_heatmap(jaccard, output_dir / "telugu_recommendation_jaccard.png", "Raw top-10 overlap")
    save_jaccard_heatmap(
        dedup_jaccard,
        output_dir / "telugu_recommendation_jaccard_headline_deduped.png",
        "Headline-deduped top-10 overlap",
    )

    summary = {
        "telugu_rows": int(telugu_features_23.shape[0]),
        "category_count": int(len(categories)),
        "full_item_features_shape": list(telugu_features_full.shape),
        "full_item_features_nnz": int(telugu_features_full.nnz),
        "top_k": int(args.top_k),
        "representative_users": representative_users.to_dict(orient="records"),
        "raw_diversity_mean": mean_dict(diversity),
        "raw_mean_pairwise_jaccard_excluding_diagonal": mean_off_diagonal(jaccard),
        "headline_deduped_diversity_mean": mean_dict(dedup_diversity),
        "headline_deduped_mean_pairwise_jaccard_excluding_diagonal": mean_off_diagonal(dedup_jaccard),
    }
    (output_dir / "telugu_recommendation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
