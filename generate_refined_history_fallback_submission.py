from __future__ import annotations

import argparse
import json
import math
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from run_mindlarge_train_metrics import load_news, make_refined_categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a refined LightFM Codabench submission with history-based fallback "
            "for cold users and cold candidate news."
        )
    )
    parser.add_argument("--train-dir", default="MINDlarge_train")
    parser.add_argument("--ranking-dir", default="MINDlarge_test")
    parser.add_argument("--model-path", default="submissions/refined_mindlarge_test_train_only/refined_submission_model.pkl")
    parser.add_argument(
        "--dataset-path",
        default="submissions/refined_mindlarge_test_train_only/refined_submission_dataset.pkl",
    )
    parser.add_argument(
        "--item-features-path",
        default="submissions/refined_mindlarge_test_train_only/refined_submission_item_features.pkl",
    )
    parser.add_argument("--output-dir", default="submissions/refined_history_fallback_test")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--model-weight",
        type=float,
        default=0.75,
        help="Blend weight for normalized LightFM scores when known train user/item scores exist.",
    )
    parser.add_argument(
        "--fallback-weight",
        type=float,
        default=0.25,
        help="Blend weight for normalized fallback scores when LightFM scores exist.",
    )
    return parser.parse_args()


def behavior_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 5:
                yield parts


def strip_label(candidate: str) -> str:
    if "-" not in candidate:
        return candidate
    item_id, suffix = candidate.rsplit("-", 1)
    return item_id if suffix in {"0", "1"} else candidate


def scan_train_users_items(train_dir: Path) -> tuple[set[str], set[str]]:
    users: set[str] = set()
    items: set[str] = set(load_news(train_dir)["newid"])
    for _, user_id, _, history, impressions in behavior_rows(train_dir / "behaviors.tsv"):
        users.add(user_id)
        if history and history != "-":
            items.update(history.split())
        for candidate in impressions.split():
            if "-" not in candidate:
                continue
            item_id, _ = candidate.rsplit("-", 1)
            items.add(item_id)
    return users, items


def load_combined_news(train_dir: Path, ranking_dir: Path) -> pd.DataFrame:
    frames = [load_news(train_dir)]
    ranking_news_path = ranking_dir / "news.tsv"
    if ranking_news_path.exists():
        frames.append(load_news(ranking_dir))
    return pd.concat(frames, ignore_index=True).drop_duplicates("newid", keep="last")


def train_popularity(train_dir: Path) -> tuple[Counter[str], Counter[str]]:
    item_popularity: Counter[str] = Counter()
    for _, _, _, history, impressions in behavior_rows(train_dir / "behaviors.tsv"):
        if history and history != "-":
            item_popularity.update(history.split())
        for candidate in impressions.split():
            if "-" not in candidate:
                continue
            item_id, label = candidate.rsplit("-", 1)
            if label == "1":
                item_popularity[item_id] += 1
    return item_popularity, Counter()


def build_news_metadata(news: pd.DataFrame, item_popularity: Counter[str]):
    refined_news = make_refined_categories(news)
    item_to_category = refined_news.set_index("newid")["new_category"].fillna("").to_dict()
    category_popularity: Counter[str] = Counter()
    for item_id, count in item_popularity.items():
        category = item_to_category.get(item_id, "")
        if category:
            category_popularity[category] += count
    return item_to_category, category_popularity


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value == min_value:
        return np.zeros_like(values, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)


def ranks_from_scores(scores: np.ndarray) -> list[int]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def history_category_counts(history: str, item_to_category: dict[str, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not history or history == "-":
        return counts
    # Recent clicks are often more informative, so cap very long histories to the last 100 clicks.
    for item_id in history.split()[-100:]:
        category = item_to_category.get(item_id, "")
        if category:
            counts[category] += 1
    return counts


def fallback_scores_for_candidates(
    candidates: list[str],
    history_counts: Counter[str],
    item_to_category: dict[str, str],
    item_popularity: Counter[str],
    category_popularity: Counter[str],
) -> np.ndarray:
    scores = np.zeros(len(candidates), dtype=np.float64)
    total_history = sum(history_counts.values())
    for idx, item_id in enumerate(candidates):
        category = item_to_category.get(item_id, "")
        history_score = 0.0
        if total_history and category:
            history_score = history_counts.get(category, 0) / total_history

        item_score = math.log1p(item_popularity.get(item_id, 0))
        category_score = math.log1p(category_popularity.get(category, 0)) if category else 0.0
        scores[idx] = (2.0 * history_score) + (0.05 * item_score) + (0.005 * category_score)
    return scores


def generate_prediction_file(
    model,
    dataset,
    item_features,
    train_users: set[str],
    train_items: set[str],
    ranking_dir: Path,
    prediction_path: Path,
    item_to_category: dict[str, str],
    item_popularity: Counter[str],
    category_popularity: Counter[str],
    model_weight: float,
    fallback_weight: float,
    num_threads: int,
) -> dict:
    user_map, _, item_map, _ = dataset.mapping()
    stats: Counter[str] = Counter()

    with prediction_path.open("w", encoding="utf-8") as out:
        for impression_id, user_id, _, history, impressions in behavior_rows(ranking_dir / "behaviors.tsv"):
            candidates = [strip_label(candidate) for candidate in impressions.split()]
            stats["groups"] += 1
            stats["candidates"] += len(candidates)

            history_counts = history_category_counts(history, item_to_category)
            fallback_raw = fallback_scores_for_candidates(
                candidates,
                history_counts,
                item_to_category,
                item_popularity,
                category_popularity,
            )
            fallback_norm = normalize(fallback_raw)

            user_idx = user_map.get(user_id)
            known_user = user_id in train_users and user_idx is not None
            known_positions = []
            known_item_indices = []

            if known_user:
                for pos, item_id in enumerate(candidates):
                    item_idx = item_map.get(item_id)
                    if item_id in train_items and item_idx is not None:
                        known_positions.append(pos)
                        known_item_indices.append(item_idx)
                    else:
                        stats["cold_candidate_items"] += 1
            else:
                stats["cold_user_groups"] += 1

            if known_user and known_item_indices:
                model_raw = model.predict(
                    np.repeat(user_idx, len(known_item_indices)),
                    np.asarray(known_item_indices, dtype=np.int32),
                    item_features=item_features,
                    num_threads=num_threads,
                )
                model_norm = normalize(np.asarray(model_raw, dtype=np.float64))
                scores = fallback_norm.copy()
                for local_idx, pos in enumerate(known_positions):
                    scores[pos] = (model_weight * model_norm[local_idx]) + (
                        fallback_weight * fallback_norm[pos]
                    )
            else:
                if known_user:
                    stats["known_user_groups_with_no_train_candidates"] += 1
                scores = fallback_norm

            ranks = ranks_from_scores(scores)
            out.write(f"{impression_id} {json.dumps(ranks, separators=(',', ':'))}\n")

            if stats["groups"] % 100000 == 0:
                print(
                    f"  wrote {stats['groups']:,} groups; "
                    f"cold_user_groups={stats['cold_user_groups']:,}; "
                    f"cold_candidate_items={stats['cold_candidate_items']:,}",
                    flush=True,
                )

    return dict(stats)


def validate_prediction_file(ranking_dir: Path, prediction_path: Path) -> dict:
    stats: Counter[str] = Counter()
    with (ranking_dir / "behaviors.tsv").open("r", encoding="utf-8") as behaviors, prediction_path.open(
        "r", encoding="utf-8"
    ) as predictions:
        for line_no, behavior_line in enumerate(behaviors, start=1):
            parts = behavior_line.rstrip("\n").split("\t")
            prediction_line = predictions.readline()
            if not prediction_line:
                raise ValueError(f"Missing prediction line {line_no}")
            pred_id, raw_ranks = prediction_line.rstrip("\n").split(" ", 1)
            if pred_id != parts[0]:
                raise ValueError(f"Line {line_no}: expected impression {parts[0]}, got {pred_id}")
            candidate_count = len(parts[4].split())
            ranks = json.loads(raw_ranks)
            if sorted(ranks) != list(range(1, candidate_count + 1)):
                raise ValueError(f"Line {line_no}: ranks are not a permutation of 1..{candidate_count}")
            stats["groups"] += 1
            stats["candidates"] += candidate_count
        if predictions.readline():
            raise ValueError("Prediction file has extra lines after behaviors.tsv ended.")
    return dict(stats)


def write_zip(prediction_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, arcname="prediction.txt")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    if names != ["prediction.txt"]:
        raise ValueError(f"Invalid zip contents: {names}")


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    ranking_dir = Path(args.ranking_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / "prediction.zip"

    for path in [
        Path(args.model_path),
        Path(args.dataset_path),
        Path(args.item_features_path),
        train_dir / "behaviors.tsv",
        ranking_dir / "behaviors.tsv",
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading tuned refined LightFM model artifacts", flush=True)
    with Path(args.model_path).open("rb") as f:
        model = pickle.load(f)
    with Path(args.dataset_path).open("rb") as f:
        dataset = pickle.load(f)
    with Path(args.item_features_path).open("rb") as f:
        item_features = pickle.load(f)

    print("Building train coverage and fallback metadata", flush=True)
    train_users, train_items = scan_train_users_items(train_dir)
    item_popularity, _ = train_popularity(train_dir)
    news = load_combined_news(train_dir, ranking_dir)
    item_to_category, category_popularity = build_news_metadata(news, item_popularity)
    print(
        f"train_users={len(train_users):,} train_items={len(train_items):,} "
        f"news={len(news):,} categories={len(category_popularity):,}",
        flush=True,
    )

    generation = generate_prediction_file(
        model,
        dataset,
        item_features,
        train_users,
        train_items,
        ranking_dir,
        prediction_path,
        item_to_category,
        item_popularity,
        category_popularity,
        args.model_weight,
        args.fallback_weight,
        args.num_threads,
    )
    validation = validate_prediction_file(ranking_dir, prediction_path)
    write_zip(prediction_path, zip_path)

    result = {
        "strategy": "refined_lightfm_history_category_fallback",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "item_features_path": args.item_features_path,
        "prediction_path": str(prediction_path),
        "zip_path": str(zip_path),
        "model_weight": args.model_weight,
        "fallback_weight": args.fallback_weight,
        "generation": generation,
        "validation": validation,
    }
    with (output_dir / "submission_metadata.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
