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
from lightfm import LightFM
from lightfm.data import Dataset

from run_mindlarge_train_metrics import (
    NEWS_COLUMNS,
    build_category_features,
    iter_interactions,
    load_news,
    make_refined_categories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the selected refined MINDlarge LightFM model and generate Codabench predictions."
    )
    parser.add_argument(
        "--train-dirs",
        nargs="+",
        default=["MINDlarge_train"],
        help="One or more labeled MIND directories used for final training.",
    )
    parser.add_argument("--ranking-dir", default="MINDlarge_test")
    parser.add_argument("--output-dir", default="submissions/refined_mindlarge")
    parser.add_argument("--prediction-name", default="prediction.txt")
    parser.add_argument("--zip-name", default="prediction.zip")
    parser.add_argument("--model-name", default="refined_submission_model.pkl")
    parser.add_argument("--dataset-name", default="refined_submission_dataset.pkl")
    parser.add_argument("--item-features-name", default="refined_submission_item_features.pkl")
    parser.add_argument("--loss", default="bpr")
    parser.add_argument("--components", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--item-alpha", type=float, default=0.0)
    parser.add_argument("--user-alpha", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--load-existing-model", action="store_true")
    parser.add_argument("--skip-zip", action="store_true")
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


def scan_behavior_ids(behavior_paths: list[Path]) -> tuple[set[str], set[str]]:
    users: set[str] = set()
    items: set[str] = set()
    for path in behavior_paths:
        for _, user_id, _, history, impressions in behavior_rows(path):
            users.add(user_id)
            if history and history != "-":
                items.update(history.split())
            for candidate in impressions.split():
                items.add(strip_label(candidate))
    return users, items


def count_training_popularity(behavior_paths: list[Path]) -> tuple[Counter[str], set[str]]:
    popularity: Counter[str] = Counter()
    users_with_interactions: set[str] = set()
    for path in behavior_paths:
        for _, user_id, _, history, impressions in behavior_rows(path):
            has_interaction = False
            if history and history != "-":
                clicked = history.split()
                popularity.update(clicked)
                has_interaction = has_interaction or bool(clicked)

            for candidate in impressions.split():
                if "-" not in candidate:
                    continue
                item_id, label = candidate.rsplit("-", 1)
                if label == "1":
                    popularity[item_id] += 1
                    has_interaction = True

            if has_interaction:
                users_with_interactions.add(user_id)
    return popularity, users_with_interactions


def load_combined_news(data_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for data_dir in data_dirs:
        news_path = data_dir / "news.tsv"
        if news_path.exists():
            frames.append(load_news(data_dir))
    if not frames:
        raise FileNotFoundError("No news.tsv files found for the provided directories.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("newid", keep="last")


def build_interactions_from_dirs(dataset: Dataset, train_dirs: list[Path]):
    def interactions():
        for train_dir in train_dirs:
            yield from iter_interactions(train_dir / "behaviors.tsv")

    matrix, _ = dataset.build_interactions(interactions())
    return matrix


def build_fallback_scores(news: pd.DataFrame, popularity: Counter[str]) -> dict[str, float]:
    refined_news = make_refined_categories(news)
    item_to_category = refined_news.set_index("newid")["new_category"].fillna("").to_dict()
    category_popularity: Counter[str] = Counter()
    for item_id, count in popularity.items():
        category = item_to_category.get(item_id, "")
        if category:
            category_popularity[category] += count

    fallback = {}
    for item_id in news["newid"]:
        item_score = math.log1p(popularity.get(item_id, 0))
        category_score = math.log1p(category_popularity.get(item_to_category.get(item_id, ""), 0))
        fallback[item_id] = item_score + 0.01 * category_score
    return fallback


def ranks_from_scores(scores: np.ndarray) -> list[int]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def generate_prediction_file(
    model: LightFM,
    dataset: Dataset,
    item_features,
    ranking_dir: Path,
    output_path: Path,
    users_with_interactions: set[str],
    fallback_scores: dict[str, float],
    num_threads: int,
) -> dict:
    user_map, _, item_map, _ = dataset.mapping()
    stats = Counter()

    with output_path.open("w", encoding="utf-8") as out:
        for impression_id, user_id, _, _, impressions in behavior_rows(ranking_dir / "behaviors.tsv"):
            candidates = [strip_label(candidate) for candidate in impressions.split()]
            stats["groups"] += 1
            stats["candidates"] += len(candidates)

            use_model = user_id in users_with_interactions and user_id in user_map
            if use_model:
                user_idx = user_map[user_id]
                scores = np.empty(len(candidates), dtype=np.float64)
                known_positions = []
                known_item_indices = []
                for pos, item_id in enumerate(candidates):
                    item_idx = item_map.get(item_id)
                    if item_idx is None:
                        scores[pos] = fallback_scores.get(item_id, 0.0)
                        stats["unknown_candidate_items"] += 1
                    else:
                        known_positions.append(pos)
                        known_item_indices.append(item_idx)

                if known_item_indices:
                    predicted = model.predict(
                        np.repeat(user_idx, len(known_item_indices)),
                        np.asarray(known_item_indices, dtype=np.int32),
                        item_features=item_features,
                        num_threads=num_threads,
                    )
                    for pos, score in zip(known_positions, predicted):
                        scores[pos] = float(score)
                else:
                    stats["groups_with_no_known_candidate_items"] += 1
            else:
                stats["cold_user_groups"] += 1
                scores = np.asarray([fallback_scores.get(item_id, 0.0) for item_id in candidates], dtype=np.float64)

            ranks = ranks_from_scores(scores)
            out.write(f"{impression_id} {json.dumps(ranks, separators=(',', ':'))}\n")

            if stats["groups"] % 100000 == 0:
                print(
                    f"  wrote {stats['groups']:,} groups; "
                    f"cold_user_groups={stats['cold_user_groups']:,}",
                    flush=True,
                )

    return dict(stats)


def validate_prediction_file(ranking_dir: Path, prediction_path: Path) -> dict:
    stats = Counter()
    with (ranking_dir / "behaviors.tsv").open("r", encoding="utf-8") as behaviors, prediction_path.open(
        "r", encoding="utf-8"
    ) as predictions:
        for line_no, behavior_line in enumerate(behaviors, start=1):
            behavior_parts = behavior_line.rstrip("\n").split("\t")
            prediction_line = predictions.readline()
            if not prediction_line:
                raise ValueError(f"Missing prediction line {line_no}")
            impression_id = behavior_parts[0]
            candidates = behavior_parts[4].split()
            pred_id, raw_ranks = prediction_line.rstrip("\n").split(" ", 1)
            if pred_id != impression_id:
                raise ValueError(f"Line {line_no}: expected impression {impression_id}, got {pred_id}")
            ranks = json.loads(raw_ranks)
            expected = list(range(1, len(candidates) + 1))
            if sorted(ranks) != expected:
                raise ValueError(f"Line {line_no}: ranks are not a permutation of 1..{len(candidates)}")
            stats["groups"] += 1
            stats["candidates"] += len(candidates)

        extra = predictions.readline()
        if extra:
            raise ValueError("Prediction file has extra lines after behaviors.tsv ended.")
    return dict(stats)


def write_zip(prediction_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, arcname="prediction.txt")


def main() -> None:
    args = parse_args()
    train_dirs = [Path(value) for value in args.train_dirs]
    ranking_dir = Path(args.ranking_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = output_dir / args.prediction_name
    zip_path = output_dir / args.zip_name
    model_path = output_dir / args.model_name
    dataset_path = output_dir / args.dataset_name
    item_features_path = output_dir / args.item_features_name

    if not (ranking_dir / "behaviors.tsv").exists():
        raise FileNotFoundError(f"Missing ranking behaviors file: {ranking_dir / 'behaviors.tsv'}")

    behavior_paths = [train_dir / "behaviors.tsv" for train_dir in train_dirs]
    for path in behavior_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing training behaviors file: {path}")

    all_news = load_combined_news([*train_dirs, ranking_dir])
    refined_news = make_refined_categories(all_news)
    train_users, train_items = scan_behavior_ids(behavior_paths)
    ranking_users, ranking_items = scan_behavior_ids([ranking_dir / "behaviors.tsv"])
    all_users = sorted(train_users | ranking_users)
    all_items = sorted(train_items | ranking_items | set(all_news["newid"]))
    feature_names = sorted(refined_news["new_category"].dropna().unique())
    popularity, users_with_interactions = count_training_popularity(behavior_paths)

    print(
        f"users={len(all_users):,} items={len(all_items):,} "
        f"features={len(feature_names):,} train_dirs={','.join(str(path) for path in train_dirs)}",
        flush=True,
    )

    if args.load_existing_model:
        with dataset_path.open("rb") as f:
            dataset = pickle.load(f)
        with item_features_path.open("rb") as f:
            item_features = pickle.load(f)
        with model_path.open("rb") as f:
            model = pickle.load(f)
    else:
        dataset = Dataset()
        dataset.fit(users=all_users, items=all_items, item_features=feature_names)
        interactions = build_interactions_from_dirs(dataset, train_dirs)
        item_features = build_category_features(dataset, refined_news, "new_category")
        print(
            f"interaction_matrix={interactions.shape} nnz={interactions.nnz:,} "
            f"item_features={item_features.shape}",
            flush=True,
        )

        model = LightFM(
            no_components=args.components,
            loss=args.loss,
            learning_rate=args.learning_rate,
            item_alpha=args.item_alpha,
            user_alpha=args.user_alpha,
        )
        model.fit(
            interactions,
            item_features=item_features,
            epochs=args.epochs,
            num_threads=args.num_threads,
            verbose=True,
        )

        with model_path.open("wb") as f:
            pickle.dump(model, f)
        with dataset_path.open("wb") as f:
            pickle.dump(dataset, f)
        with item_features_path.open("wb") as f:
            pickle.dump(item_features, f)

    fallback_scores = build_fallback_scores(all_news, popularity)
    stats = generate_prediction_file(
        model,
        dataset,
        item_features,
        ranking_dir,
        prediction_path,
        users_with_interactions,
        fallback_scores,
        args.num_threads,
    )
    validation = validate_prediction_file(ranking_dir, prediction_path)
    print(json.dumps({"generation": stats, "validation": validation}, indent=2), flush=True)

    if not args.skip_zip:
        write_zip(prediction_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        if names != ["prediction.txt"]:
            raise ValueError(f"Invalid zip contents: {names}")
        print(f"Wrote {zip_path}", flush=True)


if __name__ == "__main__":
    main()
