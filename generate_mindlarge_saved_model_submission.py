from __future__ import annotations

import argparse
import json
import math
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from run_mindlarge_train_metrics import load_news, make_refined_categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Codabench prediction.zip from an already-saved MINDlarge LightFM model."
    )
    parser.add_argument("--model-name", required=True, choices=["vertical", "refined", "tfidf", "bert"])
    parser.add_argument("--model-dir", default="results/mindlarge_train")
    parser.add_argument("--train-dir", default="MINDlarge_train")
    parser.add_argument("--ranking-dir", default="MINDlarge_test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-threads", type=int, default=4)
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


def count_train_popularity(train_behaviors_path: Path) -> Counter[str]:
    popularity: Counter[str] = Counter()
    for _, _, _, history, impressions in behavior_rows(train_behaviors_path):
        if history and history != "-":
            popularity.update(history.split())
        for candidate in impressions.split():
            if "-" not in candidate:
                continue
            item_id, label = candidate.rsplit("-", 1)
            if label == "1":
                popularity[item_id] += 1
    return popularity


def build_fallback_scores(train_dir: Path, ranking_dir: Path, popularity: Counter[str]) -> dict[str, float]:
    train_news = load_news(train_dir)
    ranking_news = load_news(ranking_dir)
    news = (
        __import__("pandas")
        .concat([train_news, ranking_news], ignore_index=True)
        .drop_duplicates("newid", keep="last")
    )
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
    model,
    dataset,
    item_features,
    ranking_dir: Path,
    prediction_path: Path,
    fallback_scores: dict[str, float],
    num_threads: int,
) -> dict:
    user_map, _, item_map, _ = dataset.mapping()
    stats: Counter[str] = Counter()

    with prediction_path.open("w", encoding="utf-8") as out:
        for impression_id, user_id, _, _, impressions in behavior_rows(ranking_dir / "behaviors.tsv"):
            candidates = [strip_label(candidate) for candidate in impressions.split()]
            stats["groups"] += 1
            stats["candidates"] += len(candidates)

            user_idx = user_map.get(user_id)
            if user_idx is None:
                stats["cold_user_groups"] += 1
                scores = np.asarray([fallback_scores.get(item_id, 0.0) for item_id in candidates], dtype=np.float64)
            else:
                scores = np.empty(len(candidates), dtype=np.float64)
                known_positions = []
                known_item_indices = []
                for position, item_id in enumerate(candidates):
                    item_idx = item_map.get(item_id)
                    if item_idx is None:
                        stats["unknown_candidate_items"] += 1
                        scores[position] = fallback_scores.get(item_id, 0.0)
                    else:
                        known_positions.append(position)
                        known_item_indices.append(item_idx)

                if known_item_indices:
                    predicted = model.predict(
                        np.repeat(user_idx, len(known_item_indices)),
                        np.asarray(known_item_indices, dtype=np.int32),
                        item_features=item_features,
                        num_threads=num_threads,
                    )
                    for position, score in zip(known_positions, predicted):
                        scores[position] = float(score)
                else:
                    stats["groups_with_no_known_candidate_items"] += 1

            ranks = ranks_from_scores(scores)
            out.write(f"{impression_id} {json.dumps(ranks, separators=(',', ':'))}\n")

            if stats["groups"] % 100000 == 0:
                print(
                    f"  wrote {stats['groups']:,} groups; "
                    f"cold_user_groups={stats['cold_user_groups']:,}; "
                    f"unknown_candidate_items={stats['unknown_candidate_items']:,}",
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
    model_dir = Path(args.model_dir)
    train_dir = Path(args.train_dir)
    ranking_dir = Path(args.ranking_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{args.model_name}_model.pkl"
    dataset_path = model_dir / f"{args.model_name}_dataset.pkl"
    item_features_path = model_dir / f"{args.model_name}_item_features.npz"
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / "prediction.zip"

    for path in [model_path, dataset_path, item_features_path, ranking_dir / "behaviors.tsv"]:
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"Loading saved {args.model_name} model artifacts", flush=True)
    with model_path.open("rb") as f:
        model = pickle.load(f)
    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)
    item_features = load_npz(item_features_path)

    popularity = count_train_popularity(train_dir / "behaviors.tsv")
    fallback_scores = build_fallback_scores(train_dir, ranking_dir, popularity)

    generation = generate_prediction_file(
        model,
        dataset,
        item_features,
        ranking_dir,
        prediction_path,
        fallback_scores,
        args.num_threads,
    )
    validation = validate_prediction_file(ranking_dir, prediction_path)
    write_zip(prediction_path, zip_path)

    result = {
        "model_name": args.model_name,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "item_features_path": str(item_features_path),
        "prediction_path": str(prediction_path),
        "zip_path": str(zip_path),
        "generation": generation,
        "validation": validation,
    }
    with (output_dir / "submission_metadata.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
