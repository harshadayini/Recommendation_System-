from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score as lightfm_auc_score
from lightfm.evaluation import precision_at_k, recall_at_k
from scipy.sparse import csr_matrix, load_npz

from run_mindlarge_train_metrics import (
    NEWS_COLUMNS,
    build_tfidf_features,
    iter_labeled_impressions,
    load_news,
    make_refined_categories,
    scan_behaviors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate already-trained MINDlarge train models on a separate labeled split."
    )
    parser.add_argument("--train-dir", default="MINDlarge_train")
    parser.add_argument("--eval-dir", default="MINDlarge_dev")
    parser.add_argument("--model-dir", default="results/mindlarge_train")
    parser.add_argument("--output", default="results/mindlarge_train/dev_metrics.json")
    parser.add_argument("--models", nargs="+", default=["vertical", "refined", "tfidf", "bert"])
    parser.add_argument("--eval-limit", type=int, default=100000, help="0 means all labeled impression groups")
    parser.add_argument(
        "--paper-metric-users",
        type=int,
        default=5000,
        help="Evaluate LightFM paper metrics on first N known eval users. Use 0 for all known eval users.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def fit_train_dataset(train_dir: Path, train_news: pd.DataFrame, model_name: str) -> Dataset:
    users, items, _ = scan_behaviors(train_dir / "behaviors.tsv")
    all_users = sorted(users)
    all_items = sorted(items | set(train_news["newid"]))

    if model_name == "vertical":
        feature_names = sorted(train_news["vertical"].dropna().unique())
    elif model_name == "refined":
        feature_names = sorted(make_refined_categories(train_news)["new_category"].dropna().unique())
    else:
        feature_names = []

    dataset = Dataset()
    dataset.fit(users=all_users, items=all_items, item_features=feature_names)
    return dataset


def load_or_create_train_dataset(
    train_dir: Path,
    train_news: pd.DataFrame,
    model_dir: Path,
    model_name: str,
) -> Dataset:
    dataset_path = model_dir / f"{model_name}_dataset.pkl"
    if dataset_path.exists():
        with dataset_path.open("rb") as f:
            return pickle.load(f)

    dataset = fit_train_dataset(train_dir, train_news, model_name)
    with dataset_path.open("wb") as f:
        pickle.dump(dataset, f)
    return dataset


def auc_score_group(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[labels == 1]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mrr_score(labels_sorted: np.ndarray) -> float:
    hits = np.flatnonzero(labels_sorted == 1)
    if len(hits) == 0:
        return 0.0
    return 1.0 / float(hits[0] + 1)


def ndcg_score(labels_sorted: np.ndarray, k: int) -> float:
    gains = labels_sorted[:k]
    if len(gains) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels_sorted)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_labeled_impressions(
    model: LightFM,
    dataset: Dataset,
    eval_behaviors_path: Path,
    item_features,
    eval_limit: int,
):
    user_map, _, item_map, _ = dataset.mapping()
    metrics = defaultdict(list)
    coverage = {
        "groups_seen": 0,
        "groups_evaluated": 0,
        "groups_skipped_unknown_user": 0,
        "groups_skipped_no_known_items": 0,
        "groups_skipped_single_class": 0,
        "candidate_items_seen": 0,
        "candidate_items_known": 0,
    }

    for user_id, item_ids, labels in iter_labeled_impressions(eval_behaviors_path):
        coverage["groups_seen"] += 1
        coverage["candidate_items_seen"] += len(item_ids)

        user_idx = user_map.get(user_id)
        if user_idx is None:
            coverage["groups_skipped_unknown_user"] += 1
            continue

        known = [(item_map[item_id], labels[i]) for i, item_id in enumerate(item_ids) if item_id in item_map]
        coverage["candidate_items_known"] += len(known)
        if len(known) < 2:
            coverage["groups_skipped_no_known_items"] += 1
            continue

        item_indices = np.asarray([x[0] for x in known], dtype=np.int32)
        group_labels = np.asarray([x[1] for x in known], dtype=np.int8)
        if group_labels.max() == group_labels.min():
            coverage["groups_skipped_single_class"] += 1
            continue

        scores = model.predict(
            np.repeat(user_idx, len(item_indices)),
            item_indices,
            item_features=item_features,
        )
        labels_sorted = group_labels[np.argsort(-scores)]
        metrics["AUC"].append(auc_score_group(group_labels, scores))
        metrics["MRR"].append(mrr_score(labels_sorted))
        metrics["nDCG@5"].append(ndcg_score(labels_sorted, 5))
        metrics["nDCG@10"].append(ndcg_score(labels_sorted, 10))

        coverage["groups_evaluated"] += 1
        if eval_limit and coverage["groups_evaluated"] >= eval_limit:
            break
        if coverage["groups_seen"] % 10000 == 0:
            print(
                f"  seen {coverage['groups_seen']:,}; "
                f"evaluated {coverage['groups_evaluated']:,}",
                flush=True,
            )

    result = {name: float(np.nanmean(values)) for name, values in metrics.items()}
    result.update(coverage)
    if coverage["candidate_items_seen"]:
        result["known_candidate_item_ratio"] = (
            coverage["candidate_items_known"] / coverage["candidate_items_seen"]
        )
    else:
        result["known_candidate_item_ratio"] = 0.0
    if coverage["groups_seen"]:
        result["evaluated_group_ratio"] = coverage["groups_evaluated"] / coverage["groups_seen"]
    else:
        result["evaluated_group_ratio"] = 0.0
    return result


def build_eval_positive_matrix(dataset: Dataset, eval_behaviors_path: Path, max_users: int):
    user_map, _, item_map, _ = dataset.mapping()
    allowed_users = set()
    rows = []
    cols = []
    data = []

    for user_id, item_ids, labels in iter_labeled_impressions(eval_behaviors_path):
        user_idx = user_map.get(user_id)
        if user_idx is None:
            continue
        if max_users and user_idx not in allowed_users:
            if len(allowed_users) >= max_users:
                continue
            allowed_users.add(user_idx)

        for item_id, label in zip(item_ids, labels):
            if label != 1:
                continue
            item_idx = item_map.get(item_id)
            if item_idx is None:
                continue
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(1.0)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_map), len(item_map)),
        dtype=np.float32,
    ), len(allowed_users) if max_users else len(set(rows))


def evaluate_paper_metrics(
    model: LightFM,
    dataset: Dataset,
    eval_behaviors_path: Path,
    item_features,
    max_users: int,
    num_threads: int,
):
    matrix, users_evaluated = build_eval_positive_matrix(dataset, eval_behaviors_path, max_users)
    if matrix.nnz == 0:
        return {
            "Precision@10": math.nan,
            "Recall@10": math.nan,
            "LightFM_AUC": math.nan,
            "paper_metric_users": users_evaluated,
            "paper_metric_positive_interactions": 0,
        }

    return {
        "Precision@10": float(
            precision_at_k(
                model,
                matrix,
                item_features=item_features,
                k=10,
                num_threads=num_threads,
            ).mean()
        ),
        "Recall@10": float(
            recall_at_k(
                model,
                matrix,
                item_features=item_features,
                k=10,
                num_threads=num_threads,
            ).mean()
        ),
        "LightFM_AUC": float(
            lightfm_auc_score(
                model,
                matrix,
                item_features=item_features,
                num_threads=num_threads,
            ).mean()
        ),
        "paper_metric_users": users_evaluated,
        "paper_metric_positive_interactions": int(matrix.nnz),
    }


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing eval directory: {eval_dir}")
    if not (eval_dir / "behaviors.tsv").exists():
        raise FileNotFoundError(f"Missing eval behaviors: {eval_dir / 'behaviors.tsv'}")

    train_news = load_news(train_dir)
    eval_news = pd.read_csv(
        eval_dir / "news.tsv",
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        dtype=str,
    )
    print(f"train_news={len(train_news):,} eval_news={len(eval_news):,}")

    results = {}
    if output_path.exists():
        with output_path.open() as f:
            results = json.load(f)

    for model_name in args.models:
        print(f"\n=== {model_name} dev evaluation ===", flush=True)
        model_path = model_dir / f"{model_name}_model.pkl"
        features_path = model_dir / f"{model_name}_item_features.npz"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Missing item features: {features_path}")

        dataset = load_or_create_train_dataset(train_dir, train_news, model_dir, model_name)
        item_features = load_npz(features_path)
        with model_path.open("rb") as f:
            model = pickle.load(f)

        paper_metrics = evaluate_paper_metrics(
            model,
            dataset,
            eval_dir / "behaviors.tsv",
            item_features,
            args.paper_metric_users,
            args.num_threads,
        )
        ranking_metrics = evaluate_labeled_impressions(
            model,
            dataset,
            eval_dir / "behaviors.tsv",
            item_features,
            args.eval_limit,
        )
        result = {
            **paper_metrics,
            **ranking_metrics,
            "eval_dir": str(eval_dir),
            "train_dir": str(train_dir),
        }
        results[model_name] = result
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)

    print("\nFinal dev results")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
