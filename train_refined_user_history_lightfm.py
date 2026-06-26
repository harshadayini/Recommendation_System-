from __future__ import annotations

import argparse
import json
import math
import pickle
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix

from run_mindlarge_train_metrics import (
    NEWS_COLUMNS,
    auc_score_group,
    iter_interactions,
    load_news,
    make_refined_categories,
    mrr_score,
    ndcg_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train refined LightFM with explicit user-history features for cold-user support."
    )
    parser.add_argument("--train-dir", default="MINDlarge_train")
    parser.add_argument("--eval-dir", default="MINDlarge_dev")
    parser.add_argument("--test-dir", default="MINDlarge_test")
    parser.add_argument("--output-dir", default="results/refined_user_history")
    parser.add_argument("--mode", choices=["eval", "submit"], default="eval")
    parser.add_argument("--loss", default="bpr")
    parser.add_argument("--components", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--item-alpha", type=float, default=0.0)
    parser.add_argument("--user-alpha", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--eval-limit", type=int, default=50000, help="0 means evaluate all valid dev groups")
    parser.add_argument("--load-existing-model", action="store_true")
    parser.add_argument("--global-user-weight", type=float, default=0.01)
    parser.add_argument("--known-user-id-weight", type=float, default=1.0)
    parser.add_argument("--train-history-weight", type=float, default=0.05)
    parser.add_argument("--predict-known-history-weight", type=float, default=0.05)
    parser.add_argument("--predict-unknown-history-weight", type=float, default=1.0)
    parser.add_argument("--known-item-id-weight", type=float, default=1.0)
    parser.add_argument("--item-category-weight", type=float, default=0.05)
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


def load_combined_news(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for data_dir in paths:
        news_path = data_dir / "news.tsv"
        if news_path.exists():
            frames.append(
                pd.read_csv(
                    news_path,
                    sep="\t",
                    header=None,
                    names=NEWS_COLUMNS,
                    dtype=str,
                )
            )
    if not frames:
        raise FileNotFoundError("No news.tsv files found.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("newid", keep="last")


def scan_users_items(behavior_paths: list[Path]) -> tuple[set[str], set[str]]:
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


def add_history_counts(counts: Counter[str], history: str, item_to_category: dict[str, str]) -> None:
    if not history or history == "-":
        return
    for item_id in history.split():
        category = item_to_category.get(item_id, "")
        if category:
            counts[category] += 1


def train_user_category_counts(train_dir: Path, item_to_category: dict[str, str]) -> dict[str, Counter[str]]:
    user_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for _, user_id, _, history, impressions in behavior_rows(train_dir / "behaviors.tsv"):
        add_history_counts(user_counts[user_id], history, item_to_category)
        for candidate in impressions.split():
            if "-" not in candidate:
                continue
            item_id, label = candidate.rsplit("-", 1)
            if label == "1":
                category = item_to_category.get(item_id, "")
                if category:
                    user_counts[user_id][category] += 1
    return user_counts


def weighted_features_from_counts(prefix: str, counts: Counter[str], scale: float, max_features: int = 8):
    total = sum(counts.values())
    if not total or scale <= 0:
        return []
    return [
        (f"{prefix}{category}", scale * count / total)
        for category, count in counts.most_common(max_features)
        if count > 0
    ]


def build_train_user_features(
    dataset: Dataset,
    all_users: list[str],
    train_users: set[str],
    train_user_counts: dict[str, Counter[str]],
    global_user_weight: float,
    known_user_id_weight: float,
    train_history_weight: float,
):
    rows = []
    for user_id in all_users:
        features = {}
        if global_user_weight > 0:
            features["global:user"] = global_user_weight
        if user_id in train_users:
            if known_user_id_weight > 0:
                features[f"uid:{user_id}"] = known_user_id_weight
            features.update(
                dict(
                    weighted_features_from_counts(
                        "hist:",
                        train_user_counts.get(user_id, Counter()),
                        train_history_weight,
                    )
                )
            )
        if not features:
            features["global:user"] = 1.0
        rows.append((user_id, features))
    return dataset.build_user_features(rows, normalize=True)


def build_item_features(
    dataset: Dataset,
    all_items: list[str],
    train_items: set[str],
    item_to_category: dict[str, str],
    known_item_id_weight: float,
    item_category_weight: float,
):
    rows = []
    for item_id in all_items:
        features = {}
        if item_id in train_items and known_item_id_weight > 0:
            features[f"iid:{item_id}"] = known_item_id_weight
        category = item_to_category.get(item_id, "")
        if category and item_category_weight > 0:
            features[f"cat:{category}"] = item_category_weight
        if not features:
            features["cat:__unknown__"] = 1.0
        rows.append((item_id, features))
    return dataset.build_item_features(rows, normalize=True)


def build_single_user_feature_row(
    user_feature_map: dict[str, int],
    user_id: str,
    history: str,
    train_users: set[str],
    item_to_category: dict[str, str],
    global_user_weight: float,
    known_user_id_weight: float,
    predict_known_history_weight: float,
    predict_unknown_history_weight: float,
) -> csr_matrix:
    counts: Counter[str] = Counter()
    add_history_counts(counts, history, item_to_category)

    weighted = []
    if global_user_weight > 0:
        weighted.append(("global:user", global_user_weight))
    if user_id in train_users:
        weighted.append((f"uid:{user_id}", known_user_id_weight))
        history_weight = predict_known_history_weight
    else:
        history_weight = predict_unknown_history_weight
    weighted.extend(weighted_features_from_counts("hist:", counts, history_weight))

    cols = []
    values = []
    for feature_name, value in weighted:
        col = user_feature_map.get(feature_name)
        if col is not None and value > 0:
            cols.append(col)
            values.append(float(value))

    if not cols:
        cols = [user_feature_map["global:user"]]
        values = [1.0]

    values_array = np.asarray(values, dtype=np.float32)
    values_array /= values_array.sum()
    return csr_matrix(
        (values_array, ([0] * len(cols), cols)),
        shape=(1, len(user_feature_map)),
        dtype=np.float32,
    )


def iter_labeled_rows(path: Path):
    for impression_id, user_id, _, history, impressions in behavior_rows(path):
        item_ids = []
        labels = []
        for candidate in impressions.split():
            if "-" not in candidate:
                continue
            item_id, label = candidate.rsplit("-", 1)
            item_ids.append(item_id)
            labels.append(int(label))
        if item_ids and any(labels):
            yield impression_id, user_id, history, item_ids, np.asarray(labels, dtype=np.int8)


def evaluate_labeled_impressions(
    model: LightFM,
    dataset: Dataset,
    eval_path: Path,
    item_features,
    train_users: set[str],
    item_to_category: dict[str, str],
    eval_limit: int,
    num_threads: int,
    global_user_weight: float,
    known_user_id_weight: float,
    predict_known_history_weight: float,
    predict_unknown_history_weight: float,
) -> dict:
    user_feature_map = dataset.mapping()[1]
    item_map = dataset.mapping()[2]
    metrics = defaultdict(list)
    coverage = Counter()

    for _, user_id, history, item_ids, labels in iter_labeled_rows(eval_path):
        coverage["groups_seen"] += 1
        coverage["candidate_items_seen"] += len(item_ids)
        known = [(item_map[item_id], labels[idx]) for idx, item_id in enumerate(item_ids) if item_id in item_map]
        coverage["candidate_items_known"] += len(known)
        if len(known) < 2:
            coverage["groups_skipped_no_known_items"] += 1
            continue

        group_labels = np.asarray([label for _, label in known], dtype=np.int8)
        if group_labels.max() == group_labels.min():
            coverage["groups_skipped_single_class"] += 1
            continue

        item_indices = np.asarray([item_idx for item_idx, _ in known], dtype=np.int32)
        user_features = build_single_user_feature_row(
            user_feature_map,
            user_id,
            history,
            train_users,
            item_to_category,
            global_user_weight,
            known_user_id_weight,
            predict_known_history_weight,
            predict_unknown_history_weight,
        )
        scores = model.predict(
            np.zeros(len(item_indices), dtype=np.int32),
            item_indices,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
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
                f"  seen {coverage['groups_seen']:,}; evaluated {coverage['groups_evaluated']:,}",
                flush=True,
            )

    result = {name: float(np.nanmean(values)) for name, values in metrics.items()}
    result.update({key: int(value) for key, value in coverage.items()})
    if coverage["candidate_items_seen"]:
        result["known_candidate_item_ratio"] = coverage["candidate_items_known"] / coverage["candidate_items_seen"]
    if coverage["groups_seen"]:
        result["evaluated_group_ratio"] = coverage["groups_evaluated"] / coverage["groups_seen"]
    return result


def ranks_from_scores(scores: np.ndarray) -> list[int]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def generate_submission(
    model: LightFM,
    dataset: Dataset,
    test_path: Path,
    item_features,
    train_users: set[str],
    item_to_category: dict[str, str],
    output_dir: Path,
    num_threads: int,
    global_user_weight: float,
    known_user_id_weight: float,
    predict_known_history_weight: float,
    predict_unknown_history_weight: float,
) -> dict:
    user_feature_map = dataset.mapping()[1]
    item_map = dataset.mapping()[2]
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / "prediction.zip"
    stats = Counter()

    with prediction_path.open("w", encoding="utf-8") as out:
        for impression_id, user_id, _, history, impressions in behavior_rows(test_path):
            candidates = [strip_label(candidate) for candidate in impressions.split()]
            stats["groups"] += 1
            stats["candidates"] += len(candidates)
            item_indices = np.asarray([item_map[item_id] for item_id in candidates], dtype=np.int32)
            user_features = build_single_user_feature_row(
                user_feature_map,
                user_id,
                history,
                train_users,
                item_to_category,
                global_user_weight,
                known_user_id_weight,
                predict_known_history_weight,
                predict_unknown_history_weight,
            )
            scores = model.predict(
                np.zeros(len(item_indices), dtype=np.int32),
                item_indices,
                user_features=user_features,
                item_features=item_features,
                num_threads=num_threads,
            )
            out.write(f"{impression_id} {json.dumps(ranks_from_scores(scores), separators=(',', ':'))}\n")
            if stats["groups"] % 100000 == 0:
                print(f"  wrote {stats['groups']:,} groups", flush=True)

    validation = validate_prediction_file(test_path, prediction_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, arcname="prediction.txt")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    if names != ["prediction.txt"]:
        raise ValueError(f"Invalid zip contents: {names}")

    return {
        "prediction_path": str(prediction_path),
        "zip_path": str(zip_path),
        "generation": dict(stats),
        "validation": validation,
    }


def validate_prediction_file(behaviors_path: Path, prediction_path: Path) -> dict:
    stats = Counter()
    with behaviors_path.open("r", encoding="utf-8") as behaviors, prediction_path.open("r", encoding="utf-8") as preds:
        for line_no, behavior_line in enumerate(behaviors, start=1):
            parts = behavior_line.rstrip("\n").split("\t")
            pred_line = preds.readline()
            if not pred_line:
                raise ValueError(f"Missing prediction line {line_no}")
            pred_id, raw_ranks = pred_line.rstrip("\n").split(" ", 1)
            if pred_id != parts[0]:
                raise ValueError(f"Line {line_no}: expected {parts[0]}, got {pred_id}")
            candidate_count = len(parts[4].split())
            ranks = json.loads(raw_ranks)
            if sorted(ranks) != list(range(1, candidate_count + 1)):
                raise ValueError(f"Line {line_no}: ranks are not 1..{candidate_count}")
            stats["groups"] += 1
            stats["candidates"] += candidate_count
        if preds.readline():
            raise ValueError("Prediction file has extra lines.")
    return dict(stats)


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dir = test_dir if args.mode == "submit" else eval_dir
    print(f"Preparing refined user-history LightFM for mode={args.mode}", flush=True)

    news = load_combined_news([train_dir, eval_dir, test_dir])
    refined_news = make_refined_categories(news)
    item_to_category = refined_news.set_index("newid")["new_category"].fillna("").to_dict()
    categories = sorted({f"hist:{cat}" for cat in refined_news["new_category"].dropna().unique()})
    item_categories = sorted({f"cat:{cat}" for cat in refined_news["new_category"].dropna().unique()} | {"cat:__unknown__"})

    train_users, train_items = scan_users_items([train_dir / "behaviors.tsv"])
    target_users, target_items = scan_users_items([target_dir / "behaviors.tsv"])
    all_users = sorted(train_users | target_users)
    all_items = sorted(train_items | target_items | set(news["newid"]))
    train_news_items = train_items | set(load_news(train_dir)["newid"])

    user_feature_names = ["global:user"] + [f"uid:{user_id}" for user_id in sorted(train_users)] + categories
    item_feature_names = [f"iid:{item_id}" for item_id in sorted(train_news_items)] + item_categories

    print(
        f"users={len(all_users):,} items={len(all_items):,} "
        f"user_features={len(user_feature_names):,} item_features={len(item_feature_names):,}",
        flush=True,
    )

    artifact_paths = {
        "model": output_dir / "model.pkl",
        "dataset": output_dir / "dataset.pkl",
        "item_features": output_dir / "item_features.pkl",
        "config": output_dir / "config.json",
    }

    if args.load_existing_model:
        with artifact_paths["model"].open("rb") as f:
            model = pickle.load(f)
        with artifact_paths["dataset"].open("rb") as f:
            dataset = pickle.load(f)
        with artifact_paths["item_features"].open("rb") as f:
            item_features = pickle.load(f)
    else:
        dataset = Dataset(user_identity_features=False, item_identity_features=False)
        dataset.fit(
            users=all_users,
            items=all_items,
            user_features=user_feature_names,
            item_features=item_feature_names,
        )
        train_user_counts = train_user_category_counts(train_dir, item_to_category)
        user_features = build_train_user_features(
            dataset,
            all_users,
            train_users,
            train_user_counts,
            args.global_user_weight,
            args.known_user_id_weight,
            args.train_history_weight,
        )
        item_features = build_item_features(
            dataset,
            all_items,
            train_news_items,
            item_to_category,
            args.known_item_id_weight,
            args.item_category_weight,
        )
        interactions, _ = dataset.build_interactions(iter_interactions(train_dir / "behaviors.tsv"))
        print(
            f"interactions={interactions.shape} nnz={interactions.nnz:,} "
            f"user_features={user_features.shape} item_features={item_features.shape}",
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
            user_features=user_features,
            item_features=item_features,
            epochs=args.epochs,
            num_threads=args.num_threads,
            verbose=True,
        )

        with artifact_paths["model"].open("wb") as f:
            pickle.dump(model, f)
        with artifact_paths["dataset"].open("wb") as f:
            pickle.dump(dataset, f)
        with artifact_paths["item_features"].open("wb") as f:
            pickle.dump(item_features, f)

    config = {
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "test_dir": str(test_dir),
        "loss": args.loss,
        "components": args.components,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "item_alpha": args.item_alpha,
        "user_alpha": args.user_alpha,
        "user_identity_features": False,
        "item_identity_features": False,
        "explicit_train_uid_features": True,
        "explicit_train_iid_features": True,
        "history_category_features": True,
        "global_user_weight": args.global_user_weight,
        "known_user_id_weight": args.known_user_id_weight,
        "train_history_weight": args.train_history_weight,
        "predict_known_history_weight": args.predict_known_history_weight,
        "predict_unknown_history_weight": args.predict_unknown_history_weight,
        "known_item_id_weight": args.known_item_id_weight,
        "item_category_weight": args.item_category_weight,
    }

    if args.mode == "eval":
        result = evaluate_labeled_impressions(
            model,
            dataset,
            eval_dir / "behaviors.tsv",
            item_features,
            train_users,
            item_to_category,
            args.eval_limit,
            args.num_threads,
            args.global_user_weight,
            args.known_user_id_weight,
            args.predict_known_history_weight,
            args.predict_unknown_history_weight,
        )
        output = {"config": config, "eval": result}
        with (output_dir / "dev_metrics.json").open("w") as f:
            json.dump(output, f, indent=2)
        print(json.dumps(output, indent=2), flush=True)
    else:
        submission = generate_submission(
            model,
            dataset,
            test_dir / "behaviors.tsv",
            item_features,
            train_users,
            item_to_category,
            output_dir,
            args.num_threads,
            args.global_user_weight,
            args.known_user_id_weight,
            args.predict_known_history_weight,
            args.predict_unknown_history_weight,
        )
        output = {"config": config, "submission": submission}
        with (output_dir / "submission_metadata.json").open("w") as f:
            json.dump(output, f, indent=2)
        print(json.dumps(output, indent=2), flush=True)

    with artifact_paths["config"].open("w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
