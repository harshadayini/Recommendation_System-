from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import zipfile
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sparse_normalize

from run_mindlarge_train_metrics import (
    NEWS_COLUMNS,
    auc_score_group,
    load_news,
    make_refined_categories,
    mrr_score,
    ndcg_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid tuned refined LightFM + TF-IDF history-similarity cold-case evaluator/submission."
    )
    parser.add_argument("--mode", choices=["eval", "submit", "sweep"], default="eval")
    parser.add_argument("--train-dir", default="MINDlarge_train")
    parser.add_argument("--eval-dir", default="MINDlarge_dev")
    parser.add_argument("--test-dir", default="MINDlarge_test")
    parser.add_argument("--output-dir", default="results/hybrid_refined_tfidf_history")
    parser.add_argument("--model-path", default="submissions/refined_mindlarge_test_train_only/refined_submission_model.pkl")
    parser.add_argument("--dataset-path", default="submissions/refined_mindlarge_test_train_only/refined_submission_dataset.pkl")
    parser.add_argument(
        "--item-features-path",
        default="submissions/refined_mindlarge_test_train_only/refined_submission_item_features.pkl",
    )
    parser.add_argument("--eval-limit", type=int, default=50000)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--history-limit", type=int, default=100)
    parser.add_argument("--known-bias", type=float, default=0.0)
    parser.add_argument("--category-boost", type=float, default=0.02)
    parser.add_argument("--popularity-boost", type=float, default=0.001)
    parser.add_argument("--known-biases", default="-0.05,0,0.05,0.1")
    parser.add_argument("--category-boosts", default="0,0.02")
    parser.add_argument("--popularity-boosts", default="0,0.001")
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def parse_float_csv(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


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


def load_combined_news(dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for data_dir in dirs:
        news_path = data_dir / "news.tsv"
        if news_path.exists():
            frames.append(pd.read_csv(news_path, sep="\t", header=None, names=NEWS_COLUMNS, dtype=str))
    if not frames:
        raise FileNotFoundError("No news.tsv files found.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("newid", keep="last").reset_index(drop=True)


def scan_train_users_items(train_dir: Path) -> tuple[set[str], set[str], Counter[str]]:
    users: set[str] = set()
    items: set[str] = set(load_news(train_dir)["newid"])
    popularity: Counter[str] = Counter()
    for _, user_id, _, history, impressions in behavior_rows(train_dir / "behaviors.tsv"):
        users.add(user_id)
        if history and history != "-":
            clicked = history.split()
            items.update(clicked)
            popularity.update(clicked)
        for candidate in impressions.split():
            if "-" not in candidate:
                continue
            item_id, label = candidate.rsplit("-", 1)
            items.add(item_id)
            if label == "1":
                popularity[item_id] += 1
    return users, items, popularity


def text_for_news(news: pd.DataFrame) -> pd.Series:
    return (
        news["title"].fillna("")
        + " "
        + news["abstract"].fillna("")
        + " "
        + news["vertical"].fillna("")
        + " "
        + news["subvertical"].fillna("")
    )


def build_tfidf(news: pd.DataFrame, max_features: int):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(text_for_news(news))
    matrix = sparse_normalize(matrix, norm="l2", copy=False)
    return vectorizer, matrix.tocsr()


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float64)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if min_value == max_value:
        return np.zeros(len(values), dtype=np.float64)
    return (values - min_value) / (max_value - min_value)


def ranks_from_scores(scores: np.ndarray) -> list[int]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def history_vector(
    history: str,
    news_to_row: dict[str, int],
    tfidf_matrix,
    history_limit: int,
) -> csr_matrix | None:
    if not history or history == "-":
        return None
    rows = [news_to_row[item_id] for item_id in history.split()[-history_limit:] if item_id in news_to_row]
    if not rows:
        return None
    vector = tfidf_matrix[rows].mean(axis=0)
    vector = csr_matrix(vector)
    vector = sparse_normalize(vector, norm="l2", copy=False)
    return vector


def history_categories(history: str, item_to_category: dict[str, str], history_limit: int) -> set[str]:
    if not history or history == "-":
        return set()
    return {
        item_to_category[item_id]
        for item_id in history.split()[-history_limit:]
        if item_to_category.get(item_id)
    }


def fallback_scores(
    candidates: list[str],
    history: str,
    news_to_row: dict[str, int],
    tfidf_matrix,
    item_to_category: dict[str, str],
    popularity: Counter[str],
    history_limit: int,
    category_boost: float,
    popularity_boost: float,
) -> np.ndarray:
    user_vector = history_vector(history, news_to_row, tfidf_matrix, history_limit)
    scores = np.zeros(len(candidates), dtype=np.float64)
    if user_vector is not None:
        candidate_rows = [news_to_row.get(item_id) for item_id in candidates]
        known_positions = [idx for idx, row in enumerate(candidate_rows) if row is not None]
        if known_positions:
            rows = [candidate_rows[idx] for idx in known_positions]
            sims = tfidf_matrix[rows].dot(user_vector.T).toarray().ravel()
            for pos, score in zip(known_positions, sims):
                scores[pos] = float(score)

    hist_categories = history_categories(history, item_to_category, history_limit)
    if category_boost and hist_categories:
        for idx, item_id in enumerate(candidates):
            if item_to_category.get(item_id) in hist_categories:
                scores[idx] += category_boost
    if popularity_boost:
        for idx, item_id in enumerate(candidates):
            scores[idx] += popularity_boost * math.log1p(popularity.get(item_id, 0))
    return scores


def hybrid_scores(
    model,
    dataset,
    item_features,
    train_users: set[str],
    train_items: set[str],
    tfidf_matrix,
    news_to_row: dict[str, int],
    item_to_category: dict[str, str],
    popularity: Counter[str],
    user_id: str,
    history: str,
    candidates: list[str],
    history_limit: int,
    known_bias: float,
    category_boost: float,
    popularity_boost: float,
    num_threads: int,
) -> tuple[np.ndarray, Counter[str]]:
    stats: Counter[str] = Counter()
    user_map, _, item_map, _ = dataset.mapping()
    fallback_raw = fallback_scores(
        candidates,
        history,
        news_to_row,
        tfidf_matrix,
        item_to_category,
        popularity,
        history_limit,
        category_boost,
        popularity_boost,
    )
    fallback_norm = normalize(fallback_raw)
    scores = fallback_norm.copy()

    user_idx = user_map.get(user_id)
    known_user = user_id in train_users and user_idx is not None
    if not known_user:
        stats["cold_user_groups"] += 1
        return scores, stats

    known_positions = []
    known_item_indices = []
    for pos, item_id in enumerate(candidates):
        item_idx = item_map.get(item_id)
        if item_id in train_items and item_idx is not None:
            known_positions.append(pos)
            known_item_indices.append(item_idx)
        else:
            stats["cold_candidate_items"] += 1

    if known_item_indices:
        lightfm_raw = model.predict(
            np.repeat(user_idx, len(known_item_indices)),
            np.asarray(known_item_indices, dtype=np.int32),
            item_features=item_features,
            num_threads=num_threads,
        )
        lightfm_norm = normalize(np.asarray(lightfm_raw, dtype=np.float64))
        for local_idx, pos in enumerate(known_positions):
            scores[pos] = lightfm_norm[local_idx] + known_bias
    else:
        stats["known_user_groups_with_no_train_candidates"] += 1

    return scores, stats


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


def evaluate(
    model,
    dataset,
    item_features,
    train_users: set[str],
    train_items: set[str],
    eval_path: Path,
    tfidf_matrix,
    news_to_row: dict[str, int],
    item_to_category: dict[str, str],
    popularity: Counter[str],
    args: argparse.Namespace,
) -> dict:
    return evaluate_with_params(
        model,
        dataset,
        item_features,
        train_users,
        train_items,
        eval_path,
        tfidf_matrix,
        news_to_row,
        item_to_category,
        popularity,
        args.eval_limit,
        args.history_limit,
        args.known_bias,
        args.category_boost,
        args.popularity_boost,
        args.num_threads,
    )


def evaluate_with_params(
    model,
    dataset,
    item_features,
    train_users: set[str],
    train_items: set[str],
    eval_path: Path,
    tfidf_matrix,
    news_to_row: dict[str, int],
    item_to_category: dict[str, str],
    popularity: Counter[str],
    eval_limit: int,
    history_limit: int,
    known_bias: float,
    category_boost: float,
    popularity_boost: float,
    num_threads: int,
) -> dict:
    metrics = {"AUC": [], "MRR": [], "nDCG@5": [], "nDCG@10": []}
    coverage: Counter[str] = Counter()
    for _, user_id, history, item_ids, labels in iter_labeled_rows(eval_path):
        coverage["groups_seen"] += 1
        coverage["candidate_items_seen"] += len(item_ids)
        if labels.max() == labels.min():
            coverage["groups_skipped_single_class"] += 1
            continue
        scores, stats = hybrid_scores(
            model,
            dataset,
            item_features,
            train_users,
            train_items,
            tfidf_matrix,
            news_to_row,
            item_to_category,
            popularity,
            user_id,
            history,
            item_ids,
            history_limit,
            known_bias,
            category_boost,
            popularity_boost,
            num_threads,
        )
        coverage.update(stats)
        labels_sorted = labels[np.argsort(-scores)]
        metrics["AUC"].append(auc_score_group(labels, scores))
        metrics["MRR"].append(mrr_score(labels_sorted))
        metrics["nDCG@5"].append(ndcg_score(labels_sorted, 5))
        metrics["nDCG@10"].append(ndcg_score(labels_sorted, 10))
        coverage["groups_evaluated"] += 1
        if eval_limit and coverage["groups_evaluated"] >= eval_limit:
            break
        if coverage["groups_seen"] % 10000 == 0:
            print(f"  seen {coverage['groups_seen']:,}; evaluated {coverage['groups_evaluated']:,}", flush=True)

    result = {name: float(np.nanmean(values)) for name, values in metrics.items()}
    result.update({key: int(value) for key, value in coverage.items()})
    result["evaluated_group_ratio"] = coverage["groups_evaluated"] / coverage["groups_seen"] if coverage["groups_seen"] else 0.0
    return result


def generate_submission(
    model,
    dataset,
    item_features,
    train_users: set[str],
    train_items: set[str],
    test_path: Path,
    tfidf_matrix,
    news_to_row: dict[str, int],
    item_to_category: dict[str, str],
    popularity: Counter[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    prediction_path = output_dir / "prediction.txt"
    zip_path = output_dir / "prediction.zip"
    coverage: Counter[str] = Counter()

    with prediction_path.open("w", encoding="utf-8") as out:
        for impression_id, user_id, _, history, impressions in behavior_rows(test_path):
            candidates = [strip_label(candidate) for candidate in impressions.split()]
            coverage["groups"] += 1
            coverage["candidates"] += len(candidates)
            scores, stats = hybrid_scores(
                model,
                dataset,
                item_features,
                train_users,
                train_items,
                tfidf_matrix,
                news_to_row,
                item_to_category,
                popularity,
                user_id,
                history,
                candidates,
                args.history_limit,
                args.known_bias,
                args.category_boost,
                args.popularity_boost,
                args.num_threads,
            )
            coverage.update(stats)
            out.write(f"{impression_id} {json.dumps(ranks_from_scores(scores), separators=(',', ':'))}\n")
            if coverage["groups"] % 100000 == 0:
                print(f"  wrote {coverage['groups']:,} groups", flush=True)

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
        "generation": dict(coverage),
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


def write_sweep_row(csv_path: Path, row: dict) -> None:
    fieldnames = [
        "known_bias",
        "category_boost",
        "popularity_boost",
        "AUC",
        "MRR",
        "nDCG@5",
        "nDCG@10",
        "groups_seen",
        "groups_evaluated",
        "candidate_items_seen",
        "cold_user_groups",
        "cold_candidate_items",
        "known_user_groups_with_no_train_candidates",
        "evaluated_group_ratio",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dir = test_dir if args.mode == "submit" else eval_dir
    for path in [Path(args.model_path), Path(args.dataset_path), Path(args.item_features_path), target_dir / "behaviors.tsv"]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading tuned refined LightFM artifacts", flush=True)
    with Path(args.model_path).open("rb") as f:
        model = pickle.load(f)
    with Path(args.dataset_path).open("rb") as f:
        dataset = pickle.load(f)
    with Path(args.item_features_path).open("rb") as f:
        item_features = pickle.load(f)

    print("Loading news and building TF-IDF vectors", flush=True)
    news = load_combined_news([train_dir, eval_dir, test_dir])
    news_to_row = {item_id: idx for idx, item_id in enumerate(news["newid"])}
    _, tfidf_matrix = build_tfidf(news, args.max_features)
    refined_news = make_refined_categories(news)
    item_to_category = refined_news.set_index("newid")["new_category"].fillna("").to_dict()
    train_users, train_items, popularity = scan_train_users_items(train_dir)
    print(
        f"news={len(news):,} tfidf={tfidf_matrix.shape} "
        f"train_users={len(train_users):,} train_items={len(train_items):,}",
        flush=True,
    )

    config = {
        "mode": args.mode,
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "test_dir": str(test_dir),
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "item_features_path": args.item_features_path,
        "max_features": args.max_features,
        "history_limit": args.history_limit,
        "known_bias": args.known_bias,
        "category_boost": args.category_boost,
        "popularity_boost": args.popularity_boost,
        "known_case_policy": "LightFM score only, normalized within known candidates; no TF-IDF blend",
        "cold_case_policy": "TF-IDF clicked-history cosine similarity plus small refined-category/popularity tie-breakers",
    }

    if args.mode == "eval":
        result = evaluate(
            model,
            dataset,
            item_features,
            train_users,
            train_items,
            eval_dir / "behaviors.tsv",
            tfidf_matrix,
            news_to_row,
            item_to_category,
            popularity,
            args,
        )
        output = {"config": config, "eval": result}
        with (output_dir / "dev_metrics.json").open("w") as f:
            json.dump(output, f, indent=2)
        print(json.dumps(output, indent=2), flush=True)
    elif args.mode == "sweep":
        csv_path = output_dir / "hybrid_tuning_results.csv"
        jsonl_path = output_dir / "hybrid_tuning_results.jsonl"
        configs = list(
            product(
                parse_float_csv(args.known_biases),
                parse_float_csv(args.category_boosts),
                parse_float_csv(args.popularity_boosts),
            )
        )
        print(f"Running {len(configs)} hybrid policy configs", flush=True)
        rows = []
        for index, (known_bias, category_boost, popularity_boost) in enumerate(configs, start=1):
            print(
                f"\n=== [{index}/{len(configs)}] "
                f"known_bias={known_bias} category_boost={category_boost} popularity_boost={popularity_boost} ===",
                flush=True,
            )
            result = evaluate_with_params(
                model,
                dataset,
                item_features,
                train_users,
                train_items,
                eval_dir / "behaviors.tsv",
                tfidf_matrix,
                news_to_row,
                item_to_category,
                popularity,
                args.eval_limit,
                args.history_limit,
                known_bias,
                category_boost,
                popularity_boost,
                args.num_threads,
            )
            row = {
                "known_bias": known_bias,
                "category_boost": category_boost,
                "popularity_boost": popularity_boost,
                **result,
            }
            write_sweep_row(csv_path, row)
            with jsonl_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

        best = sorted(rows, key=lambda row: (row["nDCG@10"], row["MRR"], row["AUC"]), reverse=True)[0]
        output = {"config": config, "best_by_nDCG@10": best, "results_csv": str(csv_path)}
        with (output_dir / "hybrid_tuning_summary.json").open("w") as f:
            json.dump(output, f, indent=2)
        print("\nBest config by nDCG@10")
        print(json.dumps(best, indent=2), flush=True)
    else:
        submission = generate_submission(
            model,
            dataset,
            item_features,
            train_users,
            train_items,
            test_dir / "behaviors.tsv",
            tfidf_matrix,
            news_to_row,
            item_to_category,
            popularity,
            output_dir,
            args,
        )
        output = {"config": config, "submission": submission}
        with (output_dir / "submission_metadata.json").open("w") as f:
            json.dump(output, f, indent=2)
        print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
