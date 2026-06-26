from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate_mindlarge_dev import auc_score_group, mrr_score, ndcg_score
from generate_mindlarge_refined_submission import (
    build_fallback_scores,
    count_training_popularity,
    load_combined_news,
    ranks_from_scores,
    scan_behavior_ids,
    strip_label,
)
from run_mindlarge_train_metrics import make_refined_categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose where the tuned refined LightFM submission policy loses dev ranking quality."
    )
    parser.add_argument("--train-dir", default="MINDlarge_train")
    parser.add_argument("--eval-dir", default="MINDlarge_dev")
    parser.add_argument("--model-path", default="submissions/refined_mindlarge_test_train_only/refined_submission_model.pkl")
    parser.add_argument(
        "--dataset-path",
        default="submissions/refined_mindlarge_test_train_only/refined_submission_dataset.pkl",
    )
    parser.add_argument(
        "--item-features-path",
        default="submissions/refined_mindlarge_test_train_only/refined_submission_item_features.pkl",
    )
    parser.add_argument("--output-dir", default="results/refined_lightfm_rootcause_dev")
    parser.add_argument("--eval-limit", type=int, default=100000)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def behavior_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 5:
                yield parts


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


def bin_count(value: int, edges: list[int]) -> str:
    previous = 0
    for edge in edges:
        if value <= edge:
            return f"{previous + 1}-{edge}" if previous else f"0-{edge}"
        previous = edge
    return f">{edges[-1]}"


def category_set(history: str, item_to_category: dict[str, str]) -> set[str]:
    if not history or history == "-":
        return set()
    return {item_to_category[item_id] for item_id in history.split() if item_to_category.get(item_id)}


def score_group(
    model,
    dataset,
    item_features,
    users_with_interactions: set[str],
    fallback_scores: dict[str, float],
    user_id: str,
    item_ids: list[str],
    num_threads: int,
) -> tuple[np.ndarray, Counter[str]]:
    user_map, _, item_map, _ = dataset.mapping()
    stats: Counter[str] = Counter()

    use_model = user_id in users_with_interactions and user_id in user_map
    if not use_model:
        stats["cold_user_groups"] += 1
        return np.asarray([fallback_scores.get(item_id, 0.0) for item_id in item_ids], dtype=np.float64), stats

    scores = np.empty(len(item_ids), dtype=np.float64)
    known_positions = []
    known_item_indices = []
    for pos, item_id in enumerate(item_ids):
        item_idx = item_map.get(item_id)
        if item_idx is None:
            scores[pos] = fallback_scores.get(item_id, 0.0)
            stats["unknown_candidate_items"] += 1
        else:
            known_positions.append(pos)
            known_item_indices.append(item_idx)

    if known_item_indices:
        predicted = model.predict(
            np.repeat(user_map[user_id], len(known_item_indices)),
            np.asarray(known_item_indices, dtype=np.int32),
            item_features=item_features,
            num_threads=num_threads,
        )
        for pos, score in zip(known_positions, predicted):
            scores[pos] = float(score)
    else:
        stats["groups_with_no_known_candidate_items"] += 1

    return scores, stats


def add_metric(bucket: dict[str, list[float]], labels: np.ndarray, scores: np.ndarray) -> None:
    labels_sorted = labels[np.argsort(-scores)]
    bucket["AUC"].append(auc_score_group(labels, scores))
    bucket["MRR"].append(mrr_score(labels_sorted))
    bucket["nDCG@5"].append(ndcg_score(labels_sorted, 5))
    bucket["nDCG@10"].append(ndcg_score(labels_sorted, 10))


def summarize_bucket(name: str, bucket: dict[str, list[float]], extra: dict | None = None) -> dict:
    row = {"slice": name, "groups": len(bucket["AUC"])}
    for metric in ["AUC", "MRR", "nDCG@5", "nDCG@10"]:
        row[metric] = float(np.nanmean(bucket[metric])) if bucket[metric] else None
    if extra:
        row.update(extra)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "slice",
        "groups",
        "AUC",
        "MRR",
        "nDCG@5",
        "nDCG@10",
        "candidate_items",
        "known_train_candidate_ratio",
        "positive_train_known_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tuned refined submission artifacts", flush=True)
    with Path(args.model_path).open("rb") as f:
        model = pickle.load(f)
    with Path(args.dataset_path).open("rb") as f:
        dataset = pickle.load(f)
    with Path(args.item_features_path).open("rb") as f:
        item_features = pickle.load(f)

    print("Loading train/dev metadata", flush=True)
    train_users, train_items = scan_behavior_ids([train_dir / "behaviors.tsv"])
    popularity, users_with_interactions = count_training_popularity([train_dir / "behaviors.tsv"])
    all_news = load_combined_news([train_dir, eval_dir])
    refined_news = make_refined_categories(all_news)
    item_to_category = refined_news.set_index("newid")["new_category"].fillna("").to_dict()
    fallback_scores = build_fallback_scores(all_news, popularity)

    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    slice_counts: Counter[str] = Counter()
    slice_candidates: Counter[str] = Counter()
    slice_known_candidates: Counter[str] = Counter()
    slice_positive_known: Counter[str] = Counter()
    slice_positives: Counter[str] = Counter()
    coverage: Counter[str] = Counter()

    def add_to_slice(name: str, labels: np.ndarray, scores: np.ndarray, item_ids: list[str]) -> None:
        add_metric(buckets[name], labels, scores)
        slice_counts[name] += 1
        slice_candidates[name] += len(item_ids)
        slice_known_candidates[name] += sum(1 for item_id in item_ids if item_id in train_items)
        positive_items = [item_id for item_id, label in zip(item_ids, labels) if label == 1]
        slice_positives[name] += len(positive_items)
        slice_positive_known[name] += sum(1 for item_id in positive_items if item_id in train_items)

    for _, user_id, history, item_ids, labels in iter_labeled_rows(eval_dir / "behaviors.tsv"):
        coverage["groups_seen"] += 1
        coverage["candidate_items_seen"] += len(item_ids)
        if labels.max() == labels.min():
            coverage["groups_skipped_single_class"] += 1
            continue

        scores, score_stats = score_group(
            model,
            dataset,
            item_features,
            users_with_interactions,
            fallback_scores,
            user_id,
            item_ids,
            args.num_threads,
        )
        coverage.update(score_stats)
        coverage["groups_evaluated"] += 1

        known_user = user_id in train_users
        known_candidate_count = sum(1 for item_id in item_ids if item_id in train_items)
        positive_items = [item_id for item_id, label in zip(item_ids, labels) if label == 1]
        positive_known_count = sum(1 for item_id in positive_items if item_id in train_items)
        hist_len = 0 if not history or history == "-" else len(history.split())
        hist_categories = category_set(history, item_to_category)
        positive_categories = {item_to_category.get(item_id, "") for item_id in positive_items}
        positive_category_seen = bool(hist_categories & positive_categories)

        add_to_slice("overall", labels, scores, item_ids)
        add_to_slice("known_user" if known_user else "cold_user", labels, scores, item_ids)
        add_to_slice(
            "all_candidates_train_known" if known_candidate_count == len(item_ids) else "has_new_candidate_items",
            labels,
            scores,
            item_ids,
        )
        add_to_slice(
            "all_positive_items_train_known" if positive_known_count == len(positive_items) else "has_new_positive_item",
            labels,
            scores,
            item_ids,
        )
        add_to_slice(f"history_len_{bin_count(hist_len, [0, 5, 20, 50, 100])}", labels, scores, item_ids)
        add_to_slice(f"candidate_count_{bin_count(len(item_ids), [5, 10, 20, 50, 100])}", labels, scores, item_ids)
        add_to_slice(
            "positive_category_seen_in_history" if positive_category_seen else "positive_category_not_seen_in_history",
            labels,
            scores,
            item_ids,
        )

        if args.eval_limit and coverage["groups_evaluated"] >= args.eval_limit:
            break
        if coverage["groups_seen"] % 10000 == 0:
            print(
                f"  seen {coverage['groups_seen']:,}; evaluated {coverage['groups_evaluated']:,}",
                flush=True,
            )

    rows = []
    for name in sorted(buckets):
        groups = slice_counts[name]
        positives = slice_positives[name]
        row = summarize_bucket(
            name,
            buckets[name],
            {
                "candidate_items": int(slice_candidates[name]),
                "known_train_candidate_ratio": (
                    slice_known_candidates[name] / slice_candidates[name] if slice_candidates[name] else 0.0
                ),
                "positive_train_known_ratio": slice_positive_known[name] / positives if positives else 0.0,
            },
        )
        row["groups"] = int(groups)
        rows.append(row)

    output = {
        "config": {
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "item_features_path": args.item_features_path,
            "eval_limit": args.eval_limit,
            "note": "Uses the saved refined submission policy: LightFM where mapped, popularity/category fallback elsewhere.",
        },
        "coverage": dict(coverage),
        "slices": rows,
    }
    with (output_dir / "rootcause_summary.json").open("w") as f:
        json.dump(output, f, indent=2)
    write_csv(output_dir / "rootcause_slices.csv", rows)
    print(json.dumps(output["coverage"], indent=2), flush=True)
    print(f"Wrote {output_dir / 'rootcause_slices.csv'}", flush=True)


if __name__ == "__main__":
    main()
