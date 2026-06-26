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
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


NEWS_COLUMNS = [
    "newid",
    "vertical",
    "subvertical",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

NEWSCAT_REMAP = {
    "middleeast": "newsworld",
    "northamerica": "newspolitics",
    "newsnational": "newsworld",
    "newsus": "newsworld",
    "newselection2020": "newspolitics",
    "newsworldpolitics": "newspolitics",
    "newsscience": "newsscienceandtechnology",
    "newsrealestate": "newsworld",
    "newsweather": "weather",
    "newsfactcheck": "news",
    "newsother": "news",
    "kids": "lifestyle",
    "newsvideo": "video",
    "newstvmedia": "video",
    "newsoffbeat": "news",
    "newsopinion": "news",
    "newsgoodnews": "news",
    "newsbusiness": "news",
    "newsphotos": "news",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="MINDlarge_train")
    parser.add_argument("--output-dir", default="results/mindlarge_train")
    parser.add_argument("--models", nargs="+", default=["vertical", "refined", "tfidf", "bert"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--components", type=int, default=30)
    parser.add_argument("--eval-limit", type=int, default=100000, help="0 means evaluate all labeled impressions")
    parser.add_argument(
        "--paper-metric-users",
        type=int,
        default=0,
        help="Evaluate paper metrics on the first N fitted users. Use 0 for all users.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--bert-embeddings", default="english_article_embeddings_large_train.pt")
    parser.add_argument("--bert-ids", default="english_news_ids_large_train.pkl")
    return parser.parse_args()


def load_news(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        data_dir / "news.tsv",
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        dtype=str,
    )


def scan_behaviors(behaviors_path: Path):
    users = set()
    items = set()
    rows = 0
    history_clicks = 0
    positive_impressions = 0
    labeled_groups = 0

    with behaviors_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            _, user_id, _, history, impressions = parts
            users.add(user_id)

            if history and history != "-":
                for item_id in history.split():
                    items.add(item_id)
                    history_clicks += 1

            group_items = []
            group_labels = []
            for imp in impressions.split():
                if "-" not in imp:
                    continue
                item_id, label = imp.rsplit("-", 1)
                if label == "1":
                    positive_impressions += 1
                items.add(item_id)
                group_items.append(item_id)
                group_labels.append(int(label))

            if group_items and any(group_labels):
                labeled_groups += 1

    return users, items, {
        "rows": rows,
        "history_clicks": history_clicks,
        "positive_impressions": positive_impressions,
        "labeled_groups": labeled_groups,
    }


def iter_interactions(behaviors_path: Path):
    with behaviors_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            _, user_id, _, history, impressions = parts
            if history and history != "-":
                for item_id in history.split():
                    yield user_id, item_id
            for imp in impressions.split():
                if "-" not in imp:
                    continue
                item_id, label = imp.rsplit("-", 1)
                if label == "1":
                    yield user_id, item_id


def iter_labeled_impressions(behaviors_path: Path):
    with behaviors_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            _, user_id, _, _, impressions = parts
            item_ids = []
            labels = []
            for imp in impressions.split():
                if "-" not in imp:
                    continue
                item_id, label = imp.rsplit("-", 1)
                item_ids.append(item_id)
                labels.append(int(label))
            if item_ids and any(labels):
                yield user_id, item_ids, np.asarray(labels, dtype=np.int8)


def make_refined_categories(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    news_children = [sv for sv in news["subvertical"].dropna().unique() if sv.startswith("news")]
    news["new_category"] = news["subvertical"].where(news["subvertical"].isin(news_children), news["vertical"])
    news["new_category"] = news["new_category"].replace(NEWSCAT_REMAP)
    return news


def fit_dataset(behaviors_path: Path, users, items, feature_names):
    ds = Dataset()
    ds.fit(users=users, items=items, item_features=feature_names)
    matrix, _ = ds.build_interactions(iter_interactions(behaviors_path))
    return ds, matrix


def build_category_features(ds: Dataset, news: pd.DataFrame, category_col: str):
    item_map = ds.mapping()[2]
    cat_map = news.set_index("newid")[category_col].fillna("").to_dict()
    tuples = [(item_id, [cat_map[item_id]]) for item_id in item_map if item_id in cat_map and cat_map[item_id]]
    return ds.build_item_features(tuples)


def build_tfidf_features(ds: Dataset, news: pd.DataFrame):
    item_map = ds.mapping()[2]
    ordered_ids = [item_id for item_id, _ in sorted(item_map.items(), key=lambda x: x[1])]
    aligned = news.set_index("newid").reindex(ordered_ids).fillna("")
    vertical_matrix = csr_matrix(pd.get_dummies(aligned["vertical"], prefix="vert").values)
    tfidf_matrix = TfidfVectorizer(max_features=2000, stop_words="english").fit_transform(aligned["title"])
    return hstack([vertical_matrix, tfidf_matrix], format="csr")


def build_bert_features(ds: Dataset, news: pd.DataFrame, emb_path: Path, ids_path: Path):
    import torch

    item_map = ds.mapping()[2]
    ordered_ids = [item_id for item_id, _ in sorted(item_map.items(), key=lambda x: x[1])]
    aligned = news.set_index("newid").reindex(ordered_ids).fillna("")
    vertical_matrix = csr_matrix(pd.get_dummies(aligned["vertical"], prefix="vert").values)

    embeddings = torch.load(emb_path, map_location="cpu").numpy()
    with ids_path.open("rb") as f:
        embedding_ids = pickle.load(f)
    id_to_row = {item_id: i for i, item_id in enumerate(embedding_ids)}

    dense = np.zeros((len(ordered_ids), embeddings.shape[1]), dtype=np.float32)
    for row_idx, item_id in enumerate(ordered_ids):
        src_idx = id_to_row.get(item_id)
        if src_idx is not None:
            dense[row_idx] = embeddings[src_idx]

    reduced = TruncatedSVD(n_components=50, random_state=42).fit_transform(dense)
    return hstack([vertical_matrix, csr_matrix(reduced)], format="csr")


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


def evaluate_ranking(model: LightFM, ds: Dataset, behaviors_path: Path, item_features, eval_limit: int):
    user_map, _, item_map, _ = ds.mapping()
    metrics = defaultdict(list)
    seen = 0

    for idx, (user_id, item_ids, labels) in enumerate(iter_labeled_impressions(behaviors_path), start=1):
        user_idx = user_map.get(user_id)
        if user_idx is None:
            continue
        known = [(item_map[item_id], labels[i]) for i, item_id in enumerate(item_ids) if item_id in item_map]
        if len(known) < 2:
            continue
        item_indices = np.asarray([x[0] for x in known], dtype=np.int32)
        group_labels = np.asarray([x[1] for x in known], dtype=np.int8)
        if group_labels.max() == group_labels.min():
            continue
        scores = model.predict(np.repeat(user_idx, len(item_indices)), item_indices, item_features=item_features)
        labels_sorted = group_labels[np.argsort(-scores)]
        metrics["AUC"].append(auc_score_group(group_labels, scores))
        metrics["MRR"].append(mrr_score(labels_sorted))
        metrics["nDCG@5"].append(ndcg_score(labels_sorted, 5))
        metrics["nDCG@10"].append(ndcg_score(labels_sorted, 10))
        seen += 1
        if eval_limit and seen >= eval_limit:
            break
        if idx % 10000 == 0:
            print(f"  evaluated {idx:,} impression groups", flush=True)

    result = {name: float(np.nanmean(values)) for name, values in metrics.items()}
    result["groups_evaluated"] = seen
    return result


def evaluate_paper_metrics(model: LightFM, interaction_matrix, item_features, max_users: int):
    if max_users:
        coo = interaction_matrix.tocoo()
        keep = coo.row < max_users
        matrix = csr_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=interaction_matrix.shape,
        )
    else:
        matrix = interaction_matrix.tocsr()

    return {
        "Precision@10": float(
            precision_at_k(model, matrix, item_features=item_features, k=10).mean()
        ),
        "Recall@10": float(
            recall_at_k(model, matrix, item_features=item_features, k=10).mean()
        ),
        "LightFM_AUC": float(
            lightfm_auc_score(model, matrix, item_features=item_features).mean()
        ),
        "paper_metric_users": int(max_users or matrix.shape[0]),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading news and behaviors")
    news = load_news(data_dir)
    refined_news = make_refined_categories(news)
    behaviors_path = data_dir / "behaviors.tsv"
    users, items, scan_stats = scan_behaviors(behaviors_path)
    news_items = set(news["newid"])
    all_items = sorted(items | news_items)
    all_users = sorted(users)
    print(
        f"users={len(all_users):,} items={len(all_items):,} "
        f"history_clicks={scan_stats['history_clicks']:,} "
        f"positive_impressions={scan_stats['positive_impressions']:,} "
        f"labeled_groups={scan_stats['labeled_groups']:,}"
    )

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open() as f:
            results = json.load(f)
    else:
        results = {}
    for model_name in args.models:
        print(f"\n=== {model_name} ===", flush=True)
        if model_name == "vertical":
            features = sorted(news["vertical"].dropna().unique())
            ds, interaction_matrix = fit_dataset(behaviors_path, all_users, all_items, features)
            item_features = build_category_features(ds, news, "vertical")
        elif model_name == "refined":
            features = sorted(refined_news["new_category"].dropna().unique())
            ds, interaction_matrix = fit_dataset(behaviors_path, all_users, all_items, features)
            item_features = build_category_features(ds, refined_news, "new_category")
        elif model_name == "tfidf":
            ds, interaction_matrix = fit_dataset(behaviors_path, all_users, all_items, [])
            item_features = build_tfidf_features(ds, news)
        elif model_name == "bert":
            emb_path = Path(args.bert_embeddings)
            ids_path = Path(args.bert_ids)
            if not emb_path.exists() or not ids_path.exists():
                raise FileNotFoundError(f"Missing BERT artifacts: {emb_path} / {ids_path}")
            ds, interaction_matrix = fit_dataset(behaviors_path, all_users, all_items, [])
            item_features = build_bert_features(ds, news, emb_path, ids_path)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"interaction_matrix={interaction_matrix.shape} item_features={item_features.shape}", flush=True)
        save_npz(output_dir / f"{model_name}_item_features.npz", item_features)

        model = LightFM(no_components=args.components, loss="warp")
        model.fit(interaction_matrix, item_features=item_features, epochs=args.epochs, num_threads=args.num_threads, verbose=True)

        with (output_dir / f"{model_name}_model.pkl").open("wb") as f:
            pickle.dump(model, f)

        result = evaluate_paper_metrics(model, interaction_matrix, item_features, args.paper_metric_users)
        result.update(evaluate_ranking(model, ds, behaviors_path, item_features, args.eval_limit))
        results[model_name] = result
        print(model_name, result, flush=True)
        with metrics_path.open("w") as f:
            json.dump(results, f, indent=2)

    print("\nFinal results")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
