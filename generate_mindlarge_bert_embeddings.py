from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


NEWS_COLUMNS = [
    "newid",
    "vertical",
    "subvertical",
    "title",
    "abstract",
    "url",
    "entities in title",
    "entities in abstract",
]


def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", default="MINDlarge_train/news.tsv")
    parser.add_argument("--output-embeddings", default="english_article_embeddings_large_train.pt")
    parser.add_argument("--output-ids", default="english_news_ids_large_train.pkl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    news_path = Path(args.news)
    embeddings_path = Path(args.output_embeddings)
    ids_path = Path(args.output_ids)

    mind_news = pd.read_csv(
        news_path,
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        dtype=str,
    )
    mind_news["title_clean"] = mind_news["title"].fillna("").apply(clean)
    mind_news["abstract_clean"] = mind_news["abstract"].fillna("").apply(clean)
    mind_news["full_text"] = mind_news["title_clean"] + " " + mind_news["abstract_clean"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    texts = mind_news["full_text"].tolist()
    embeddings = []
    total = len(texts)

    with torch.no_grad():
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            batch = texts[start:end]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            attention = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * attention, dim=1)
            counts = torch.clamp(attention.sum(dim=1), min=1e-9)
            embeddings.append((summed / counts).cpu())

            if end == total or end % (args.batch_size * 20) == 0:
                print(f"embedded {end:,}/{total:,}", flush=True)

    english_embeddings = torch.cat(embeddings, dim=0)
    torch.save(english_embeddings, embeddings_path)
    with ids_path.open("wb") as f:
        pickle.dump(mind_news["newid"].tolist(), f)

    print(f"saved {embeddings_path} {tuple(english_embeddings.shape)}")
    print(f"saved {ids_path} {len(mind_news):,} ids")


if __name__ == "__main__":
    main()
