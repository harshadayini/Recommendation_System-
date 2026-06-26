from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, save_npz


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


def load_news(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        data_dir / "news.tsv",
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        dtype=str,
    )


def make_refined_categories(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    news_children = [sv for sv in news["subvertical"].dropna().unique() if sv.startswith("news")]
    news["new_category"] = news["subvertical"].where(news["subvertical"].isin(news_children), news["vertical"])
    news["new_category"] = news["new_category"].replace(NEWSCAT_REMAP)
    return news


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Telugu articles with the same classifier-to-LightFM flow used in the paper, aligned to MINDlarge refined categories."
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "classify", "build-features", "all"],
        default="prepare",
    )
    parser.add_argument("--mind-train-dir", default="MINDlarge_train")
    parser.add_argument("--telugu-dir", default="telugu")
    parser.add_argument("--output-dir", default="results/telugu_mindlarge_refined")
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--classifier-dir", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug only; 0 uses all English train rows.")
    parser.add_argument("--max-telugu-samples", type=int, default=0, help="Debug only; 0 uses all Telugu rows.")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_english_labeled(mind_train_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    news = make_refined_categories(load_news(mind_train_dir))
    news["text"] = news["title"].fillna("") + " " + news["abstract"].fillna("")
    categories = sorted(news["new_category"].dropna().unique())
    cat_to_id = {cat: idx for idx, cat in enumerate(categories)}
    news["label"] = news["new_category"].map(cat_to_id)
    news = news[["newid", "text", "new_category", "label"]].dropna(subset=["label"]).reset_index(drop=True)
    news["label"] = news["label"].astype(int)
    return news, categories


def load_telugu_articles(telugu_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(telugu_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {telugu_dir}")
    frames = []
    for path in parquet_files:
        frame = pd.read_parquet(path)
        frame["source_file"] = path.name
        frames.append(frame)
    telugu = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("story_id", keep="first")
        .reset_index(drop=True)
    )
    telugu["story_id"] = telugu["story_id"].astype(str)
    telugu["text"] = telugu["headline"].fillna("") + " " + telugu["article"].fillna("")
    return telugu


def write_category_files(output_dir: Path, categories: list[str]) -> None:
    (output_dir / "mindlarge_refined_categories.txt").write_text(
        "\n".join(categories) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": categories,
                "cat_to_id": {cat: idx for idx, cat in enumerate(categories)},
                "id_to_cat": {str(idx): cat for idx, cat in enumerate(categories)},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def prepare_data(args: argparse.Namespace, output_dir: Path) -> dict:
    english, categories = load_english_labeled(Path(args.mind_train_dir))
    telugu = load_telugu_articles(Path(args.telugu_dir))
    if args.max_train_samples:
        english = english.sample(n=min(args.max_train_samples, len(english)), random_state=args.seed).reset_index(drop=True)
    if args.max_telugu_samples:
        telugu = telugu.head(args.max_telugu_samples).copy()

    write_category_files(output_dir, categories)
    english.to_pickle(output_dir / "english_mindlarge_refined_train.pkl")
    telugu.to_pickle(output_dir / "telugu_df_full.pkl")
    telugu[["story_id"]].to_csv(output_dir / "telugu_story_ids.csv", index=False)

    category_counts = english["new_category"].value_counts().sort_index()
    category_counts.to_csv(output_dir / "mindlarge_refined_category_counts.csv", header=["count"])

    summary = {
        "mind_train_dir": args.mind_train_dir,
        "telugu_dir": args.telugu_dir,
        "english_rows": int(len(english)),
        "telugu_rows": int(len(telugu)),
        "category_count": len(categories),
        "categories": categories,
        "category_counts": {cat: int(count) for cat, count in category_counts.items()},
        "telugu_source_file_counts": {
            name: int(count) for name, count in telugu["source_file"].value_counts().sort_index().items()
        },
        "outputs": {
            "english_pickle": str(output_dir / "english_mindlarge_refined_train.pkl"),
            "telugu_pickle": str(output_dir / "telugu_df_full.pkl"),
            "story_ids": str(output_dir / "telugu_story_ids.csv"),
            "category_file": str(output_dir / "mindlarge_refined_categories.txt"),
            "label_mapping": str(output_dir / "label_mapping.json"),
        },
    }
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def train_classifier(args: argparse.Namespace, output_dir: Path) -> dict:
    import inspect
    from datasets import Dataset as HFDataset
    import evaluate
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    english_path = output_dir / "english_mindlarge_refined_train.pkl"
    mapping_path = output_dir / "label_mapping.json"
    if not english_path.exists() or not mapping_path.exists():
        prepare_data(args, output_dir)

    english = pd.read_pickle(english_path)
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    categories = mapping["categories"]

    dataset = HFDataset.from_pandas(english[["newid", "text", "label"]].rename(columns={"newid": "id"}))
    split = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_tok = split["train"].map(tokenize, batched=True)
    val_tok = split["test"].map(tokenize, batched=True)
    train_tok = train_tok.rename_column("label", "labels")
    val_tok = val_tok.rename_column("label", "labels")
    remove_columns = [name for name in ["id", "text", "__index_level_0__"] if name in train_tok.column_names]
    train_tok = train_tok.remove_columns(remove_columns)
    val_tok = val_tok.remove_columns([name for name in remove_columns if name in val_tok.column_names])

    accuracy = evaluate.load("accuracy")
    f1_score = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1_score.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(categories),
        id2label={idx: cat for idx, cat in enumerate(categories)},
        label2id={cat: idx for idx, cat in enumerate(categories)},
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    classifier_dir = Path(args.classifier_dir) if args.classifier_dir else output_dir / "xlm_roberta_mindlarge_refined_classifier"

    training_args = TrainingArguments(
        output_dir=str(classifier_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
        seed=args.seed,
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tok,
        "eval_dataset": val_tok,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(classifier_dir)
    tokenizer.save_pretrained(classifier_dir)

    result = {
        "classifier_dir": str(classifier_dir),
        "model_name": args.model_name,
        "train_rows": len(train_tok),
        "validation_rows": len(val_tok),
        "category_count": len(categories),
        "train_result": {key: float(value) for key, value in train_result.metrics.items()},
        "eval_metrics": {key: float(value) for key, value in metrics.items()},
    }
    with (output_dir / "classifier_training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def classify_telugu(args: argparse.Namespace, output_dir: Path) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    telugu_path = output_dir / "telugu_df_full.pkl"
    mapping_path = output_dir / "label_mapping.json"
    if not telugu_path.exists() or not mapping_path.exists():
        prepare_data(args, output_dir)

    classifier_dir = Path(args.classifier_dir) if args.classifier_dir else output_dir / "xlm_roberta_mindlarge_refined_classifier"
    if not classifier_dir.exists():
        raise FileNotFoundError(
            f"Classifier directory not found: {classifier_dir}. Run --mode train first."
        )

    telugu = pd.read_pickle(telugu_path)
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    categories = mapping["categories"]

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(classifier_dir)
    model = AutoModelForSequenceClassification.from_pretrained(classifier_dir).to(device)
    model.eval()

    probs = []
    for start in range(0, len(telugu), args.infer_batch_size):
        batch = telugu["text"].iloc[start : start + args.infer_batch_size].tolist()
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probs.append(batch_probs)
        if (start // args.infer_batch_size + 1) % 100 == 0:
            print(f"  classified {min(start + args.infer_batch_size, len(telugu)):,}/{len(telugu):,}", flush=True)

    all_probs = np.vstack(probs).astype(np.float32)
    top_idx = all_probs.argmax(axis=1)
    pred = pd.DataFrame(
        {
            "story_id": telugu["story_id"],
            "headline": telugu["headline"],
            "source_file": telugu["source_file"],
            "predicted_category": [categories[idx] for idx in top_idx],
            "confidence": all_probs[np.arange(len(all_probs)), top_idx],
        }
    )
    np.save(output_dir / "telugu_category_probs.npy", all_probs)
    pred.to_csv(output_dir / "telugu_category_predictions.csv", index=False)

    summary = {
        "classifier_dir": str(classifier_dir),
        "telugu_rows": int(len(telugu)),
        "category_count": len(categories),
        "probabilities_path": str(output_dir / "telugu_category_probs.npy"),
        "predictions_path": str(output_dir / "telugu_category_predictions.csv"),
        "prediction_category_counts": {
            cat: int(count) for cat, count in pred["predicted_category"].value_counts().sort_index().items()
        },
        "mean_top_confidence": float(pred["confidence"].mean()),
        "median_top_confidence": float(pred["confidence"].median()),
    }
    with (output_dir / "telugu_classification_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def build_telugu_features(args: argparse.Namespace, output_dir: Path) -> dict:
    mapping_path = output_dir / "label_mapping.json"
    probs_path = output_dir / "telugu_category_probs.npy"
    telugu_path = output_dir / "telugu_df_full.pkl"
    if not mapping_path.exists() or not probs_path.exists() or not telugu_path.exists():
        raise FileNotFoundError("Missing label mapping, Telugu dataframe, or category probabilities.")

    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    categories = mapping["categories"]
    probs = np.load(probs_path)
    telugu = pd.read_pickle(telugu_path)
    if probs.shape != (len(telugu), len(categories)):
        raise ValueError(f"Probability shape {probs.shape} does not match Telugu/category shape {(len(telugu), len(categories))}")

    rows, cols = np.nonzero(probs > 1e-6)
    data = probs[rows, cols]
    item_features = csr_matrix((data, (rows, cols)), shape=probs.shape, dtype=np.float32)
    save_npz(output_dir / "telugu_item_features_23cat.npz", item_features)
    with (output_dir / "telugu_story_ids.pkl").open("wb") as f:
        pickle.dump(telugu["story_id"].tolist(), f)

    summary = {
        "telugu_rows": int(len(telugu)),
        "category_count": len(categories),
        "item_features_shape": list(item_features.shape),
        "item_features_nnz": int(item_features.nnz),
        "item_features_path": str(output_dir / "telugu_item_features_23cat.npz"),
        "story_ids_path": str(output_dir / "telugu_story_ids.pkl"),
    }
    with (output_dir / "telugu_item_features_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    if args.mode in {"prepare", "all"}:
        outputs["prepare"] = prepare_data(args, output_dir)
    if args.mode in {"train", "all"}:
        outputs["train"] = train_classifier(args, output_dir)
    if args.mode in {"classify", "all"}:
        outputs["classify"] = classify_telugu(args, output_dir)
    if args.mode in {"build-features", "all"}:
        outputs["build_features"] = build_telugu_features(args, output_dir)

    print(json.dumps(outputs, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
