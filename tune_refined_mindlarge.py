from __future__ import annotations

import argparse
import csv
import json
import pickle
from itertools import product
from pathlib import Path

import pandas as pd
from lightfm import LightFM

from evaluate_mindlarge_dev import evaluate_labeled_impressions
from run_mindlarge_train_metrics import (
    build_category_features,
    fit_dataset,
    load_news,
    make_refined_categories,
    scan_behaviors,
)


def parse_csv_values(raw: str, cast):
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the refined-category LightFM model on MINDlarge_train -> MINDlarge_dev."
    )
    parser.add_argument("--train-dir", default="MINDlarge_train")
    parser.add_argument("--eval-dir", default="MINDlarge_dev")
    parser.add_argument("--output-dir", default="results/mindlarge_tuning")
    parser.add_argument("--losses", default="warp,bpr")
    parser.add_argument("--components", default="30,50")
    parser.add_argument("--learning-rates", default="0.03")
    parser.add_argument("--epochs", default="5")
    parser.add_argument("--item-alphas", default="0")
    parser.add_argument("--user-alphas", default="0")
    parser.add_argument("--eval-limit", type=int, default=50000)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def config_id(config: dict) -> str:
    return (
        f"loss-{config['loss']}"
        f"_comp-{config['no_components']}"
        f"_lr-{config['learning_rate']}"
        f"_epochs-{config['epochs']}"
        f"_ia-{config['item_alpha']}"
        f"_ua-{config['user_alpha']}"
    ).replace(".", "p")


def write_result(csv_path: Path, row: dict) -> None:
    fieldnames = [
        "config_id",
        "loss",
        "no_components",
        "learning_rate",
        "epochs",
        "item_alpha",
        "user_alpha",
        "AUC",
        "MRR",
        "nDCG@5",
        "nDCG@10",
        "groups_seen",
        "groups_evaluated",
        "groups_skipped_unknown_user",
        "groups_skipped_no_known_items",
        "groups_skipped_single_class",
        "known_candidate_item_ratio",
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
    output_dir = Path(args.output_dir)
    model_output_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        model_output_dir.mkdir(parents=True, exist_ok=True)

    losses = parse_csv_values(args.losses, str)
    components = parse_csv_values(args.components, int)
    learning_rates = parse_csv_values(args.learning_rates, float)
    epochs_list = parse_csv_values(args.epochs, int)
    item_alphas = parse_csv_values(args.item_alphas, float)
    user_alphas = parse_csv_values(args.user_alphas, float)

    print("Loading MINDlarge train data", flush=True)
    news = load_news(train_dir)
    refined_news = make_refined_categories(news)
    behaviors_path = train_dir / "behaviors.tsv"
    users, items, scan_stats = scan_behaviors(behaviors_path)
    all_users = sorted(users)
    all_items = sorted(items | set(news["newid"]))
    feature_names = sorted(refined_news["new_category"].dropna().unique())
    print(
        f"users={len(all_users):,} items={len(all_items):,} "
        f"history_clicks={scan_stats['history_clicks']:,} "
        f"positive_impressions={scan_stats['positive_impressions']:,}",
        flush=True,
    )

    print("Building refined train matrix and item features", flush=True)
    dataset, interaction_matrix = fit_dataset(behaviors_path, all_users, all_items, feature_names)
    item_features = build_category_features(dataset, refined_news, "new_category")
    print(
        f"interaction_matrix={interaction_matrix.shape} item_features={item_features.shape}",
        flush=True,
    )

    csv_path = output_dir / "refined_tuning_results.csv"
    jsonl_path = output_dir / "refined_tuning_results.jsonl"
    configs = [
        {
            "loss": loss,
            "no_components": no_components,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "item_alpha": item_alpha,
            "user_alpha": user_alpha,
        }
        for loss, no_components, learning_rate, epochs, item_alpha, user_alpha in product(
            losses,
            components,
            learning_rates,
            epochs_list,
            item_alphas,
            user_alphas,
        )
    ]
    print(f"Running {len(configs)} refined tuning configs", flush=True)

    for index, config in enumerate(configs, start=1):
        run_id = config_id(config)
        print(f"\n=== [{index}/{len(configs)}] {run_id} ===", flush=True)
        model = LightFM(
            no_components=config["no_components"],
            loss=config["loss"],
            learning_rate=config["learning_rate"],
            item_alpha=config["item_alpha"],
            user_alpha=config["user_alpha"],
        )
        model.fit(
            interaction_matrix,
            item_features=item_features,
            epochs=config["epochs"],
            num_threads=args.num_threads,
            verbose=True,
        )

        metrics = evaluate_labeled_impressions(
            model,
            dataset,
            eval_dir / "behaviors.tsv",
            item_features,
            args.eval_limit,
        )
        row = {"config_id": run_id, **config, **metrics}
        write_result(csv_path, row)
        with jsonl_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if args.save_models:
            with (model_output_dir / f"{run_id}.pkl").open("wb") as f:
                pickle.dump(model, f)
        print(json.dumps(row, indent=2), flush=True)

    results = pd.read_csv(csv_path)
    best = results.sort_values(["nDCG@10", "MRR", "AUC"], ascending=False).iloc[0]
    print("\nBest config by nDCG@10")
    print(best.to_string())


if __name__ == "__main__":
    main()
