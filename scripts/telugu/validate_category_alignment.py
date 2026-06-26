from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from process_mindlarge_refined import load_news, make_refined_categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether Telugu article category processing aligns with the final MINDlarge refined LightFM model."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository directory for relative input/output paths.",
    )
    parser.add_argument("--mindlarge-train-dir", default="MINDlarge_train")
    parser.add_argument("--mindlarge-test-dir", default="MINDlarge_test")
    parser.add_argument("--telugu-dir", default="telugu")
    parser.add_argument("--old-newcat-csv", default="newcat.csv")
    parser.add_argument(
        "--final-dataset-path",
        default="submissions/refined_mindlarge_test_train_only/refined_submission_dataset.pkl",
    )
    parser.add_argument("--output-dir", default="results/telugu_category_alignment")
    return parser.parse_args()


def resolve_path(repo_dir: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else repo_dir / path


def load_final_model_categories(dataset_path: Path) -> list[str]:
    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)
    _, _, item_map, item_feature_map = dataset.mapping()
    return sorted(set(item_feature_map) - set(item_map))


def load_refined_categories_from_dirs(dirs: list[Path]) -> list[str]:
    frames = []
    for data_dir in dirs:
        news_path = data_dir / "news.tsv"
        if news_path.exists():
            frames.append(load_news(data_dir))
    if not frames:
        return []
    news = pd.concat(frames, ignore_index=True).drop_duplicates("newid", keep="last")
    return sorted(make_refined_categories(news)["new_category"].dropna().unique())


def inspect_telugu_dir(telugu_dir: Path) -> dict:
    files = sorted(telugu_dir.glob("*.parquet"))
    result = {"parquet_files": [str(path) for path in files], "splits": {}, "total_rows": 0}
    for path in files:
        df = pd.read_parquet(path)
        result["splits"][path.name] = {"rows": int(len(df)), "columns": list(df.columns)}
        result["total_rows"] += int(len(df))
    return result


def load_old_newcat_categories(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "new_category" not in df.columns:
        return []
    return sorted(df["new_category"].dropna().astype(str).unique())


def artifact_status(paths: list[Path]) -> dict[str, dict]:
    status = {}
    for path in paths:
        status[str(path)] = {
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        }
    return status


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    output_dir = resolve_path(repo_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dataset_path = resolve_path(repo_dir, args.final_dataset_path)
    final_model_categories = load_final_model_categories(final_dataset_path)
    mindlarge_categories = load_refined_categories_from_dirs(
        [
            resolve_path(repo_dir, args.mindlarge_train_dir),
            resolve_path(repo_dir, args.mindlarge_test_dir),
        ]
    )
    old_newcat_categories = load_old_newcat_categories(resolve_path(repo_dir, args.old_newcat_csv))
    telugu_info = inspect_telugu_dir(resolve_path(repo_dir, args.telugu_dir))

    missing_from_old = sorted(set(final_model_categories) - set(old_newcat_categories))
    old_not_in_final = sorted(set(old_newcat_categories) - set(final_model_categories))
    final_vs_news_mismatch = {
        "model_not_in_mindlarge_news": sorted(set(final_model_categories) - set(mindlarge_categories)),
        "mindlarge_news_not_in_model": sorted(set(mindlarge_categories) - set(final_model_categories)),
    }

    artifacts = artifact_status(
        [
            Path("telugu_classifier"),
            Path("telugu_classifier_newcat"),
            Path("telugu_all_probs_newcat.npy"),
            Path("telugu_story_ids_newcat.csv"),
            Path("telugu_df_full.pkl"),
            Path("item_features_newcat.npz"),
            Path("lightfm_model_newcat.pkl"),
        ]
    )

    report = {
        "final_dataset_path": str(final_dataset_path),
        "final_model_category_count": len(final_model_categories),
        "final_model_categories": final_model_categories,
        "mindlarge_news_category_count": len(mindlarge_categories),
        "mindlarge_news_categories": mindlarge_categories,
        "old_newcat_csv": args.old_newcat_csv,
        "old_newcat_category_count": len(old_newcat_categories),
        "old_newcat_categories": old_newcat_categories,
        "missing_from_old_newcat": missing_from_old,
        "old_newcat_not_in_final_model": old_not_in_final,
        "final_model_vs_mindlarge_news_mismatch": final_vs_news_mismatch,
        "telugu_data": telugu_info,
        "artifact_status": artifacts,
        "aligned": not missing_from_old and not old_not_in_final and not final_vs_news_mismatch["model_not_in_mindlarge_news"],
        "conclusion": (
            "Telugu category processing is aligned with the final MINDlarge refined model."
            if not missing_from_old and not old_not_in_final
            else "Telugu category processing is not fully aligned with the final MINDlarge refined model."
        ),
    }

    with (output_dir / "telugu_category_alignment_report.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    (output_dir / "final_mindlarge_refined_categories.txt").write_text(
        "\n".join(final_model_categories) + "\n",
        encoding="utf-8",
    )
    (output_dir / "old_newcat_categories.txt").write_text(
        "\n".join(old_newcat_categories) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
