# Thesis Reproduction: English-Telugu News Recommendation

This repo reproduces the LightFM-based experiment used in the thesis:

- Train an English news recommender on MIND-small.
- Build refined 19-category item features from MIND metadata.
- Train an XLM-R category classifier on English MIND titles/abstracts.
- Apply that classifier to Telugu news articles.
- Recommend Telugu articles to English users with the trained LightFM model.

The notebooks were originally run on macOS with local data and model artifacts ignored by git. The committed notebooks now use repo-relative paths.

## 1. Environment

Use Python 3.11 on macOS.

```bash
cd /path/to/Recommendation_System-

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install "numpy<2" pandas scipy scikit-learn matplotlib seaborn tqdm pyarrow nltk
pip install torch transformers datasets evaluate accelerate
pip install lightfm
pip install jupyterlab notebook ipykernel
```

Register the Jupyter kernel:

```bash
python -m ipykernel install --user --name recsys-thesis --display-name "Python (recsys-thesis)"
```

Verify:

```bash
python - <<'PY'
import numpy, pandas, scipy, sklearn, torch, transformers, datasets, evaluate, lightfm, pyarrow
print("env ok")
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("sklearn", sklearn.__version__)
print("lightfm", lightfm.__version__)
PY
```

Start Jupyter:

```bash
jupyter lab
```

Select kernel: `Python (recsys-thesis)`.

## 2. Data Setup

The following paths are intentionally ignored by git:

- `MINDsmall_train/`
- `MINDsmall_dev/`
- `telugu/`
- generated `*.csv`, `*.pkl`, `*.npy`, `*.npz`, `*.pt`
- `telugu_classifier_newcat/`

### MIND-small

Use the Hugging Face mirror because the old Microsoft Azure blob may return `PublicAccessNotPermitted`.

```bash
curl -L -o MINDsmall_train.zip \
  https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_train.zip

curl -L -o MINDsmall_dev.zip \
  https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_dev.zip

mkdir -p MINDsmall_train MINDsmall_dev
unzip -o MINDsmall_train.zip -d MINDsmall_train
unzip -o MINDsmall_dev.zip -d MINDsmall_dev
```

Expected files:

```text
MINDsmall_train/news.tsv
MINDsmall_train/behaviors.tsv
MINDsmall_train/entity_embedding.vec
MINDsmall_train/relation_embedding.vec

MINDsmall_dev/news.tsv
MINDsmall_dev/behaviors.tsv
MINDsmall_dev/entity_embedding.vec
MINDsmall_dev/relation_embedding.vec
```

### Telugu Dataset

Dataset link:

```text
https://huggingface.co/datasets/saidines12/telugu_news_dataset
```

Download the parquet files:

```bash
mkdir -p telugu

curl -L -o telugu/train-00000-of-00001.parquet \
  https://huggingface.co/datasets/saidines12/telugu_news_dataset/resolve/main/data/train-00000-of-00001.parquet

curl -L -o telugu/val-00000-of-00001.parquet \
  https://huggingface.co/datasets/saidines12/telugu_news_dataset/resolve/main/data/val-00000-of-00001.parquet

curl -L -o telugu/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/saidines12/telugu_news_dataset/resolve/main/data/test-00000-of-00001.parquet
```

Expected files:

```text
telugu/train-00000-of-00001.parquet
telugu/val-00000-of-00001.parquet
telugu/test-00000-of-00001.parquet
```

## 3. Notebook Run Order

Run only the LightFM/refined-category pipeline for the thesis results. `nrms_MIND.ipynb` is a separate demo-style NRMS experiment and is not part of the main thesis rerun.

### Step 1: English Refined-Category LightFM

Notebook:

```text
lightfm_VerticalFeature-subvertical.ipynb
```

Run the cells that:

1. Build interactions from:
   - `MINDsmall_train/behaviors.tsv`
   - `MINDsmall_dev/behaviors.tsv`
2. Save:
   - `train_interactions.csv`
   - `val_interactions.csv`
3. Load:
   - `MINDsmall_train/news.tsv`
4. Create and refine `new_category`.
5. Save:
   - `newcat.csv`
   - `item_features_newcat.npz`
6. Train `model1 = LightFM(no_components=30, loss="warp")` for 20 epochs.
7. Evaluate and save:
   - `lightfm_model_newcat.pkl`

Expected reference output from the original run:

```text
Train Precision@10: 0.0872    |  Val Precision@10: 0.0735
Train Recall@10:    0.0320    |  Val Recall@10:    0.0282
Train AUC:          0.9844    |  Val AUC:          0.8727
Saved LightFM model to lightfm_model_newcat.pkl
```

Minor drift is acceptable because LightFM training is stochastic.

### Step 2: Telugu Category Classification and Feature Matrix

Notebook:

```text
TeluguArticleProcessing-newcat.ipynb
```

Prerequisites from Step 1:

```text
newcat.csv
lightfm_model_newcat.pkl
```

Run cells that:

1. Load `newcat.csv`.
2. Train XLM-RoBERTa on English MIND text to predict the 19 refined categories.
3. Save:
   - `telugu_classifier_newcat/`
4. Load all Telugu parquet files from `telugu/`.
5. Deduplicate by `story_id`.
6. Save:
   - `telugu_df_full.pkl`
7. Predict Telugu category probabilities.
8. Save:
   - `telugu_all_probs_newcat.npy`
   - `telugu_story_ids_newcat.csv`
9. Build Telugu LightFM item features with shape `(46816, 19)`.

Expected reference checks:

```text
Loaded telugu_df with 46816 rows
all_probs ready; shape = (46816, 19)
CSR shape: (46816, 19)
```

The XLM-R training cell is expensive. Use GPU/MPS if available.

### Step 3: Telugu Recommendation Analysis

Notebook:

```text
Untitled1.ipynb
```

Prerequisites:

```text
lightfm_model_newcat.pkl
telugu_df_full.pkl
telugu_all_probs_newcat.npy
telugu_story_ids_newcat.csv
```

Run cells that:

1. Load Telugu dataframe and probability matrix.
2. Rebuild `item_features_telugu`.
3. Load `lightfm_model_newcat.pkl`.
4. Assign `assigned_category`.
5. Run `topk_telugu(...)` for representative users.
6. Generate tables and diversity metrics.

Expected reference checks:

```text
Representative user IDs: [12, 30, 5]
Found 94057 users with >0 Telugu score.
```

The thesis qualitative examples are based on users:

```text
12, 30, 5
```

## 4. Expected Generated Artifacts

After a successful full rerun, these ignored files/folders should exist:

```text
train_interactions.csv
val_interactions.csv
newcat.csv
item_features_newcat.npz
lightfm_model_newcat.pkl
telugu_classifier_newcat/
telugu_df_full.pkl
telugu_all_probs_newcat.npy
telugu_story_ids_newcat.csv
```

Optional embedding/inheritance notebooks may create:

```text
english_article_embeddings.pt
english_article_embeddings_val.pt
english_news_ids.pkl
english_news_ids_val.pkl
telugu_article_embeddings.pt
telugu_news_ids.pkl
inherited_telugu_interactions.csv
```

These are not required for the main refined-category thesis rerun unless you are reproducing the inheritance/embedding experiments.

## 5. Notes and Caveats

- The Windows rerun hit a native LightFM WARP-extension failure. macOS is the recommended environment because the original notebooks were produced there and LightFM generally builds more cleanly.
- LightFM may print `compiled without OpenMP support`; that is acceptable, but training will be slower.
- Do not change model hyperparameters if the goal is to verify the thesis values.
- Do not run `nrms_MIND.ipynb` for this verification unless you separately want the NRMS demo. It uses `recommenders` and TensorFlow dependencies that are not needed for the LightFM thesis pipeline.
- The original official Microsoft MIND Azure URLs may no longer be public; use the Hugging Face mirror listed above.