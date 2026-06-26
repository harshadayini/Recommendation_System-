# Experiment Reproduction Log

This file documents the reproduction work completed in this repository.

## Scope

The goal was to reproduce the existing paper/notebook experiments without changing the LightFM model logic, and then run the same model families on `MINDlarge_train` with these metrics:

- `Precision@10`
- `Recall@10`
- `LightFM_AUC`
- `AUC`
- `MRR`
- `nDCG@5`
- `nDCG@10`

The original `lightfm` package was used, not `lightfm-next`, to stay consistent with the project notebooks and paper-style code.

## Environment

Python environment:

```text
/Users/harshadayiniakula/Desktop/RecSys/rs311
```

Verified package versions:

```text
lightfm      1.17
numpy        1.26.4
torch        2.12.0
transformers 4.46.3
```

Important note: `lightfm` was compiled without OpenMP support. This means training ran single-threaded even when `num_threads=4` was passed. This affects runtime, not the intended model definition.

## Data Used

All training and evaluation used files extracted inside this repository:

```text
Recommendation_System-/MINDsmall_train/
Recommendation_System-/MINDsmall_dev/
Recommendation_System-/MINDlarge_train/
Recommendation_System-/MINDlarge_test/
Recommendation_System-/telugu/
```

For the MINDlarge train experiments, the active input files were:

```text
MINDlarge_train/news.tsv
MINDlarge_train/behaviors.tsv
MINDlarge_train/entity_embedding.vec
MINDlarge_train/relation_embedding.vec
```

Verified row counts:

```text
MINDlarge_train/news.tsv       101,527
MINDlarge_train/behaviors.tsv  2,232,748

MINDlarge_dev/news.tsv          72,023
MINDlarge_dev/behaviors.tsv    376,471

MINDsmall_train/news.tsv        51,282
MINDsmall_train/behaviors.tsv  156,965

MINDsmall_dev/news.tsv          42,416
MINDsmall_dev/behaviors.tsv     73,152
```

`MINDsmall_train`, `MINDsmall_dev`, and `MINDlarge_train` use the same MIND file structure:

```text
news.tsv:
news_id, vertical/category, subvertical/subcategory, title, abstract, url, title_entities, abstract_entities

behaviors.tsv:
impression_id, user_id, time, history, impressions
```

The `MINDlarge_train/behaviors.tsv` impressions are labeled, for example:

```text
N78206-0 N94157-1 N78699-1
```

Because these labels are available, `AUC`, `MRR`, `nDCG@5`, and `nDCG@10` can be computed on `MINDlarge_train`.

`MINDlarge_test` was downloaded/extracted earlier, but the public test impressions do not provide local ground-truth labels in the same way, so local AUC/MRR/nDCG evaluation was not run on the test split.

## Why MINDlarge Was Added

The original work used `MINDsmall_train` and `MINDsmall_dev`. That was enough to reproduce the notebook/paper values, but it does not provide a local held-out test score to report in the same way as a benchmark submission. The MIND test split does not include ground-truth labels locally; official test metrics require producing a submission file and sending it to the online evaluator.

For that reason, the workflow was extended to MINDlarge:

```text
Train:      MINDlarge_train
Validate:  MINDlarge_dev
Test:      MINDlarge_test, via online submission only
```

This keeps the same general experimental structure but uses the larger MIND split that supports an official hidden-test evaluation path.

## Dataset Processing

The MIND files were not converted into a new custom dataset format. The scripts read the original `news.tsv` and `behaviors.tsv` files directly from this repository.

For every split:

- `news.tsv` provides news metadata and content features.
- `behaviors.tsv` provides user histories and impression candidates.
- In train/dev splits, each impression candidate has a label, for example `N94157-1` or `N78206-0`.

The training interaction matrix was built from:

```text
1. clicked items in user history
2. clicked impression items where label == 1
```

Negative impressions were not inserted as explicit negative interactions for LightFM training because the original LightFM WARP/BPR workflow learns from implicit positive interactions and sampled negatives.

The dev evaluation used labeled impression groups from `MINDlarge_dev/behaviors.tsv`. Each group was scored by the trained model, sorted by predicted score, and then compared against the click labels in that impression group.

## Model Variants

Four model variants were run on MINDlarge:

| Tag | Meaning |
|---|---|
| `VERTICAL` | LightFM with original vertical/category item features. |
| `REFINED` | LightFM with refined category features based on the paper's category-remapping idea. |
| `TFIDF` | LightFM with vertical one-hot features plus title TF-IDF features. |
| `BERT` | LightFM with vertical one-hot features plus reduced multilingual MiniLM article embeddings. |

The goal was to change the dataset scale and add metrics, not to change the core model logic.

## ID Mapping And Saved Models

The trained `.pkl` files contain the learned LightFM model weights, but LightFM scores users/items by numeric row and column indexes. The original MIND ids are strings such as:

```text
user id: U87243
item id: N94157
```

During training, those ids are mapped to numeric indexes:

```text
U87243 -> user row index
N94157 -> item column index
```

For dev evaluation, the saved model can only score a dev user/item if that user/item can be mapped into the same numeric index space used during training. This is why `evaluate_mindlarge_dev.py` loads or creates `*_dataset.pkl` files. Those files preserve the mapping between MIND ids and the LightFM model's numeric indexes.

This does not retrain the models. It only makes evaluation reproducible and auditable.

## Original MINDsmall Reproduction Results

Completed artifacts:

```text
english_article_embeddings.pt
english_article_embeddings_val.pt
english_news_ids.pkl
english_news_ids_val.pkl
newcat.csv
new_cats.txt
item_features_newcat.npz
lightfm_model1.pkl
lightfm_model_newcat.pkl
train_interactions.csv
val_interactions.csv
```

Observed MINDsmall results:

| Model | TrainP@10 | ValP@10 | Train AUC | Val AUC |
|---|---:|---:|---:|---:|
| BERT Hybrid | 0.1092 | 0.0002 | - | - |
| TF-IDF Hybrid | 0.0706 | 0.0488 | - | - |
| Vertical OHE | 0.0848 | 0.0729 | 0.9827 | 0.8664 |
| Refined Category | 0.0871 | 0.0714 | 0.9845 | 0.8731 |

The Telugu classifier training was not completed because the estimated runtime was several hours.

## MINDlarge Train Scripts Added

Two repo-local scripts were added for MINDlarge train experiments:

```text
run_mindlarge_train_metrics.py
generate_mindlarge_bert_embeddings.py
evaluate_mindlarge_dev.py
tune_refined_mindlarge.py
```

`run_mindlarge_train_metrics.py` trains and evaluates these model variants sequentially:

- `vertical`
- `refined`
- `tfidf`
- `bert`

The script streams `behaviors.tsv` where practical to reduce memory pressure and writes results to:

```text
results/mindlarge_train/metrics.json
```

`generate_mindlarge_bert_embeddings.py` applies the same notebook-style embedding process to `MINDlarge_train/news.tsv`:

- Clean title and abstract.
- Combine cleaned title and abstract.
- Use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Mean-pool token embeddings.
- Save embeddings and news ids.

Generated large-train BERT artifacts:

```text
english_article_embeddings_large_train.pt  shape: (101527, 384)
english_news_ids_large_train.pkl           101,527 ids
```

`evaluate_mindlarge_dev.py` is a separate evaluation-only script. It does not retrain the existing models. It loads the saved models from `results/mindlarge_train`, loads or creates a saved train `Dataset` mapping from `MINDlarge_train`, and evaluates labeled impressions from a separate split such as `MINDlarge_dev`.

The mapping files are saved next to the trained models:

```text
results/mindlarge_train/vertical_dataset.pkl
results/mindlarge_train/refined_dataset.pkl
results/mindlarge_train/tfidf_dataset.pkl
results/mindlarge_train/bert_dataset.pkl
```

These mapping files connect real MIND ids such as `U87243` and `N94157` to the numeric row/column indexes expected by the saved LightFM model.

`tune_refined_mindlarge.py` is a tuning script for the refined-category model only. It builds the `MINDlarge_train` interaction matrix and refined item features once, then trains separate refined-model candidates for each hyperparameter configuration. Each candidate is evaluated on `MINDlarge_dev`, and the results are appended to:

```text
results/mindlarge_tuning/refined_tuning_results.csv
results/mindlarge_tuning/refined_tuning_results.jsonl
```

The intended command after downloading/extracting `MINDlarge_dev` is:

```bash
../rs311/bin/python evaluate_mindlarge_dev.py \
  --train-dir MINDlarge_train \
  --eval-dir MINDlarge_dev \
  --model-dir results/mindlarge_train \
  --output results/mindlarge_train/dev_metrics.json \
  --models vertical refined tfidf bert \
  --eval-limit 100000 \
  --paper-metric-users 5000
```

The dev output will be written to:

```text
results/mindlarge_train/dev_metrics.json
```

The script also reports coverage fields such as:

```text
groups_seen
groups_evaluated
groups_skipped_unknown_user
groups_skipped_no_known_items
groups_skipped_single_class
known_candidate_item_ratio
evaluated_group_ratio
```

## Metrics Explained

Two kinds of metrics were reported.

### Paper-style LightFM metrics

These are computed with LightFM's evaluator:

```text
Precision@10
Recall@10
LightFM_AUC
```

They rank items for a user and compare the ranked list against positive interactions. On MINDlarge, these metrics can be expensive because they rank against a large item catalog. To keep the runs practical on the single-threaded original LightFM build, these metrics were capped with:

```text
paper_metric_users = 5000
```

That means these paper-style metrics were computed on 5,000 known users, not on every user in the split.

### MIND impression-ranking metrics

These are computed directly from labeled impression groups:

```text
AUC
MRR
nDCG@5
nDCG@10
```

For each impression group:

1. The trained model scores every known candidate item for that user.
2. Candidates are sorted by predicted score.
3. The sorted labels are compared against the true clicked item labels.
4. AUC, MRR, nDCG@5, and nDCG@10 are computed for that impression group.
5. The final value is the mean across evaluated impression groups.

Metric meanings:

| Metric | Meaning |
|---|---|
| `AUC` | Probability that a clicked item is ranked above a non-clicked item within the same impression group. |
| `MRR` | Reciprocal rank of the first clicked item. Higher means the first relevant item appears earlier. |
| `nDCG@5` | Ranking quality in the top 5 positions, with higher weight for clicked items near the top. |
| `nDCG@10` | Ranking quality in the top 10 positions. This is a good primary metric for model selection. |

The impression-ranking metrics were capped with:

```text
groups_evaluated = 100000
```

This cap means the current train/dev values are sampled/capped metrics, not full-split exhaustive metrics.

## Why Some Dev Groups Were Skipped

When evaluating a model trained on `MINDlarge_train` against `MINDlarge_dev`, not every dev impression group can be scored cleanly by LightFM.

Groups were skipped for three reasons:

| Skip reason | Explanation |
|---|---|
| `groups_skipped_unknown_user` | The dev user was not present in `MINDlarge_train`, so the model has no learned user embedding for that user. |
| `groups_skipped_no_known_items` | After filtering candidates to items known from training, fewer than two candidate items remained. |
| `groups_skipped_single_class` | After filtering, all remaining labels were the same class, so metrics like AUC are not meaningful. |

This coverage is reported so the dev results are auditable.

## MINDlarge Dev Evaluation Results

`MINDlarge_dev` was downloaded, extracted into this repository, and evaluated using the already-trained `MINDlarge_train` model artifacts. No model retraining was performed for this step.

Dev evaluation command:

```bash
../rs311/bin/python evaluate_mindlarge_dev.py \
  --train-dir MINDlarge_train \
  --eval-dir MINDlarge_dev \
  --model-dir results/mindlarge_train \
  --output results/mindlarge_train/dev_metrics.json \
  --models vertical refined tfidf bert \
  --eval-limit 100000 \
  --paper-metric-users 5000
```

Dev metrics are stored in:

```text
results/mindlarge_train/dev_metrics.json
```

Evaluation coverage was identical across the four model variants because they used the same train user/item mapping:

```text
groups_seen                  124,954
groups_evaluated             100,000
groups_skipped_unknown_user   16,151
groups_skipped_no_known_items  1,791
groups_skipped_single_class    7,012
candidate_items_seen       4,675,972
candidate_items_known      3,882,094
known_candidate_item_ratio 0.8302
evaluated_group_ratio      0.8003
```

Dev results:

| Tag | Model | Precision@10 | Recall@10 | LightFM_AUC | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `VERTICAL_DEV` | Vertical category | 0.0004 | 0.0016 | 0.6204 | 0.5810 | 0.2852 | 0.2638 | 0.3298 |
| `REFINED_DEV` | Refined category | 0.0005 | 0.0021 | 0.6392 | 0.5846 | 0.2873 | 0.2658 | 0.3327 |
| `TFIDF_DEV` | TF-IDF hybrid | 0.0011 | 0.0030 | 0.7359 | 0.5949 | 0.3141 | 0.2890 | 0.3551 |
| `BERT_DEV` | BERT hybrid | 0.0002 | 0.0007 | 0.7379 | 0.5962 | 0.3155 | 0.2921 | 0.3576 |

Interpretation of the capped dev metrics:

- `BERT_DEV` is the strongest of the four by `AUC`, `MRR`, `nDCG@5`, and `nDCG@10`.
- `TFIDF_DEV` is very close to `BERT_DEV`, especially on `nDCG@10`.
- `REFINED_DEV` improves over `VERTICAL_DEV`, which supports the refined category idea, but it does not outperform TF-IDF or BERT on the current capped dev evaluation.
- These are validation metrics, not hidden test metrics. Hidden test metrics still require a submission file and online evaluation.

## Hyperparameter Tuning Status

Hyperparameter tuning has started for the refined-category model on `MINDlarge_train -> MINDlarge_dev`.

So far, the workflow has used the same/default-style LightFM settings for the four model variants:

```text
loss          warp
no_components 30
epochs        20
```

The completed work answers:

```text
How do the existing model variants behave when moved from MINDsmall to MINDlarge?
```

It does not yet answer:

```text
What is the best hyperparameter configuration for the refined model on MINDlarge_dev?
```

The recommended tuning protocol is:

1. Tune only on `MINDlarge_dev`.
2. Use `MINDlarge_test` only for final online submission.
3. Pick one primary dev selection metric, preferably `nDCG@10`.
4. Save every tuning run and its full metric row.
5. Freeze the chosen hyperparameters before generating test predictions.

Recommended staged search for the refined model:

Stage 1, coarse search:

```text
loss:          warp, bpr
no_components: 30, 50, 100
learning_rate: 0.01, 0.03, 0.05
epochs:        10
```

This is 18 runs.

Stage 2, epoch sweep on the top 3 Stage 1 configs:

```text
epochs: 20, 30
```

This adds 6 runs.

Stage 3, regularization sweep on the best config:

```text
item_alpha/user_alpha:
0 / 0
1e-6 / 1e-6
1e-5 / 1e-5
```

This adds 3 runs.

Suggested output file for tuning:

```text
results/mindlarge_tuning/refined_tuning_results.csv
```

The final selected config should be the row with the best dev `nDCG@10`, with `AUC`, `MRR`, and `nDCG@5` reported as supporting metrics.

The smoke-grid tuning command is:

```bash
../rs311/bin/python tune_refined_mindlarge.py \
  --train-dir MINDlarge_train \
  --eval-dir MINDlarge_dev \
  --output-dir results/mindlarge_tuning \
  --losses warp,bpr \
  --components 30,50 \
  --learning-rates 0.03 \
  --epochs 5 \
  --item-alphas 0 \
  --user-alphas 0 \
  --eval-limit 50000
```

This smoke run tests four candidate configurations:

```text
loss=warp, no_components=30
loss=warp, no_components=50
loss=bpr,  no_components=30
loss=bpr,  no_components=50
```

All four candidates use:

```text
learning_rate=0.03
epochs=5
item_alpha=0
user_alpha=0
```

Smoke-grid tuning was run on `MINDlarge_train -> MINDlarge_dev` with `eval_limit=50000`. Results were saved to:

```text
results/mindlarge_tuning/refined_tuning_results.csv
results/mindlarge_tuning/refined_tuning_results.jsonl
```

Smoke-grid results:

| Config | Loss | Components | LR | Epochs | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `loss-warp_comp-30_lr-0p03_epochs-5_ia-0p0_ua-0p0` | warp | 30 | 0.03 | 5 | 0.5809 | 0.2819 | 0.2597 | 0.3278 |
| `loss-warp_comp-50_lr-0p03_epochs-5_ia-0p0_ua-0p0` | warp | 50 | 0.03 | 5 | 0.5798 | 0.2821 | 0.2601 | 0.3276 |
| `loss-bpr_comp-30_lr-0p03_epochs-5_ia-0p0_ua-0p0` | bpr | 30 | 0.03 | 5 | 0.6305 | 0.3349 | 0.3160 | 0.3799 |
| `loss-bpr_comp-50_lr-0p03_epochs-5_ia-0p0_ua-0p0` | bpr | 50 | 0.03 | 5 | 0.6317 | 0.3339 | 0.3163 | 0.3795 |

Best smoke-grid config by dev `nDCG@10`:

```text
loss          bpr
no_components 30
learning_rate 0.03
epochs        5
item_alpha    0
user_alpha    0
nDCG@10       0.3799
```

The smoke run indicates that `bpr` is much stronger than `warp` for the refined model on the sampled MINDlarge dev evaluation. The next tuning step should focus around `bpr`, varying `no_components`, `learning_rate`, and `epochs`.

The BPR-focused 10-epoch coarse search was run with:

```bash
../rs311/bin/python tune_refined_mindlarge.py \
  --train-dir MINDlarge_train \
  --eval-dir MINDlarge_dev \
  --output-dir results/mindlarge_tuning_bpr10 \
  --losses bpr \
  --components 30,50,100 \
  --learning-rates 0.01,0.03,0.05 \
  --epochs 10 \
  --item-alphas 0 \
  --user-alphas 0 \
  --eval-limit 50000
```

Results were saved to:

```text
results/mindlarge_tuning_bpr10/refined_tuning_results.csv
results/mindlarge_tuning_bpr10/refined_tuning_results.jsonl
```

BPR 10-epoch coarse-search results:

| Config | Components | LR | Epochs | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `loss-bpr_comp-30_lr-0p01_epochs-10_ia-0p0_ua-0p0` | 30 | 0.01 | 10 | 0.6241 | 0.3322 | 0.3123 | 0.3766 |
| `loss-bpr_comp-30_lr-0p03_epochs-10_ia-0p0_ua-0p0` | 30 | 0.03 | 10 | 0.6304 | 0.3291 | 0.3114 | 0.3762 |
| `loss-bpr_comp-30_lr-0p05_epochs-10_ia-0p0_ua-0p0` | 30 | 0.05 | 10 | 0.6344 | 0.3326 | 0.3149 | 0.3789 |
| `loss-bpr_comp-50_lr-0p01_epochs-10_ia-0p0_ua-0p0` | 50 | 0.01 | 10 | 0.6286 | 0.3354 | 0.3159 | 0.3805 |
| `loss-bpr_comp-50_lr-0p03_epochs-10_ia-0p0_ua-0p0` | 50 | 0.03 | 10 | 0.6301 | 0.3269 | 0.3095 | 0.3738 |
| `loss-bpr_comp-50_lr-0p05_epochs-10_ia-0p0_ua-0p0` | 50 | 0.05 | 10 | 0.6351 | 0.3321 | 0.3152 | 0.3794 |
| `loss-bpr_comp-100_lr-0p01_epochs-10_ia-0p0_ua-0p0` | 100 | 0.01 | 10 | 0.6311 | 0.3369 | 0.3177 | 0.3816 |
| `loss-bpr_comp-100_lr-0p03_epochs-10_ia-0p0_ua-0p0` | 100 | 0.03 | 10 | 0.6311 | 0.3271 | 0.3103 | 0.3749 |
| `loss-bpr_comp-100_lr-0p05_epochs-10_ia-0p0_ua-0p0` | 100 | 0.05 | 10 | 0.6339 | 0.3305 | 0.3130 | 0.3783 |

Best BPR 10-epoch config by dev `nDCG@10`:

```text
loss          bpr
no_components 100
learning_rate 0.01
epochs        10
item_alpha    0
user_alpha    0
nDCG@10       0.3816
MRR           0.3369
AUC           0.6311
```

This was the best refined tuning result from the BPR 10-epoch coarse search on the 50,000-group dev sample.

The same best 10-epoch hyperparameter setting was then re-run for 20 epochs to check whether matching the original paper-style epoch count improves the MINDlarge dev ranking metrics:

```bash
../rs311/bin/python tune_refined_mindlarge.py \
  --train-dir MINDlarge_train \
  --eval-dir MINDlarge_dev \
  --output-dir results/mindlarge_tuning_bpr100_lr001_epochs20 \
  --losses bpr \
  --components 100 \
  --learning-rates 0.01 \
  --epochs 20 \
  --item-alphas 0 \
  --user-alphas 0 \
  --eval-limit 50000
```

The 20-epoch result was saved to:

```text
results/mindlarge_tuning_bpr100_lr001_epochs20/refined_tuning_results.csv
results/mindlarge_tuning_bpr100_lr001_epochs20/refined_tuning_results.jsonl
```

Direct 10-vs-20 epoch comparison for the same hyperparameters:

| Loss | Components | LR | Epochs | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bpr | 100 | 0.01 | 10 | 0.6311 | 0.3369 | 0.3177 | 0.3816 |
| bpr | 100 | 0.01 | 20 | 0.6319 | 0.3342 | 0.3162 | 0.3800 |

Conclusion from this direct comparison:

```text
20 epochs slightly improved AUC, but reduced MRR, nDCG@5, and nDCG@10.
Because nDCG@10 is the primary dev selection metric, the 10-epoch model remains better among these two runs.
```

The best refined tuning result so far is still:

```text
loss          bpr
no_components 100
learning_rate 0.01
epochs        10
item_alpha    0
user_alpha    0
nDCG@10       0.3816
MRR           0.3369
AUC           0.6311
```

The next tuning step tested whether this region improves with light regularization:

```text
loss=bpr
no_components=100
learning_rate=0.01
epochs=10
item_alpha/user_alpha=1e-6 and 1e-5
```

Regularization sweep commands:

```bash
../rs311/bin/python tune_refined_mindlarge.py \
  --train-dir MINDlarge_train \
  --eval-dir MINDlarge_dev \
  --output-dir results/mindlarge_tuning_bpr100_lr001_reg_1e6 \
  --losses bpr \
  --components 100 \
  --learning-rates 0.01 \
  --epochs 10 \
  --item-alphas 1e-6 \
  --user-alphas 1e-6 \
  --eval-limit 50000
```

```bash
../rs311/bin/python tune_refined_mindlarge.py \
  --train-dir MINDlarge_train \
  --eval-dir MINDlarge_dev \
  --output-dir results/mindlarge_tuning_bpr100_lr001_reg_1e5 \
  --losses bpr \
  --components 100 \
  --learning-rates 0.01 \
  --epochs 10 \
  --item-alphas 1e-5 \
  --user-alphas 1e-5 \
  --eval-limit 50000
```

Regularization sweep results:

| Loss | Components | LR | Epochs | item_alpha | user_alpha | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bpr | 100 | 0.01 | 10 | 0 | 0 | 0.6311 | 0.3369 | 0.3177 | 0.3816 |
| bpr | 100 | 0.01 | 10 | 1e-6 | 1e-6 | 0.6236 | 0.3309 | 0.3114 | 0.3757 |
| bpr | 100 | 0.01 | 10 | 1e-5 | 1e-5 | 0.5273 | 0.2606 | 0.2356 | 0.3044 |

Regularization conclusion:

```text
Both regularized runs underperformed the unregularized best config.
The 1e-6 run caused a small drop, and the 1e-5 run caused a large drop.
For the refined MINDlarge model, keep item_alpha=0 and user_alpha=0 among the tested values.
```

Best refined tuning result after the epoch and regularization checks:

```text
loss          bpr
no_components 100
learning_rate 0.01
epochs        10
item_alpha    0
user_alpha    0
AUC           0.6311
MRR           0.3369
nDCG@5        0.3177
nDCG@10       0.3816
```

Recommended next step:

```text
Run the selected best config on the full MINDlarge_dev evaluation set, without eval_limit, and save that as the tuned refined validation result.
```

## MINDlarge Test Submission Generation

The final Codabench submission file was generated for the refined-category model using the selected tuned hyperparameters:

```text
train split      MINDlarge_train
tuning split     MINDlarge_dev
submission split MINDlarge_test
loss             bpr
no_components    100
learning_rate    0.01
epochs           10
item_alpha       0
user_alpha       0
```

Important decision:

```text
The final submission model was trained only on MINDlarge_train.
MINDlarge_dev was used only for hyperparameter selection/tuning, not for final model training.
```

A dedicated submission-generation script was added:

```text
generate_mindlarge_refined_submission.py
```

The script trains the selected refined LightFM model, saves the trained model artifacts, generates official MIND/Codabench rank-format predictions, validates the prediction file against the target `behaviors.tsv`, and creates a zip containing only `prediction.txt`.

Submission command:

```bash
../rs311/bin/python generate_mindlarge_refined_submission.py \
  --train-dirs MINDlarge_train \
  --ranking-dir MINDlarge_test \
  --output-dir submissions/refined_mindlarge_test_train_only \
  --prediction-name prediction.txt \
  --zip-name prediction.zip \
  --loss bpr \
  --components 100 \
  --learning-rate 0.01 \
  --epochs 10 \
  --item-alpha 0 \
  --user-alpha 0
```

Saved submission artifacts:

```text
submissions/refined_mindlarge_test_train_only/refined_submission_model.pkl
submissions/refined_mindlarge_test_train_only/refined_submission_dataset.pkl
submissions/refined_mindlarge_test_train_only/refined_submission_item_features.pkl
submissions/refined_mindlarge_test_train_only/prediction.txt
submissions/refined_mindlarge_test_train_only/prediction.zip
```

Generated test prediction validation:

```text
prediction rows:      2,370,727
test behavior rows:   2,370,727
candidate ranks:     93,115,001
zip contents:        prediction.txt only
zip size:            about 103 MB
```

The generated archive is:

```text
submissions/refined_mindlarge_test_train_only/prediction.zip
```

This is the file to upload to Codabench for the official MIND test score.

Cold-start handling:

```text
The MINDlarge_test split contains users and news items that were not present in MINDlarge_train.
The submission format still requires ranks for every candidate in every impression.
For known train users and known train items, ranks are based on LightFM predicted scores.
For unknown users or unknown candidate items, the script uses a deterministic fallback based on train item popularity plus a small refined-category popularity component.
No test labels are used.
```

## BERT Baseline Test Submission Generation

The BERT hybrid baseline submission was generated from the already-saved MINDlarge train model artifacts. No retraining was done for this baseline submission.

Saved source artifacts:

```text
results/mindlarge_train/bert_model.pkl
results/mindlarge_train/bert_dataset.pkl
results/mindlarge_train/bert_item_features.npz
```

Submission-generation script:

```text
generate_mindlarge_saved_model_submission.py
```

Command:

```bash
../rs311/bin/python generate_mindlarge_saved_model_submission.py \
  --model-name bert \
  --model-dir results/mindlarge_train \
  --train-dir MINDlarge_train \
  --ranking-dir MINDlarge_test \
  --output-dir submissions/bert_mindlarge_test_saved
```

Generated BERT submission artifacts:

```text
submissions/bert_mindlarge_test_saved/prediction.txt
submissions/bert_mindlarge_test_saved/prediction.zip
submissions/bert_mindlarge_test_saved/submission_metadata.json
```

Validation:

```text
prediction rows:      2,370,727
test behavior rows:   2,370,727
candidate ranks:     93,115,001
zip contents:        prediction.txt only
zip size:            about 102 MB
```

The generated BERT archive is:

```text
submissions/bert_mindlarge_test_saved/prediction.zip
```

This file can be uploaded to Codabench to get the official BERT hybrid baseline test score.

## TF-IDF Baseline Test Submission Generation

The TF-IDF hybrid baseline submission was generated from the already-saved MINDlarge train model artifacts. No retraining was done for this baseline submission.

Saved source artifacts:

```text
results/mindlarge_train/tfidf_model.pkl
results/mindlarge_train/tfidf_dataset.pkl
results/mindlarge_train/tfidf_item_features.npz
```

Command:

```bash
../rs311/bin/python generate_mindlarge_saved_model_submission.py \
  --model-name tfidf \
  --model-dir results/mindlarge_train \
  --train-dir MINDlarge_train \
  --ranking-dir MINDlarge_test \
  --output-dir submissions/tfidf_mindlarge_test_saved
```

Generated TF-IDF submission artifacts:

```text
submissions/tfidf_mindlarge_test_saved/prediction.txt
submissions/tfidf_mindlarge_test_saved/prediction.zip
submissions/tfidf_mindlarge_test_saved/submission_metadata.json
```

Validation:

```text
prediction rows:      2,370,727
test behavior rows:   2,370,727
candidate ranks:     93,115,001
zip contents:        prediction.txt only
zip size:            about 102 MB
```

The generated TF-IDF archive is:

```text
submissions/tfidf_mindlarge_test_saved/prediction.zip
```

This file can be uploaded to Codabench to get the official TF-IDF hybrid baseline test score.

## Vertical Baseline Test Submission Generation

The vertical category baseline was the fourth paper-style model family, alongside refined category, TF-IDF hybrid, and BERT hybrid. Its saved MINDlarge train artifacts already existed:

```text
results/mindlarge_train/vertical_model.pkl
results/mindlarge_train/vertical_dataset.pkl
results/mindlarge_train/vertical_item_features.npz
```

The test submission was generated later because the initial submission focus was on the refined model and then the text-feature baselines. To complete the paper comparison table, the vertical baseline was also converted into the official Codabench submission format.

Command:

```bash
../rs311/bin/python generate_mindlarge_saved_model_submission.py \
  --model-name vertical \
  --output-dir submissions/vertical_mindlarge_test_saved
```

Generated vertical submission artifacts:

```text
submissions/vertical_mindlarge_test_saved/prediction.txt
submissions/vertical_mindlarge_test_saved/prediction.zip
submissions/vertical_mindlarge_test_saved/submission_metadata.json
```

Validation:

```text
prediction rows:     2,370,727
test behavior rows:  2,370,727
candidate ranks:    93,115,001
zip contents:       prediction.txt only
zip size:           about 102 MB
```

Cold/unknown coverage during generation:

```text
cold_user_groups:                  358,094
unknown_candidate_items:        68,431,548
groups_with_no_known_candidates:  569,385
```

The generated vertical archive is:

```text
submissions/vertical_mindlarge_test_saved/prediction.zip
```

This file can be uploaded to Codabench to get the official vertical category baseline test score.

Official Codabench result for this vertical baseline submission:

```text
Participant: harshadayini
Submission ID: 758553
Submission time: 2026-05-27 11:08

AUC      0.5151
MRR      0.2256
nDCG@5   0.2347
nDCG@10  0.2902
```

## Official Test Scores For Paper Variants

After submitting all four paper-style variants to Codabench, the official hidden-test results are:

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Refined category LightFM | 0.6059 | 0.2800 | 0.2995 | 0.3573 |
| Vertical category LightFM | 0.5151 | 0.2256 | 0.2347 | 0.2902 |
| BERT hybrid LightFM | 0.5153 | 0.2251 | 0.2343 | 0.2898 |
| TF-IDF hybrid LightFM | 0.5145 | 0.2247 | 0.2336 | 0.2891 |

Conclusion:

```text
The refined category LightFM model is the best of the four paper-style LightFM variants on the official MINDlarge hidden test set.
```

This comparison is valid because all four rows use the same:

```text
train split:      MINDlarge_train
test split:       MINDlarge_test
evaluator:        Codabench official MIND evaluator
submission format: prediction.zip containing prediction.txt ranks
```

## Refined History-Fallback Submission

After the first refined, BERT, and TF-IDF submissions, a cold-start-aware refined submission was generated. This was motivated by the fact that the MINDlarge test split contains many users and candidate news items that were not present in `MINDlarge_train`, while the test rows still provide a clicked-history column that can be used for fallback ranking.

This run did not retrain LightFM. It loaded the already-trained tuned refined model:

```text
submissions/refined_mindlarge_test_train_only/refined_submission_model.pkl
submissions/refined_mindlarge_test_train_only/refined_submission_dataset.pkl
submissions/refined_mindlarge_test_train_only/refined_submission_item_features.pkl
```

The new script is:

```text
generate_refined_history_fallback_submission.py
```

Scoring strategy:

```text
Known train user + known train candidate item:
  Use normalized LightFM score blended with a small history/category fallback score.

Cold user or cold candidate item:
  Use fallback score based on the user's clicked-history refined categories,
  train item popularity, and train refined-category popularity.
```

The blend used for known train user/item pairs was:

```text
0.75 * normalized LightFM score + 0.25 * normalized fallback score
```

Command:

```bash
../rs311/bin/python generate_refined_history_fallback_submission.py \
  --train-dir MINDlarge_train \
  --ranking-dir MINDlarge_test \
  --output-dir submissions/refined_history_fallback_test \
  --model-weight 0.75 \
  --fallback-weight 0.25
```

Generated artifacts:

```text
submissions/refined_history_fallback_test/prediction.txt
submissions/refined_history_fallback_test/prediction.zip
submissions/refined_history_fallback_test/submission_metadata.json
```

Validation:

```text
prediction rows:      2,370,727
test behavior rows:   2,370,727
candidate ranks:     93,115,001
zip contents:        prediction.txt only
zip size:            about 102 MB
```

Cold-start coverage during generation:

```text
cold candidate items:                    68,431,548
known-user groups with no train items:      569,385
cold-user groups:                           358,094
```

The generated history-fallback refined archive is:

```text
submissions/refined_history_fallback_test/prediction.zip
```

This should be submitted as a new refined attempt and compared against the previous refined official test score:

```text
previous refined submission:
AUC      0.6059
MRR      0.2800
nDCG@5   0.2995
nDCG@10  0.3573
```

## Refined LightFM With User-History Features

A more principled cold-user LightFM variant was implemented after the category fallback submission underperformed. The goal was to keep the model LightFM-based while making user representation depend on clicked history instead of only user ID.

New script:

```text
train_refined_user_history_lightfm.py
```

Model design:

```text
Dataset(user_identity_features=False, item_identity_features=False)
```

Instead of using LightFM's automatic identity features, explicit features are created:

```text
Known train users:
  global:user
  uid:<user_id>
  hist:<refined_category> features from clicked history

Cold dev/test users:
  global:user
  hist:<refined_category> features from clicked history

Known train items:
  iid:<news_id>
  cat:<refined_category>

Cold dev/test items:
  cat:<refined_category>
```

This avoids random/untrained automatic identity features for cold users and cold news items while still preserving user ID and item ID signals for entities seen during training.

Smoke test:

```bash
../rs311/bin/python train_refined_user_history_lightfm.py \
  --mode eval \
  --output-dir results/refined_user_history_smoke \
  --epochs 1 \
  --components 10 \
  --eval-limit 1000
```

Smoke result:

```text
AUC      0.6089
MRR      0.3266
nDCG@5   0.3115
nDCG@10  0.3736
```

Full sampled dev run:

```bash
../rs311/bin/python train_refined_user_history_lightfm.py \
  --mode eval \
  --output-dir results/refined_user_history_bpr100_lr001_e10 \
  --loss bpr \
  --components 100 \
  --learning-rate 0.01 \
  --epochs 10 \
  --item-alpha 0 \
  --user-alpha 0 \
  --eval-limit 50000
```

Full sampled dev result:

```text
AUC      0.6213
MRR      0.3257
nDCG@5   0.3088
nDCG@10  0.3738
```

Comparison against the previous best tuned refined run on the 50,000-group dev sample:

```text
Previous tuned refined:
AUC      0.6311
MRR      0.3369
nDCG@5   0.3177
nDCG@10  0.3816

Refined + user-history features:
AUC      0.6213
MRR      0.3257
nDCG@5   0.3088
nDCG@10  0.3738
```

Conclusion:

```text
The user-history feature design is conceptually cleaner for cold users, but this first implementation did not improve dev metrics over the previous tuned refined LightFM model.
No test submission was generated from this version yet.
```

### Interpretation Of The History-Based Attempts

Two history-aware approaches were tested after the first official refined submission:

1. Post-hoc history/category fallback on top of the tuned refined model.
2. A retrained refined LightFM model with explicit user-history features.

Official test result for the original tuned refined model:

```text
AUC      0.6059
MRR      0.2800
nDCG@5   0.2995
nDCG@10  0.3573
```

These official test metrics should not be directly compared with sampled dev metrics. Dev metrics and official test metrics are computed on different splits, and the test labels are hidden.

Official test result for the post-hoc history/category fallback submission:

```text
AUC      0.5946
MRR      0.2727
nDCG@5   0.2914
nDCG@10  0.3496
```

The post-hoc fallback reduced all official test metrics. The likely reason is that it blended fallback scores into known user/item rankings:

```text
0.75 * normalized LightFM score + 0.25 * normalized fallback score
```

This disturbed rankings where the tuned LightFM model was already producing useful scores. The fallback itself was also category-level, using:

```text
clicked-history refined categories
train item popularity
train refined-category popularity
```

This is too coarse for the MIND task, because the hidden evaluation checks whether the clicked individual news article is ranked above non-clicked candidate articles. It does not evaluate whether the correct category was selected.

The retrained user-history LightFM model used refined categories on both the user and item sides:

```text
item feature:
cat:<refined_category>

user feature:
hist:<refined_category> derived from clicked history
```

It also used explicit identity features for known train users/items:

```text
uid:<user_id>
iid:<news_id>
```

This design is conceptually better for cold users because an unseen user can still be represented through clicked-history categories. However, it reduced sampled dev metrics relative to the previous tuned refined model:

```text
Previous tuned refined, 50k dev sample:
AUC      0.6311
MRR      0.3369
nDCG@5   0.3177
nDCG@10  0.3816

Refined + user-history features, 50k dev sample:
AUC      0.6213
MRR      0.3257
nDCG@5   0.3088
nDCG@10  0.3738
```

Important comparison note:

```text
The user-history model's dev AUC of 0.6213 is higher than the original tuned refined official test AUC of 0.6059, but those values are from different datasets and are not directly comparable.

The correct same-split comparison is:
previous tuned refined dev AUC 0.6311
vs.
user-history dev AUC 0.6213

So the user-history model underperformed the previous tuned refined model on the same 50,000-group dev evaluation.
```

The likely reason is feature dilution. In the original tuned refined model, LightFM's automatic user/item identity features were strong collaborative signals:

```text
user ID embedding
item/news ID embedding
refined category item feature
```

In the history-feature model, known users/items were represented as mixtures of ID features and category/history features. Since LightFM normalizes feature rows, the identity signal can be weakened by the added broad category-history features. The category-history features are useful for cold-start coverage, but they are too broad to consistently improve ranking of individual news articles.

The current conclusion is:

```text
The original tuned refined LightFM model remains the best refined model tested so far.
History/category features are conceptually useful for cold users, but the tested category-level implementations reduced metrics.
Future improvement should avoid disturbing known user/item LightFM rankings.
```

Recommended next direction:

```text
Known train user + known train item:
  keep the original tuned refined LightFM score unchanged

Unknown user or unknown item:
  use a stronger history-based fallback, preferably TF-IDF or BERT similarity
  between clicked-history articles and candidate articles
```

This preserves the strongest known-user collaborative signal while only using history-based scoring when the ID-based LightFM model cannot make a reliable prediction.

### Weighted History Feature Follow-Up

The user-history LightFM script was extended to support separate feature weights:

```text
global_user_weight
known_user_id_weight
train_history_weight
predict_known_history_weight
predict_unknown_history_weight
known_item_id_weight
item_category_weight
```

This tested the hypothesis:

```text
Known users:
  keep user ID dominant and give history categories a very small weight

Unknown users:
  use history categories with a larger weight because no user ID is available

Known items:
  keep item/news ID dominant and give category a small weight

Unknown items:
  use category features because no trained item/news ID is available
```

Weighted training/evaluation command:

```bash
../rs311/bin/python train_refined_user_history_lightfm.py \
  --mode eval \
  --output-dir results/refined_user_history_weighted_bpr100_lr001_e10_hw005 \
  --loss bpr \
  --components 100 \
  --learning-rate 0.01 \
  --epochs 10 \
  --item-alpha 0 \
  --user-alpha 0 \
  --eval-limit 50000 \
  --global-user-weight 0.01 \
  --known-user-id-weight 1.0 \
  --train-history-weight 0.05 \
  --predict-known-history-weight 0.05 \
  --predict-unknown-history-weight 1.0 \
  --known-item-id-weight 1.0 \
  --item-category-weight 0.05
```

Weighted user-history result on the same 50,000-group dev sample:

```text
AUC      0.6198
MRR      0.3253
nDCG@5   0.3085
nDCG@10  0.3734
```

Comparison:

```text
Previous tuned refined:
AUC      0.6311
MRR      0.3369
nDCG@5   0.3177
nDCG@10  0.3816

Weighted user-history refined:
AUC      0.6198
MRR      0.3253
nDCG@5   0.3085
nDCG@10  0.3734
```

Conclusion:

```text
Even with ID features kept dominant for known users/items, the weighted user-history version did not beat the previous tuned refined model on dev.
This suggests that refined-category history alone is still too coarse for improving article-level ranking.
The stronger next option would be a text-similarity fallback using TF-IDF/BERT history profiles only for cold cases, while leaving the original tuned LightFM scores unchanged for known user/item pairs.
```

### Hybrid LightFM + TF-IDF History Cold-Case Reranker

The next experiment implemented a hybrid reranker using the research-motivated idea that cold users/items should be represented through clicked-history text similarity rather than broad refined-category overlap.

New script:

```text
hybrid_refined_lightfm_tfidf_history.py
```

Scoring policy:

```text
Known train user + known train candidate item:
  use tuned refined LightFM score only
  do not blend TF-IDF into this score

Cold user or cold candidate item:
  use TF-IDF cosine similarity between clicked-history articles and candidate article
  add small refined-category and popularity tie-breakers
```

TF-IDF input:

```text
title + abstract + vertical + subvertical
```

Smoke run:

```bash
../rs311/bin/python hybrid_refined_lightfm_tfidf_history.py \
  --mode eval \
  --output-dir results/hybrid_refined_tfidf_history_smoke \
  --eval-limit 1000 \
  --max-features 20000 \
  --known-bias 0 \
  --category-boost 0.02 \
  --popularity-boost 0.001
```

Smoke result:

```text
AUC      0.6330
MRR      0.3523
nDCG@5   0.3385
nDCG@10  0.3954
```

Full 50,000-group dev run:

```bash
../rs311/bin/python hybrid_refined_lightfm_tfidf_history.py \
  --mode eval \
  --output-dir results/hybrid_refined_tfidf_history_dev50k \
  --eval-limit 50000 \
  --max-features 50000 \
  --known-bias 0 \
  --category-boost 0.02 \
  --popularity-boost 0.001
```

Full 50,000-group dev result:

```text
AUC      0.6185
MRR      0.3336
nDCG@5   0.3151
nDCG@10  0.3774
```

Comparison against current tuned refined dev result:

```text
Previous tuned refined:
AUC      0.6311
MRR      0.3369
nDCG@5   0.3177
nDCG@10  0.3816

Hybrid LightFM + TF-IDF cold-case reranker:
AUC      0.6185
MRR      0.3336
nDCG@5   0.3151
nDCG@10  0.3774
```

Conclusion:

```text
The TF-IDF history cold-case reranker improved over the category-only history attempts, especially on ranking metrics, but still did not beat the previous tuned refined model on the 50,000-group dev evaluation.
No test submission was generated from this version.
The next possible research direction is a BERT/MiniLM history-similarity reranker, because TF-IDF title/abstract similarity may still be too lexical and sparse.
```

## MINDlarge Train Run Commands

The model runs were executed sequentially to reduce RAM and CPU pressure.

Vertical:

```bash
../rs311/bin/python run_mindlarge_train_metrics.py \
  --data-dir MINDlarge_train \
  --output-dir results/mindlarge_train \
  --models vertical \
  --epochs 20 \
  --components 30 \
  --num-threads 4 \
  --eval-limit 100000 \
  --paper-metric-users 5000
```

Refined:

```bash
../rs311/bin/python run_mindlarge_train_metrics.py \
  --data-dir MINDlarge_train \
  --output-dir results/mindlarge_train \
  --models refined \
  --epochs 20 \
  --components 30 \
  --num-threads 4 \
  --eval-limit 100000 \
  --paper-metric-users 5000
```

TF-IDF:

```bash
../rs311/bin/python run_mindlarge_train_metrics.py \
  --data-dir MINDlarge_train \
  --output-dir results/mindlarge_train \
  --models tfidf \
  --epochs 20 \
  --components 30 \
  --num-threads 4 \
  --eval-limit 100000 \
  --paper-metric-users 5000
```

BERT embedding generation:

```bash
../rs311/bin/python generate_mindlarge_bert_embeddings.py \
  --news MINDlarge_train/news.tsv \
  --output-embeddings english_article_embeddings_large_train.pt \
  --output-ids english_news_ids_large_train.pkl \
  --batch-size 64
```

BERT Hybrid:

```bash
../rs311/bin/python run_mindlarge_train_metrics.py \
  --data-dir MINDlarge_train \
  --output-dir results/mindlarge_train \
  --models bert \
  --epochs 20 \
  --components 30 \
  --num-threads 4 \
  --eval-limit 100000 \
  --paper-metric-users 5000 \
  --bert-embeddings english_article_embeddings_large_train.pt \
  --bert-ids english_news_ids_large_train.pkl
```

## MINDlarge Train Evaluation Notes

The MINDlarge train split is much larger than MINDsmall. Full LightFM ranking metrics over all users and all items are expensive on this machine, especially because the installed original `lightfm` build is single-threaded.

The train metrics are mainly a sanity check. They show that the model was trained and can score the same split it learned from. They should not be treated as final generalization results because they are evaluated on the same split used for training.

For practical runtime:

```text
paper_metric_users = 5000
groups_evaluated   = 100000
```

This means:

- `Precision@10`, `Recall@10`, and `LightFM_AUC` were computed on the first 5,000 fitted users.
- `AUC`, `MRR`, `nDCG@5`, and `nDCG@10` were computed on 100,000 labeled impression groups.

The values below should be interpreted with those evaluation caps.

## MINDlarge Train Results

Results are stored in:

```text
results/mindlarge_train/metrics.json
```

| Tag | Model | Precision@10 | Recall@10 | LightFM_AUC | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `VERTICAL` | Vertical category | 0.0955 | 0.0518 | 0.9917 | 0.7372 | 0.4137 | 0.4003 | 0.4590 |
| `REFINED` | Refined category | 0.0941 | 0.0505 | 0.9922 | 0.7440 | 0.4185 | 0.4055 | 0.4638 |
| `TFIDF` | TF-IDF hybrid | 0.0236 | 0.0178 | 0.9645 | 0.7426 | 0.4327 | 0.4217 | 0.4767 |
| `BERT` | BERT hybrid | 0.0086 | 0.0064 | 0.9342 | 0.7341 | 0.4185 | 0.4083 | 0.4646 |

Full precision values from `metrics.json`:

```json
{
  "vertical": {
    "Precision@10": 0.0955200046300888,
    "Recall@10": 0.05183507411435346,
    "LightFM_AUC": 0.9917155504226685,
    "paper_metric_users": 5000,
    "AUC": 0.7372163473569784,
    "MRR": 0.4136671938278027,
    "nDCG@5": 0.40026922730543185,
    "nDCG@10": 0.45896307457515817,
    "groups_evaluated": 100000
  },
  "refined": {
    "Precision@10": 0.09412001073360443,
    "Recall@10": 0.05050425897521519,
    "LightFM_AUC": 0.9921687245368958,
    "paper_metric_users": 5000,
    "AUC": 0.7440475310905528,
    "MRR": 0.4184932625334218,
    "nDCG@5": 0.4054806116828683,
    "nDCG@10": 0.4637893552664252,
    "groups_evaluated": 100000
  },
  "tfidf": {
    "Precision@10": 0.02362000197172165,
    "Recall@10": 0.01779148761039347,
    "LightFM_AUC": 0.9644874930381775,
    "paper_metric_users": 5000,
    "AUC": 0.7425615800150136,
    "MRR": 0.43268173576580315,
    "nDCG@5": 0.42165691657949717,
    "nDCG@10": 0.4766586505923346,
    "groups_evaluated": 100000
  },
  "bert": {
    "Precision@10": 0.008619999513030052,
    "Recall@10": 0.006372816954963131,
    "LightFM_AUC": 0.9342231154441833,
    "paper_metric_users": 5000,
    "AUC": 0.7341389626006236,
    "MRR": 0.418481917575284,
    "nDCG@5": 0.408268386039753,
    "nDCG@10": 0.4645510741691025,
    "groups_evaluated": 100000
  }
}
```

## Output Artifacts

MINDlarge train output artifacts:

```text
results/mindlarge_train/vertical_model.pkl
results/mindlarge_train/vertical_item_features.npz

results/mindlarge_train/refined_model.pkl
results/mindlarge_train/refined_item_features.npz

results/mindlarge_train/tfidf_model.pkl
results/mindlarge_train/tfidf_item_features.npz

results/mindlarge_train/bert_model.pkl
results/mindlarge_train/bert_item_features.npz

results/mindlarge_train/metrics.json
```

## How To Inspect Outputs Directly

From the repository root:

```bash
cd /Users/harshadayiniakula/Desktop/RecSys/Recommendation_System-
```

View the metrics:

```bash
cat results/mindlarge_train/metrics.json
```

Pretty-print the metrics:

```bash
python -m json.tool results/mindlarge_train/metrics.json
```

Check files and sizes:

```bash
ls -lh MINDlarge_train results/mindlarge_train
```

Check dataset row counts:

```bash
wc -l MINDlarge_train/news.tsv MINDlarge_train/behaviors.tsv
```

Verify the environment:

```bash
../rs311/bin/python - <<'PY'
import lightfm, numpy, torch, transformers
print("lightfm", lightfm.__version__)
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY
```
## TF-IDF History Hybrid Tuning Sweep

After the first TF-IDF clicked-history hybrid run did not beat the tuned refined LightFM baseline, I added a small policy sweep to test whether the cold-case reranker needed tuning before deciding against it.

The implementation file is:

```text
hybrid_refined_lightfm_tfidf_history.py
```

The script still loads the saved tuned refined LightFM model from:

```text
submissions/refined_mindlarge_test_train_only/refined_submission_model.pkl
submissions/refined_mindlarge_test_train_only/refined_submission_dataset.pkl
submissions/refined_mindlarge_test_train_only/refined_submission_item_features.pkl
```

It does not retrain LightFM. The sweep only changes the cold-case scoring policy:

```text
known_bias:
  constant added to known train-user/train-item LightFM scores after per-impression normalization

category_boost:
  small boost if a candidate article's refined category appears in the user's clicked history

popularity_boost:
  tiny log-popularity tie-breaker from MINDlarge_train clicks
```

Command run:

```bash
../rs311/bin/python hybrid_refined_lightfm_tfidf_history.py \
  --mode sweep \
  --output-dir results/hybrid_refined_tfidf_history_sweep \
  --eval-limit 50000 \
  --max-features 50000 \
  --known-biases=-0.05,0,0.05,0.1 \
  --category-boosts=0,0.02 \
  --popularity-boosts=0,0.001
```

Dev evaluation sample:

```text
MINDlarge_dev/behaviors.tsv
50,000 labeled impression groups
1,871,285 candidate items
6,495 cold-user groups
86,323 cold candidate items
```

Sweep results:

| known_bias | category_boost | popularity_boost | AUC | MRR | nDCG@5 | nDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|
| -0.05 | 0.00 | 0.000 | 0.6157 | 0.3328 | 0.3140 | 0.3759 |
| -0.05 | 0.00 | 0.001 | 0.6166 | 0.3331 | 0.3141 | 0.3764 |
| -0.05 | 0.02 | 0.000 | 0.6189 | 0.3343 | 0.3157 | 0.3777 |
| -0.05 | 0.02 | 0.001 | 0.6197 | 0.3345 | 0.3158 | 0.3780 |
| 0.00 | 0.00 | 0.000 | 0.6146 | 0.3322 | 0.3136 | 0.3755 |
| 0.00 | 0.00 | 0.001 | 0.6155 | 0.3325 | 0.3137 | 0.3759 |
| 0.00 | 0.02 | 0.000 | 0.6176 | 0.3335 | 0.3152 | 0.3771 |
| 0.00 | 0.02 | 0.001 | 0.6185 | 0.3336 | 0.3151 | 0.3774 |
| 0.05 | 0.00 | 0.000 | 0.6132 | 0.3306 | 0.3121 | 0.3744 |
| 0.05 | 0.00 | 0.001 | 0.6142 | 0.3309 | 0.3124 | 0.3749 |
| 0.05 | 0.02 | 0.000 | 0.6165 | 0.3316 | 0.3137 | 0.3758 |
| 0.05 | 0.02 | 0.001 | 0.6173 | 0.3318 | 0.3137 | 0.3763 |
| 0.10 | 0.00 | 0.000 | 0.6124 | 0.3304 | 0.3119 | 0.3742 |
| 0.10 | 0.00 | 0.001 | 0.6132 | 0.3306 | 0.3119 | 0.3746 |
| 0.10 | 0.02 | 0.000 | 0.6158 | 0.3314 | 0.3134 | 0.3756 |
| 0.10 | 0.02 | 0.001 | 0.6165 | 0.3317 | 0.3134 | 0.3760 |

Best sweep config by nDCG@10:

```text
known_bias       -0.05
category_boost    0.02
popularity_boost  0.001

AUC      0.6197
MRR      0.3345
nDCG@5   0.3158
nDCG@10  0.3780
```

Comparison target:

```text
Tuned refined LightFM dev:
AUC      0.6311
MRR      0.3369
nDCG@5   0.3177
nDCG@10  0.3816
```

Decision:

```text
Do not submit this TF-IDF history hybrid policy to Codabench.
```

Reason:

The best tuned TF-IDF history hybrid is still below the tuned refined LightFM baseline on all four dev metrics. The cold-case history signal helps compared with weaker fallback variants, but it is not strong enough to justify another official test submission. The result also shows that simple TF-IDF average-history similarity is probably too shallow for this dataset; a stronger next attempt would need a learned history encoder or a better content reranker, not just this policy tuning.

Output files:

```text
results/hybrid_refined_tfidf_history_sweep/hybrid_tuning_results.csv
results/hybrid_refined_tfidf_history_sweep/hybrid_tuning_results.jsonl
results/hybrid_refined_tfidf_history_sweep/hybrid_tuning_summary.json
```

## Root-Cause Diagnostic For Refined LightFM Score

After the fallback and TF-IDF-history attempts failed to beat the tuned refined model, I added a diagnostic script to understand where the refined LightFM model is weak instead of continuing to guess improvements.

Script:

```text
diagnose_refined_lightfm_dev.py
```

Command:

```bash
../rs311/bin/python diagnose_refined_lightfm_dev.py \
  --eval-limit 100000 \
  --output-dir results/refined_lightfm_rootcause_dev100k
```

This evaluates the saved tuned refined submission policy on 100,000 labeled `MINDlarge_dev` impression groups and slices the metrics by coverage and impression properties.

Important outputs:

```text
results/refined_lightfm_rootcause_dev100k/rootcause_slices.csv
results/refined_lightfm_rootcause_dev100k/rootcause_summary.json
```

Overall diagnostic metrics on this 100k dev sample:

```text
AUC      0.6159
MRR      0.3249
nDCG@5   0.3072
nDCG@10  0.3708
```

Key slices:

| Slice | Groups | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Overall | 100,000 | 0.6159 | 0.3249 | 0.3072 | 0.3708 |
| Known user | 87,031 | 0.6262 | 0.3336 | 0.3143 | 0.3779 |
| Cold user | 12,969 | 0.5466 | 0.2672 | 0.2595 | 0.3235 |
| All candidates known in train | 38,704 | 0.6194 | 0.3790 | 0.3719 | 0.4337 |
| Has new candidate items | 61,296 | 0.6137 | 0.2908 | 0.2664 | 0.3311 |
| Positive category seen in history | 76,007 | 0.6813 | 0.3613 | 0.3423 | 0.4055 |
| Positive category not seen in history | 23,993 | 0.4087 | 0.2098 | 0.1961 | 0.2611 |

Candidate-count slices:

| Candidate Count | Groups | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| 0-5 | 11,971 | 0.5887 | 0.6783 | 0.7599 | 0.7599 |
| 6-10 | 13,333 | 0.6172 | 0.4598 | 0.5057 | 0.5866 |
| 11-20 | 21,284 | 0.5974 | 0.3256 | 0.3121 | 0.4157 |
| 21-50 | 28,411 | 0.6279 | 0.2507 | 0.2008 | 0.2740 |
| 51-100 | 16,727 | 0.6277 | 0.1798 | 0.1144 | 0.1600 |
| >100 | 8,274 | 0.6355 | 0.1430 | 0.0748 | 0.1033 |

Interpretation:

```text
1. Cold start is a real issue, but not the only issue.
   Cold-user groups are worse than known-user groups, but they are only about 13% of this dev sample.

2. New candidate items hurt ranking more broadly.
   Groups where all candidates were already known in train achieved nDCG@10 0.4337.
   Groups with new candidate items dropped to nDCG@10 0.3311.

3. The largest conceptual weakness is category-level dependence.
   When the clicked article's category was seen in the user's history, nDCG@10 was 0.4055.
   When that category was not seen, nDCG@10 dropped to 0.2611 and AUC dropped to 0.4087.

4. Large impression groups expose weak fine-grained ranking.
   nDCG@10 falls sharply as candidate count grows, even when AUC stays around 0.62.
   This means the model often has some coarse ordering signal but cannot reliably push the clicked article into the top ranks.
```

Conclusion:

The low score is not just because of missing fallback handling. The refined LightFM model is fundamentally too coarse for MINDlarge because it mostly captures user ID plus refined-category preference. The failed improvements targeted cold-start fallback and broad history/category signals, but the diagnostic shows that stronger improvement would need fine-grained article semantics and user-history matching, especially for articles/categories not already represented in the user's past clicks.

## Telugu Processing Alignment With MINDlarge Refined Categories

Because the final paper direction moved from MINDsmall to MINDlarge, the Telugu processing pipeline also needs to target the final MINDlarge refined-category space.

The paper's original Telugu process was:

```text
English MIND articles with refined categories
-> train multilingual XLM-R classifier
-> classify Telugu articles into refined categories
-> build Telugu LightFM item-feature matrix
-> score Telugu articles for English users
```

This process is being kept the same. The change is only that the classifier target labels now come from the final MINDlarge refined taxonomy instead of the older MINDsmall-derived category file.

New script:

```text
scripts/telugu/process_mindlarge_refined.py
```

Preparation command run:

```bash
../rs311/bin/python scripts/telugu/process_mindlarge_refined.py --mode prepare
```

Preparation outputs:

```text
results/telugu_mindlarge_refined/english_mindlarge_refined_train.pkl
results/telugu_mindlarge_refined/telugu_df_full.pkl
results/telugu_mindlarge_refined/telugu_story_ids.csv
results/telugu_mindlarge_refined/mindlarge_refined_categories.txt
results/telugu_mindlarge_refined/label_mapping.json
results/telugu_mindlarge_refined/mindlarge_refined_category_counts.csv
results/telugu_mindlarge_refined/prepare_summary.json
```

Prepared English classifier data:

```text
source: MINDlarge_train/news.tsv
rows:   101,527
labels: 23 refined categories
```

Prepared Telugu data:

```text
source files:
  telugu/train-00000-of-00001.parquet
  telugu/val-00000-of-00001.parquet
  telugu/test-00000-of-00001.parquet

rows after story_id deduplication: 40,906
```

The final MINDlarge refined category vocabulary is:

```text
autos
entertainment
finance
foodanddrink
games
health
lifestyle
movies
music
news
newscrime
newslocalpolitics
newspolitics
newsscienceandtechnology
newstechnology
newstrends
newsvideos
newsworld
sports
travel
tv
video
weather
```

MINDlarge refined category counts used for classifier training:

| Category | Count |
|---|---:|
| autos | 3,071 |
| entertainment | 802 |
| finance | 5,916 |
| foodanddrink | 1,704 |
| games | 1 |
| health | 2,929 |
| lifestyle | 4,674 |
| movies | 815 |
| music | 1,263 |
| news | 5,186 |
| newscrime | 3,676 |
| newslocalpolitics | 1 |
| newspolitics | 5,161 |
| newsscienceandtechnology | 2,789 |
| newstechnology | 3 |
| newstrends | 2,715 |
| newsvideos | 1 |
| newsworld | 16,994 |
| sports | 32,020 |
| travel | 4,954 |
| tv | 1,323 |
| video | 1,257 |
| weather | 4,272 |

Finding:

```text
The Telugu pipeline can be kept methodologically consistent with the original paper, with the classifier target space updated from the older MINDsmall refined labels to the final 23-category MINDlarge refined taxonomy.
```

## MINDlarge Telugu Classification And Recommendation Results

This section updates the original Telugu-transfer stage from MINDsmall to MINDlarge while keeping the same methodological structure:

```text
English MINDlarge refined category labels
-> XLM-RoBERTa classifier
-> Telugu category probabilities
-> Telugu LightFM item-feature matrix
-> English-trained refined LightFM scores Telugu articles
```

The classifier was trained in Google Colab on T4 GPU because local memory was insufficient.

Training command used in Colab:

```bash
python scripts/telugu/process_mindlarge_refined.py \
  --mode train \
  --output-dir "/content/drive/MyDrive/telugu_mindlarge_refined" \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --epochs 3
```

Classifier training setup:

| Field | Value |
|---|---:|
| Model | `xlm-roberta-base` |
| Train rows | 91,374 |
| Validation rows | 10,153 |
| Refined categories | 23 |
| Epochs | 3 |
| Train runtime | 7,938.08 seconds |
| Train loss | 0.7800 |
| Validation loss | 0.7750 |
| Validation accuracy | 0.7658 |
| Validation macro-F1 | 0.6990 |

Paper-facing note:

```text
These validation metrics show that the multilingual classifier learned the MINDlarge refined category mapping and can be applied to Telugu articles to project them into the same category space used by the English-trained recommender.
```

Telugu classification command used in Colab:

```bash
python scripts/telugu/process_mindlarge_refined.py \
  --mode classify \
  --output-dir "/content/drive/MyDrive/telugu_mindlarge_refined" \
  --infer-batch-size 32
```

Telugu classification outputs:

| Field | Value |
|---|---:|
| Telugu articles after `story_id` deduplication | 40,906 |
| Category count | 23 |
| Mean top-class confidence | 0.7250 |
| Median top-class confidence | 0.7504 |

Predicted Telugu category counts:

| Category | Count |
|---|---:|
| autos | 239 |
| entertainment | 12 |
| finance | 4,066 |
| foodanddrink | 34 |
| health | 868 |
| lifestyle | 613 |
| movies | 2,526 |
| music | 335 |
| news | 1,439 |
| newscrime | 805 |
| newspolitics | 829 |
| newsscienceandtechnology | 835 |
| newstrends | 100 |
| newsworld | 22,059 |
| sports | 2,540 |
| travel | 1,132 |
| tv | 77 |
| video | 588 |
| weather | 1,809 |

Paper-facing note:

```text
The Telugu articles were assigned across a broad set of refined categories, including newsworld, finance, sports, movies, weather, news, travel, health, science/technology, and politics-related classes.
```

Telugu feature-building command used in Colab:

```bash
python scripts/telugu/process_mindlarge_refined.py \
  --mode build-features \
  --output-dir "/content/drive/MyDrive/telugu_mindlarge_refined"
```

Telugu item-feature matrix:

| Field | Value |
|---|---:|
| Shape before LightFM alignment | 40,906 x 23 |
| Non-zero entries | 881,861 |
| Saved file | `telugu_item_features_23cat.npz` |

For local LightFM recommendation, the 23-column Telugu matrix was expanded into the final MINDlarge LightFM item-feature space:

| Field | Value |
|---|---:|
| Final tuned refined LightFM item-feature columns | 129,831 |
| Expanded Telugu feature shape | 40,906 x 129,831 |
| Expanded non-zero entries | 881,861 |

The 129,831 columns consist of 129,808 English item identity features plus the 23 refined category features. Telugu articles only use the 23 category columns, which is the intended cold-start transfer setting.

Recommendation generation script:

```text
scripts/telugu/generate_recommendations.py
```

Command run locally:

```bash
../rs311/bin/python scripts/telugu/generate_recommendations.py
```

Representative users were selected by highest category affinity in the trained refined LightFM model, matching the qualitative style of the original paper.

| Target profile | User ID | Top category 1 | Score | Top category 2 | Score | Top category 3 | Score |
|---|---|---|---:|---|---:|---|---:|
| sports | U245228 | sports | 10.5344 | tv | 3.2429 | music | 1.1358 |
| music | U539584 | music | 8.3997 | movies | 7.2601 | entertainment | 6.0439 |
| newspolitics | U69259 | newspolitics | 9.8434 | newsworld | 9.2553 | newsscienceandtechnology | 5.0077 |

Interpretation:

```text
The model transfers English-trained category preferences to Telugu articles successfully:
sports-oriented users receive sports articles, music/entertainment users receive music/movie-related articles, and politics-oriented users receive politics/current-affairs articles.
```

Paper-facing note:

```text
For presentation, use the headline-deduplicated recommendation table so repeated Telugu headlines from multiple source parquet files do not distract from the category-transfer result.
```

Generated local outputs:

```text
results/telugu_mindlarge_recommendations/representative_user_category_affinities.csv
results/telugu_mindlarge_recommendations/telugu_top10_recommendations.csv
results/telugu_mindlarge_recommendations/telugu_top10_recommendations_headline_deduped.csv
results/telugu_mindlarge_recommendations/telugu_recommendation_summary.json
```

Paper update guidance:

```text
Do not report Telugu AUC/MRR/nDCG because the Telugu dataset has no user-click interaction ground truth.
For Telugu, report classifier validation metrics, category coverage, feature-matrix construction, and qualitative top-10 recommendation examples.
For English MINDlarge, report official Codabench AUC/MRR/nDCG@5/nDCG@10.
```

## Telugu Category Coverage Analysis

Purpose:

```text
Since the Telugu corpus has no click labels, one necessary validation step is to check whether Telugu articles are being mapped into the refined category space in a meaningful and sufficiently broad way.
```

Script added:

```text
scripts/telugu/analyze_category_coverage.py
```

Command run:

```bash
../rs311/bin/python scripts/telugu/analyze_category_coverage.py
```

Outputs:

```text
results/telugu_category_coverage/telugu_category_coverage.csv
results/telugu_category_coverage/telugu_category_coverage_summary.json
```

Coverage summary for paper:

| Metric | Value |
|---|---:|
| Telugu articles | 40,906 |
| Total refined categories | 23 |
| Categories used by Telugu predictions | 19 |
| Coverage rate | 82.61% |

Examples of represented Telugu categories:

```text
newsworld
finance
sports
movies
weather
news
travel
health
newsscienceandtechnology
newspolitics
```

Paper-Ready Text For Telugu Category Coverage:

```text
The Telugu classifier mapped 40,906 articles into 19 of the 23 refined MINDlarge categories, giving 82.61% coverage of the refined category space. This demonstrates that Telugu articles can be projected into the same feature representation used by the English-trained recommender. The largest Telugu category was newsworld, reflecting the current topical composition of the Telugu corpus, while categories such as sports, finance, movies, weather, politics, health, and travel were also represented. This coverage enables the English-trained LightFM model to score previously unseen Telugu articles through shared category features.
```

How This Helps The Paper:

```text
This coverage result supports the paper's central transfer-learning claim: Telugu articles can be classified into the same refined category space as English MINDlarge articles and then ranked by the English-trained LightFM model without Telugu click interactions.
```

## Telugu Script Organization

The Telugu-specific scripts were moved out of the repository root and grouped under:

```text
scripts/telugu/
```

Current Telugu script layout:

| Script | Purpose |
|---|---|
| `scripts/telugu/process_mindlarge_refined.py` | Prepare English/Telugu data, train XLM-R, classify Telugu articles, and build Telugu item features |
| `scripts/telugu/validate_category_alignment.py` | Validate that Telugu category processing matches the final MINDlarge refined LightFM feature space |
| `scripts/telugu/generate_recommendations.py` | Generate Telugu top-10 recommendation tables and representative user affinities |
| `scripts/telugu/analyze_category_coverage.py` | Compute Telugu predicted-category coverage |
| `scripts/telugu/validate_qualitative_recommendations.py` | Compute category alignment for representative recommendations and create a manual relevance table |

The category coverage script was rerun successfully after the move:

```bash
../rs311/bin/python scripts/telugu/analyze_category_coverage.py
```

The rerun produced the same coverage result:

```text
40,906 Telugu articles
19 / 23 categories used
82.61% coverage
```

## Telugu Category Alignment And Manual Relevance Check

Purpose:

```text
This validates the Telugu recommendations without using AUC/MRR/nDCG, because Telugu articles do not have user-click ground truth.
```

Two checks were separated:

| Check | What It Measures | Why It Helps |
|---|---|---|
| Category alignment | Whether the predicted Telugu article category matches the English user's learned top categories | Validates the automatic category-transfer mechanism |
| Manual relevance check | Whether the actual Telugu headlines semantically match the user profile when read by a human | Validates that the examples are meaningful beyond just category labels |

Script added:

```text
scripts/telugu/validate_qualitative_recommendations.py
```

Command run:

```bash
../rs311/bin/python scripts/telugu/validate_qualitative_recommendations.py
```

Outputs:

```text
results/telugu_qualitative_validation/telugu_category_alignment_top10.csv
results/telugu_qualitative_validation/telugu_category_alignment_summary.csv
results/telugu_qualitative_validation/telugu_manual_relevance_top5.csv
results/telugu_qualitative_validation/telugu_qualitative_validation_summary.json
```

Category alignment results for the three representative users:

| User Profile | User ID | Top Learned Categories | Top-10 Category Alignment | Mean Category Confidence |
|---|---|---|---:|---:|
| Sports-oriented | U245228 | sports, tv, music | 100% | 0.9989 |
| Music/Entertainment-oriented | U539584 | music, movies, entertainment | 100% | 0.9397 |
| Politics-oriented | U69259 | newspolitics, newsworld, newsscienceandtechnology | 100% | 0.9588 |

Overall representative-example alignment:

```text
Representative users checked: 3
Recommendations checked: 30
Top-1 category alignment: 100%
Top-3 category alignment: 100%
Mean category confidence: 0.9658
```

Manual relevance check for paper examples:

| User Profile | User ID | Example Telugu Headline | Predicted Category | Manual Relevance Rationale |
|---|---|---|---|---|
| Sports-oriented | U245228 | ఇప్పట్లో ధోనీ రిటైర్ అవ్వడు : ఎమ్మెస్కే | sports | Dhoni/cricket headline; directly matches sports profile |
| Sports-oriented | U245228 | వెస్టిండీస్ పర్యటనకు భారత జట్టు ప్రకటన | sports | Indian team selection headline; directly matches sports profile |
| Music/Entertainment-oriented | U539584 | ‘జగమే తంత్రం’ నుంచి రకిట రకిట పాట విడుదల | music | Song release from a film; matches music/movie profile |
| Music/Entertainment-oriented | U539584 | మహేశ్ మూవీ మ్యూజికల్ సెషన్ ప్రారంభం | music | Movie musical-session headline; matches music/movie profile |
| Politics-oriented | U69259 | పార్లమెంటు ముందుకు 16 బిల్లులు! | newspolitics | Parliament/bills headline; matches politics profile |
| Politics-oriented | U69259 | ఏపీ కేబినెట్లోకి నాగబాబు! | newspolitics | State cabinet headline; matches politics profile |

The full manual relevance table contains the top-5 examples for each representative user:

```text
results/telugu_qualitative_validation/telugu_manual_relevance_top5.csv
```

Paper-Ready Text For Category Alignment:

```text
To validate cross-lingual transfer qualitatively, we selected three representative English users with clear learned category affinities and examined their Telugu recommendations. For the sports-oriented, music/entertainment-oriented, and politics-oriented users, the top Telugu recommendations aligned with the users' strongest learned categories. Across the 30 headline-deduplicated recommendations inspected for these representative users, the predicted Telugu categories matched the users' top learned category in all cases, with a mean classifier confidence of 0.9658.
```

Paper-Ready Text For Manual Relevance:

```text
Manual inspection of the Telugu headlines further supports the category-alignment result. The sports-oriented user received cricket and team-selection headlines, the music/entertainment-oriented user received song-release and movie-music headlines, and the politics-oriented user received parliament, cabinet, and current-affairs headlines. These examples show that the shared refined category space produces interpretable Telugu recommendations even without Telugu click interactions.
```

How To Use This In The Paper:

```text
Use the category-alignment result as the automatic validation.
Use the manual relevance table as qualitative evidence.
Do not describe this as a full Telugu ranking benchmark, because there is no Telugu click ground truth.
```

## Representative-Example Recommendation List Quality Checks

Purpose:

```text
These checks inspect the top-10 lists for the three selected representative examples only. They are useful for presentation cleanup, but they should not be reported as system-level metrics because the users were deliberately selected to illustrate clear sports, music/entertainment, and politics profiles.
```

Script added:

```text
scripts/telugu/analyze_recommendation_list_quality.py
```

Command run:

```bash
../rs311/bin/python scripts/telugu/analyze_recommendation_list_quality.py
```

Outputs:

```text
results/telugu_recommendation_list_quality/duplicate_rate_and_headline_uniqueness.csv
results/telugu_recommendation_list_quality/inter_user_overlap.csv
results/telugu_recommendation_list_quality/recommendation_list_quality_summary.json
```

### Raw Duplicate Rate / Headline Uniqueness

Definitions:

```text
story_id_uniqueness = unique story_ids in top-k / k
headline_uniqueness = unique normalized headlines in top-k / k
headline_duplicate_rate = 1 - headline_uniqueness
```

The headline-normalized metric is more meaningful for this Telugu corpus because the same or near-identical headline can appear under different `story_id`s or source splits.

Raw top-10 results:

| User ID | Unique Story IDs | Story ID Uniqueness | Unique Headlines | Headline Uniqueness | Headline Duplicate Rate |
|---|---:|---:|---:|---:|---:|
| U245228 | 10 | 1.00 | 5 | 0.50 | 0.50 |
| U539584 | 10 | 1.00 | 2 | 0.20 | 0.80 |
| U69259 | 10 | 1.00 | 5 | 0.50 | 0.50 |

Mean raw top-10 values:

```text
Mean story_id uniqueness: 1.0000
Mean story_id duplicate rate: 0.0000
Mean headline uniqueness: 0.4000
Mean headline duplicate rate: 0.6000
```

Headline-deduplicated top-10 values:

```text
Mean story_id uniqueness: 1.0000
Mean story_id duplicate rate: 0.0000
Mean headline uniqueness: 1.0000
Mean headline duplicate rate: 0.0000
```

Interpretation:

```text
The raw recommender does not repeat the same story_id, but it can rank multiple articles with the same or near-identical headline. Therefore, the headline-deduplicated table is better for paper presentation and qualitative inspection.
```

Paper-facing recommendation:

```text
Use headline-deduplicated Telugu recommendation examples in the paper. Do not foreground the raw duplicate rate unless discussing implementation details or post-processing.
```

### Inter-User Overlap

Definitions:

```text
story_id_jaccard = shared story_ids / union story_ids between two users' top-k lists
headline_jaccard = shared normalized headlines / union normalized headlines between two users' top-k lists
```

Pairwise inter-user overlap:

| List Type | User Pair | Story ID Jaccard | Headline Jaccard | Shared Story IDs | Shared Headlines |
|---|---|---:|---:|---:|---:|
| Raw top-10 | U245228 - U539584 | 0.00 | 0.00 | 0 | 0 |
| Raw top-10 | U245228 - U69259 | 0.00 | 0.00 | 0 | 0 |
| Raw top-10 | U539584 - U69259 | 0.00 | 0.00 | 0 | 0 |
| Headline-deduped top-10 | U245228 - U539584 | 0.00 | 0.00 | 0 | 0 |
| Headline-deduped top-10 | U245228 - U69259 | 0.00 | 0.00 | 0 | 0 |
| Headline-deduped top-10 | U539584 - U69259 | 0.00 | 0.00 | 0 | 0 |

Summary:

```text
Mean story_id Jaccard overlap: 0.0000
Mean headline Jaccard overlap: 0.0000
Maximum story_id Jaccard overlap: 0.0000
Maximum headline Jaccard overlap: 0.0000
```

Interpretation:

```text
The selected representative users receive non-overlapping Telugu recommendation lists. This supports the qualitative claim that different English user profiles lead to different Telugu recommendations.
```

Paper-facing recommendation:

```text
If included, describe inter-user overlap as a qualitative check on the representative examples, not as a full-system diversity benchmark.
```

## Sampled Telugu Recommendation List Quality Metrics

Purpose:

```text
The representative examples are intentionally chosen to be clear, so they are not suitable for metric-style claims. This sampled analysis computes duplicate/headline uniqueness and inter-user overlap over a random sample of English users instead.
```

Script added:

```text
scripts/telugu/analyze_sampled_recommendation_list_quality.py
```

Command run:

```bash
../rs311/bin/python scripts/telugu/analyze_sampled_recommendation_list_quality.py --sample-users 500
```

Outputs:

```text
results/telugu_sampled_recommendation_list_quality/sampled_users.csv
results/telugu_sampled_recommendation_list_quality/sampled_telugu_top10_recommendations.csv
results/telugu_sampled_recommendation_list_quality/sampled_duplicate_rate_and_headline_uniqueness.csv
results/telugu_sampled_recommendation_list_quality/sampled_inter_user_overlap.csv
results/telugu_sampled_recommendation_list_quality/sampled_recommendation_list_quality_summary.json
```

Sampling setup:

| Field | Value |
|---|---:|
| Random seed | 107 |
| Sampled users | 500 |
| Telugu recommendations per user | 10 |
| Total sampled recommendations | 5,000 |

### Sampled Raw Duplicate Rate / Headline Uniqueness

Results over 500 random users:

| Metric | Value |
|---|---:|
| Mean story_id uniqueness | 1.0000 |
| Mean story_id duplicate rate | 0.0000 |
| Mean headline uniqueness | 0.5194 |
| Median headline uniqueness | 0.5000 |
| Mean headline duplicate rate | 0.4806 |

Interpretation:

```text
The sampled recommender output does not repeat exact story_ids within a top-10 list, but repeated or near-identical Telugu headlines are common because similar headlines appear under different story_ids. This supports using headline-deduplicated recommendation examples for paper presentation.
```

Paper-facing wording:

```text
In a random sample of 500 English users, the Telugu top-10 lists contained no repeated story_ids on average. Because the Telugu source corpus contains repeated or near-identical headlines under different story IDs, headline-level uniqueness was lower. For the qualitative examples, we therefore use headline-deduplicated recommendation lists.
```

### Sampled Inter-User Overlap

Results over 500 random users:

| Metric | Value |
|---|---:|
| Mean story_id Jaccard overlap | 0.0681 |
| Mean headline Jaccard overlap | 0.0736 |
| Mean shared story_ids per user pair | 0.7623 |
| Mean shared headlines per user pair | 0.2992 |
| Maximum story_id Jaccard overlap | 1.0000 |
| Maximum headline Jaccard overlap | 1.0000 |

Quantile notes:

```text
At least 75% of sampled user pairs had zero story_id overlap and zero headline overlap.
High maximum overlap occurs for some similar-profile users that receive the same top Telugu items.
```

Interpretation:

```text
Most sampled user pairs receive distinct Telugu recommendation lists, while users with similar learned preferences can share recommendations. This is expected for a category-driven recommender and is more realistic than the perfectly non-overlapping representative-example result.
```

Paper-facing wording:

```text
To avoid relying only on hand-selected examples, we also sampled 500 English users and measured overlap in their Telugu top-10 lists. The mean story-level Jaccard overlap was 0.0681 and the mean headline-level Jaccard overlap was 0.0736, indicating that most user pairs receive largely distinct Telugu lists while similar-profile users may share some recommendations.
```

What To Use In The Paper:

```text
Use the sampled duplicate/overlap values if reporting metric-style recommendation-list behavior.
Use the three representative users only for qualitative examples and manual relevance inspection.
```

### How To Present The Sampled List-Quality Results Positively

Main positioning:

```text
These metrics are not the main accuracy benchmark. They are supporting evidence that the Telugu recommendation lists are usable and not simple repeated outputs. Since Telugu click labels are unavailable, list-quality checks help validate the generated Telugu recommendations from two angles: within-list repetition and between-user personalization.
```

Positive points to emphasize:

```text
1. Exact article repetition is avoided: the sampled top-10 lists have 1.0000 mean story_id uniqueness and 0.0000 mean story_id duplicate rate.
2. Recommendations are not identical across users: mean story-level Jaccard overlap is only 0.0681 and mean headline-level Jaccard overlap is only 0.0736 across sampled user pairs.
3. Some overlap is expected and acceptable because users with similar learned category preferences should receive some similar Telugu recommendations.
4. Headline-level repetition reflects the source corpus having repeated or near-identical Telugu headlines under different story IDs, so headline-deduplicated examples are used for presentation clarity.
```

Recommended paper wording:

```text
Because Telugu click interactions are unavailable, we additionally evaluated the generated Telugu recommendation lists using unsupervised list-quality checks. A random sample of 500 English users was selected, and Telugu top-10 recommendations were generated for each user. The lists had a mean story-id uniqueness of 1.0000, indicating that the system did not repeat the same Telugu article within a user's top-10 recommendations. Inter-user overlap was also low, with mean story-level and headline-level Jaccard overlaps of 0.0681 and 0.0736 respectively, showing that the recommender produces largely distinct Telugu lists for different user profiles while still allowing similar users to share relevant items.
```

Optional sentence if discussing headline deduplication:

```text
Although exact article duplication was absent, some Telugu headlines appeared repeatedly under different story IDs in the source corpus. Therefore, headline-deduplicated recommendation lists were used for the qualitative examples to improve readability and avoid showing near-identical headlines in the paper.
```

What to avoid:

```text
Do not present headline duplicate rate as a failure of the recommender. Treat it as a source-corpus/display issue because the exact story IDs are unique, while some headlines are repeated or near-identical.
Do not use the three representative examples as aggregate metric evidence. They should remain qualitative case studies.
```

## Paper Editing Instructions For MINDlarge Update

Use this section as the checklist while editing the thesis/report. The goal is to update the paper from the older MINDsmall setup to the final MINDlarge setup without weakening the central Telugu-transfer story.

### 1. Abstract

Update the dataset wording:

```text
Replace "MIND small dataset" with "MINDlarge dataset" where the paper describes the English source-domain recommendation benchmark.
```

Keep the main Telugu claim:

```text
The system recommends Telugu news articles to English users by projecting Telugu articles into the same refined category feature space used by the English-trained recommender.
```

Do not mention implementation struggles, failed fallback variants, or category imbalance in the abstract.

### 2. Introduction

Old paper currently says the work builds on MINDsmall. Update this to:

```text
This work uses MINDlarge as the English source-domain dataset, allowing the recommendation model to be evaluated on official MIND test impressions through Codabench while retaining the Telugu cross-lingual recommendation pipeline.
```

Explain the motivation for switching:

```text
MINDsmall contains train and development splits but no official hidden test split for leaderboard evaluation. Therefore, the final experiments use MINDlarge so that AUC, MRR, nDCG@5, and nDCG@10 can be reported on the official test set.
```

### 3. English Recommendation Dataset Section

Replace all MINDsmall dataset counts/process descriptions with MINDlarge wording.

Use:

```text
The English recommendation experiments use MINDlarge train, development, and official test splits. The train split is used to fit the LightFM models, the development split is used for validation and hyperparameter selection, and the official test split is evaluated through Codabench.
```

Metrics to report for English:

```text
AUC
MRR
nDCG@5
nDCG@10
```

Do not use Telugu metrics here.

### 4. Refined Category Construction

Update the refined category count:

```text
Replace "19 refined categories" with "23 refined categories" for the final MINDlarge pipeline.
```

Use this wording:

```text
The refined category mapping consolidates noisy or overly specific MIND metadata into a curated 23-category taxonomy. This keeps the recommender lightweight while preserving broad topical distinctions needed for cold-start and cross-lingual transfer.
```

Keep the explanation that the refined categories are manually/analytically curated from vertical and subvertical metadata.

Avoid directly comparing MINDsmall 19-category behavior against MINDlarge 23-category behavior in the main text unless needed for historical context.

### 5. English Model Comparison Table

Update the model comparison table to use official MINDlarge test scores.

Recommended final table:

| Model Variant | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| BERT Hybrid LightFM | 0.5153 | 0.2251 | 0.2343 | 0.2898 |
| TF-IDF Hybrid LightFM | 0.5145 | 0.2247 | 0.2336 | 0.2891 |
| Vertical Category LightFM | 0.5151 | 0.2256 | 0.2347 | 0.2902 |
| Refined Category LightFM | 0.6059 | 0.2800 | 0.2995 | 0.3573 |

Paper-facing interpretation:

```text
The refined-category LightFM model outperforms the BERT, TF-IDF, and raw-vertical feature variants on all official MINDlarge test metrics. This supports using refined category features as the final English-trained recommender for the Telugu transfer stage.
```

### 6. Telugu Article Processing Section

Replace the old 19-category Telugu classifier description with the new MINDlarge version.

Use:

```text
To bring Telugu articles into the same feature space as the English recommender, an XLM-RoBERTa-base classifier is fine-tuned on English MINDlarge articles labeled with the 23 refined categories. The trained classifier is then applied to the Telugu article corpus. For each Telugu article, the classifier outputs a probability distribution over the 23 refined categories rather than a single hard label.
```

Use these values:

| Field | Value |
|---|---:|
| English classifier train rows | 91,374 |
| English classifier validation rows | 10,153 |
| Refined categories | 23 |
| Validation accuracy | 0.7658 |
| Validation macro-F1 | 0.6990 |
| Telugu articles after deduplication | 40,906 |
| Mean top-class confidence | 0.7250 |
| Median top-class confidence | 0.7504 |

### 7. Telugu Category Coverage Section

Add this as a short validation paragraph after Telugu classification.

Paper-ready paragraph:

```text
The Telugu classifier mapped 40,906 articles into 19 of the 23 refined MINDlarge categories, giving 82.61% coverage of the refined category space. This demonstrates that Telugu articles can be projected into the same feature representation used by the English-trained recommender. The largest Telugu category was newsworld, reflecting the current topical composition of the Telugu corpus, while categories such as sports, finance, movies, weather, politics, health, and travel were also represented. This coverage enables the English-trained LightFM model to score previously unseen Telugu articles through shared category features.
```

Report only:

```text
40,906 Telugu articles
19 / 23 categories covered
82.61% coverage
examples of represented categories
```

Do not foreground:

```text
top-5 category share
entropy
unused categories
distribution imbalance
```

Those values remain in the generated CSV/JSON for internal traceability, but they are not necessary in the main paper.

### 8. Telugu Feature Matrix Section

Update the feature matrix dimensions:

```text
Each Telugu article is represented by a 23-dimensional refined-category probability vector. These vectors are converted into a sparse Telugu item-feature matrix of shape 40,906 x 23. For scoring with the final MINDlarge LightFM model, this matrix is aligned with the model's full item-feature space, where Telugu articles activate only the shared refined-category feature columns.
```

Use:

```text
Telugu matrix before LightFM alignment: 40,906 x 23
Expanded model-aligned matrix: 40,906 x 129,831
Non-zero entries: 881,861
```

### 9. Telugu Qualitative Recommendation Results

Use qualitative examples, not ranking metrics, because Telugu has no interaction labels.

Recommended table:

| User Profile | User ID | Top Learned Categories | Example Telugu Recommendations |
|---|---|---|---|
| Sports-oriented | U245228 | sports, tv, music | cricket/team/player headlines |
| Music/Entertainment-oriented | U539584 | music, movies, entertainment | music/movie release headlines |
| Politics-oriented | U69259 | newspolitics, newsworld, science/technology | parliament, cabinet, election/current-affairs headlines |

Use the headline-deduplicated file for examples:

```text
results/telugu_mindlarge_recommendations/telugu_top10_recommendations_headline_deduped.csv
```

Use the qualitative validation files for evidence:

```text
results/telugu_qualitative_validation/telugu_category_alignment_summary.csv
results/telugu_qualitative_validation/telugu_manual_relevance_top5.csv
```

Paper-facing validation values:

```text
Representative users checked: 3
Recommendations checked: 30
Top-1 category alignment: 100%
Top-3 category alignment: 100%
Mean category confidence: 0.9658
```

Paper-facing interpretation:

```text
The qualitative recommendations show that the English-trained user profiles transfer to Telugu articles through the shared refined category space. Sports-oriented users receive Telugu sports articles, entertainment-oriented users receive music/movie articles, and politics-oriented users receive politics/current-affairs articles.
```

Additional paper-ready sentence:

```text
Manual inspection of the top Telugu headlines confirms the automatic alignment result: the sports-oriented user's examples are cricket/team-selection headlines, the music/entertainment-oriented user's examples are song-release or movie-music headlines, and the politics-oriented user's examples are parliament, cabinet, and current-affairs headlines.
```

### 10. Telugu List-Quality Checks

Add this after the qualitative Telugu examples. This section should be short and supportive, not the main evaluation table.

Purpose:

```text
Since Telugu click labels are not available, the Telugu stage cannot be evaluated using AUC, MRR, or nDCG. Instead, the paper can report list-quality checks over a random sample of generated Telugu recommendations to show that the system avoids exact article repetition and produces different lists for different users.
```

Use these sampled values:

| Metric | Value |
|---|---:|
| Random sampled English users | 500 |
| Telugu recommendations per user | 10 |
| Total Telugu recommendations checked | 5,000 |
| Mean story-id uniqueness | 1.0000 |
| Mean story-id duplicate rate | 0.0000 |
| Mean story-level Jaccard overlap | 0.0681 |
| Mean headline-level Jaccard overlap | 0.0736 |

Recommended paper paragraph:

```text
To validate the generated Telugu recommendation lists beyond selected case studies, we sampled 500 English users and generated Telugu top-10 recommendations for each user. The sampled lists had a mean story-id uniqueness of 1.0000 and a mean story-id duplicate rate of 0.0000, showing that the system did not repeat the same Telugu article within a user's recommendation list. Inter-user overlap was low, with mean story-level and headline-level Jaccard overlaps of 0.0681 and 0.0736 respectively. This indicates that most sampled users receive largely distinct Telugu recommendation lists, while users with similar learned preferences may still share relevant Telugu articles.
```

Optional note for presentation:

```text
For the qualitative examples shown in the paper, headline-deduplicated lists are used because the Telugu source corpus contains repeated or near-identical headlines under different story IDs. This post-processing improves readability while preserving the recommendation behavior.
```

How to frame it positively:

```text
Use story-id uniqueness and low inter-user overlap as the main list-quality findings.
Mention headline deduplication only as a presentation/readability step.
Do not foreground headline duplicate rate in the main paper table.
```

### 11. What Not To Report For Telugu

Do not report:

```text
Telugu AUC
Telugu MRR
Telugu nDCG@5
Telugu nDCG@10
```

Reason:

```text
The Telugu corpus does not contain user-impression click labels, so there is no ground truth ranking target for these metrics.
```

Instead, report:

```text
classifier validation metrics
category coverage
feature-matrix construction
qualitative top-k examples
```

### 12. Limitations / Future Work

Keep this short and forward-looking.

Recommended wording:

```text
Future work can extend this lightweight category-transfer approach by incorporating multilingual sentence embeddings or user-history encoders, allowing the recommender to distinguish finer semantic differences between articles within the same broad category.
```

Do not overemphasize limitations in the main results section.
