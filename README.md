# BirdCLEF+ 2026: audio classification without a trained neural network

Machine Learning project (M1) on the Kaggle competition BirdCLEF+ 2026: identify species (birds, amphibians, insects...) from the Pantanal in audio recordings, on 5-second windows.

Project constraint: no neural network training. Two approaches are compared:

- **Strict track**: handcrafted audio features (temporal, spectral, MFCC, mel, bioacoustic, chroma: ~969 features) + one-vs-rest LightGBM.
- **Perch track**: embeddings from Google's pretrained Perch model, used frozen, + L2 logistic regression or LightGBM.

The full report (in French) is in `RAPPORT_ML.pdf`.

## Main results

Macro ROC-AUC. CV = 5-fold StratifiedGroupKFold grouped by file. Held-out = annotated soundscapes (739 windows, 47 species with positives).

| Experiment | Strict | Perch |
|---|---|---|
| CV, 10 species (Optuna-tuned LightGBM / L2 LogReg) | 0.887 | 0.966 |
| CV, 206 species | 0.842 | 0.953 |
| Held-out soundscapes | 0.610 | 0.746 |
| Held-out + pseudo-labelling | 0.656 | 0.714 |

Perch embeddings win everywhere. The gap between CV and held-out scores comes from the domain shift between focal recordings and soundscapes.

## Layout

```
src/              reusable code (audio, features, validation, models, tuning)
scripts/          one script per experiment, each writes its result to experiments/
kaggle_kernels/   Kaggle kernels to extract Perch embeddings (GPU)
experiments/      raw results as JSON
notebooks/eda.ipynb   exploratory analysis
report/figures/   report figures
submissions/      submission file
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The dataset (~15 GB) is not versioned. Download it from Kaggle into `birdclef-2026/` at the root:

```bash
kaggle competitions download -c birdclef-2026
unzip birdclef-2026.zip -d birdclef-2026
```

Trained models and intermediate features (`models/`, `data/processed/`) are not versioned either; rerun the scripts to regenerate them.

## Reproduce

Suggested order:

```bash
python scripts/baseline.py                  # 10-species baseline
python scripts/extract_features.py          # handcrafted features
python scripts/cv_lgbm.py                   # LightGBM CV
python scripts/optuna_lgbm.py               # tuning
python scripts/benchmark_models.py          # XGBoost, RandomForest, LogReg
python scripts/train_on_embeddings.py       # Perch track (embeddings from kaggle_kernels/)
python scripts/full_volume_comparison.py    # 206 species
python scripts/heldout_eval.py              # soundscape evaluation
python scripts/pseudo_label.py
python scripts/metrics.py                   # mAP, precision, recall, F1
```
