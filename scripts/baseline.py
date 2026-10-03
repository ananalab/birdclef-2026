import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ajout du dossier racine au PYTHONPATH pour pouvoir importer src.* en lancement
# direct du script (Python interdit les noms de module commencant par un chiffre,
# donc on ne peut pas faire 'python -m scripts.baseline').
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio import load_audio, normalize_rms, window_audio
from src.config import (
    EXPERIMENTS_DIR,
    RANDOM_SEED,
    SUBMISSIONS_DIR,
    TRAIN_SOUNDSCAPES_DIR,
)
from src.dataset import (
    build_baseline_dataset,
    load_train_metadata,
    sample_files_per_species,
    select_top_species,
)
from src.features import extract_mfcc_baseline
from src.model import (
    macro_roc_auc,
    predict_proba,
    split_train_val,
    train_one_vs_rest,
)
from src.submission import format_submission


N_TOP_SPECIES = 10
N_FILES_PER_SPECIES = 50
N_SOUNDSCAPES_FOR_SUBMISSION = 10


def predict_on_soundscapes(
    models: dict,
    species: list[str],
    n_soundscapes: int,
) -> dict[str, np.ndarray]:
    """Prédit sur les n premiers soundscapes (par ordre alphabétique).

    Pour chaque soundscape on charge l'audio, on normalise, on découpe en
    fenêtres 5 s, on extrait les features et on prédit la probabilité de
    chaque espèce. Le résultat est un dict {filename_stem: array (n_windows, n_species)}
    consommable par format_submission.
    """
    soundscape_files = sorted(TRAIN_SOUNDSCAPES_DIR.glob("*.ogg"))[:n_soundscapes]
    predictions: dict[str, np.ndarray] = {}

    for path in tqdm(soundscape_files, desc="Inference soundscapes"):
        signal = normalize_rms(load_audio(path))
        windows = window_audio(signal)

        # Construction des features pour chaque fenêtre du fichier.
        feats_list = [extract_mfcc_baseline(w) for w in windows]
        X_file = pd.DataFrame(feats_list)

        probs = predict_proba(models, X_file, species)
        predictions[path.stem] = probs

    return predictions


def main() -> None:
    t_start = time.time()
    timestamp = datetime.now().isoformat(timespec="seconds")

    print(f"[1/5] Chargement metadata et selection top {N_TOP_SPECIES} especes")
    df = load_train_metadata()
    top_species = select_top_species(df, n=N_TOP_SPECIES)
    print(f"      Especes retenues: {top_species}")

    print(f"[2/5] Echantillonnage {N_FILES_PER_SPECIES} fichiers par espece")
    sample = sample_files_per_species(
        df, top_species, n_files=N_FILES_PER_SPECIES, seed=RANDOM_SEED
    )
    print(f"      Total fichiers selectionnes: {len(sample)}")

    print("[3/5] Extraction des features (26 MFCC par fichier)")
    X, y, groups = build_baseline_dataset(sample)
    print(f"      Dataset construit: X={X.shape}, y={y.shape}")

    print("[4/5] Entrainement LightGBM one-vs-rest (10 modeles binaires)")
    train_idx, val_idx = split_train_val(X, y, groups)
    models = train_one_vs_rest(X.iloc[train_idx], y[train_idx], top_species)
    val_probs = predict_proba(models, X.iloc[val_idx], top_species)
    macro, per_species = macro_roc_auc(y[val_idx], val_probs, top_species)
    print(f"      Macro ROC-AUC val: {macro:.4f}")
    for code, score in per_species.items():
        score_str = f"{score:.3f}" if not np.isnan(score) else "n/a"
        print(f"        {code}: {score_str}")

    print(f"[5/5] Inference sur {N_SOUNDSCAPES_FOR_SUBMISSION} soundscapes et ecriture submission.csv")
    predictions = predict_on_soundscapes(models, top_species, N_SOUNDSCAPES_FOR_SUBMISSION)
    submission_path = SUBMISSIONS_DIR / "baseline.csv"
    df_sub = format_submission(predictions, top_species, submission_path)
    print(f"      submission.csv ecrit: {submission_path} ({df_sub.shape})")

    duration = time.time() - t_start
    log = {
        "experiment_id": "baseline",
        "timestamp": timestamp,
        "duration_seconds": round(duration, 1),
        "config": {
            "n_top_species": N_TOP_SPECIES,
            "n_files_per_species": N_FILES_PER_SPECIES,
            "n_soundscapes_for_submission": N_SOUNDSCAPES_FOR_SUBMISSION,
            "random_seed": RANDOM_SEED,
            "n_features": 26,
            "features_type": "mfcc_baseline (13 MFCC x mean+std)",
            "model": "LightGBM one-vs-rest, params par defaut",
        },
        "dataset": {
            "top_species": top_species,
            "n_samples_total": int(len(X)),
            "n_samples_train": int(len(train_idx)),
            "n_samples_val": int(len(val_idx)),
        },
        "scores": {
            "macro_roc_auc_val": round(macro, 4),
            "per_species_roc_auc": {
                code: (round(s, 4) if not np.isnan(s) else None)
                for code, s in per_species.items()
            },
        },
        "outputs": {
            "submission_csv": str(submission_path.relative_to(Path.cwd())),
        },
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "baseline.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n=== Termine en {duration:.1f}s ===")
    print(f"Log: {log_path}")
    print(f"Submission: {submission_path}")


if __name__ == "__main__":
    main()
