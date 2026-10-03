import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.audio import load_audio, normalize_rms
from src.config import DATA_PROCESSED, SAMPLE_RATE, TRAIN_AUDIO_DIR, WINDOW_SAMPLES
from src.features import extract_features

PERCH_META = DATA_PROCESSED / "embeddings_perch_v8_focal_full_meta.csv"
SHARDS_DIR = DATA_PROCESSED / "_shards_features_focal"
OUTPUT = DATA_PROCESSED / "features_focal_full_w3_all.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-windows", type=int, default=3,
                   help="Fenetres max par fichier (defaut: 3, pour aligner Perch sous-echantillonne).")
    p.add_argument("--max-files-per-species", type=int, default=None,
                   help="Plafond de fichiers par espece (defaut: aucun). "
                        "Les especes rares sous le plafond sont gardees entieres ; "
                        "les communes sont coupees. Rendements decroissants au-dela de ~100.")
    p.add_argument("--families", nargs="+", default=None,
                   help="Familles de features (defaut: toutes les 6).")
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def get_window(audio: np.ndarray, win_idx: int) -> np.ndarray:
    """Fenetre win_idx avec la meme logique que le kernel Perch (zero-pad)."""
    start = win_idx * WINDOW_SAMPLES
    end = start + WINDOW_SAMPLES
    if start >= len(audio):
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    if end > len(audio):
        chunk = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        chunk[: len(audio) - start] = audio[start:]
        return chunk
    return audio[start:end].astype(np.float32)


def extract_one_file(filename: str, label: str, win_indices: list[int],
                     families) -> list[dict]:
    """Charge un fichier une fois, extrait les features pour les fenetres demandees."""
    try:
        signal = normalize_rms(load_audio(TRAIN_AUDIO_DIR / filename))
    except Exception:
        return []
    rows = []
    for w_idx in win_indices:
        try:
            feats = extract_features(get_window(signal, w_idx), families=families)
        except Exception:
            continue
        feats["filename"] = filename
        feats["primary_label"] = label
        feats["window_idx"] = w_idx
        rows.append(feats)
    return rows


def main() -> None:
    args = parse_args()
    t_start = time.time()

    print(f"[1/3] Lecture de la liste de fichiers depuis {PERCH_META.name}")
    meta = pd.read_csv(PERCH_META)
    meta = meta[meta["window_idx"] < args.n_windows]
    # {(filename, label): [window_idx,...]}
    grouped = meta.groupby(["primary_label", "filename"])["window_idx"].apply(list)
    species_list = sorted(meta["primary_label"].unique())
    n_files = meta[["filename"]].drop_duplicates().shape[0]
    print(f"      {len(species_list)} especes, {n_files} fichiers, {len(meta)} fenetres a extraire")

    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Extraction par espece (checkpoint), n_jobs={args.n_jobs}")
    for s_i, species in enumerate(species_list, start=1):
        shard_path = SHARDS_DIR / f"{species}.parquet"
        if shard_path.exists():
            print(f"  ({s_i}/{len(species_list)}) {species} : deja fait, saute")
            continue

        files_of_species = grouped.loc[species]  # Series indexee par filename
        if args.max_files_per_species is not None and len(files_of_species) > args.max_files_per_species:
            # Plafond : on garde les N premiers fichiers (ordre trie deterministe).
            kept = sorted(files_of_species.index)[: args.max_files_per_species]
            files_of_species = files_of_species.loc[kept]
        t_sp = time.time()
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(extract_one_file)(fn, species, win_idx, args.families)
            for fn, win_idx in files_of_species.items()
        )
        rows = [r for sub in results for r in sub]
        if rows:
            pd.DataFrame(rows).to_parquet(shard_path, index=False, compression="snappy")
        elapsed = time.time() - t_sp
        total_elapsed = (time.time() - t_start) / 60
        print(f"  ({s_i}/{len(species_list)}) {species} : {len(rows)} fenetres, "
              f"{elapsed:.0f}s | total {total_elapsed:.0f} min")

    print("[3/3] Concatenation des shards en un parquet unique")
    shards = sorted(SHARDS_DIR.glob("*.parquet"))
    parts = [pd.read_parquet(s) for s in shards]
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(OUTPUT, index=False, compression="snappy")
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    n_features = full.shape[1] - 3
    duration = (time.time() - t_start) / 60
    print(f"      Ecrit : {OUTPUT} ({size_mb:.1f} MB), shape {full.shape}, {n_features} features")
    print(f"\n=== Termine en {duration:.0f} min ===")


if __name__ == "__main__":
    main()
