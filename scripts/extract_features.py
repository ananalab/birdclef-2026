import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, RANDOM_SEED
from src.dataset import (
    build_dataset_rich,
    load_train_metadata,
    sample_files_per_species,
    select_top_species,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--n-species", type=int, default=10,
                       help="Nombre d'espèces top à utiliser (defaut: 10).")
    group.add_argument("--all", action="store_true",
                       help="Toutes les espèces, tous les fichiers.")
    p.add_argument("--n-files", type=int, default=50,
                   help="Fichiers par espèce (defaut: 50). Ignoré si --all.")
    p.add_argument("--n-windows", type=int, default=3,
                   help="Fenêtres 5s à extraire par fichier (defaut: 3).")
    p.add_argument("--families", nargs="+", default=None,
                   help="Familles à calculer (defaut: toutes). "
                        "Choix: temporal spectral mfcc mel bioacoustic pitch_chroma.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Processus paralleles (defaut: -1 = tous les coeurs).")
    p.add_argument("--output", type=str, default=None,
                   help="Chemin du parquet de sortie (defaut: data/processed/features_<tag>.parquet).")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/3] Chargement metadata et selection")
    df = load_train_metadata()
    if args.all:
        sample = df.copy()
        tag = "all"
        print(f"      Mode --all : {len(sample)} fichiers, {sample['primary_label'].nunique()} especes")
    else:
        top = select_top_species(df, n=args.n_species)
        sample = sample_files_per_species(df, top, n_files=args.n_files, seed=args.seed)
        tag = f"top{args.n_species}_n{args.n_files}"
        print(f"      Top {args.n_species} especes, {args.n_files} fichiers/espece : {len(sample)} fichiers")

    families_str = ",".join(args.families) if args.families else "all"
    print(f"[2/3] Extraction features ({families_str}), {args.n_windows} fenetres/fichier, n_jobs={args.n_jobs}")
    t0 = time.time()
    features_df = build_dataset_rich(
        sample,
        n_windows=args.n_windows,
        families=args.families,
        n_jobs=args.n_jobs,
    )
    duration = time.time() - t0
    print(f"      Termine en {duration:.1f}s : {features_df.shape}")

    # Calcule combien de colonnes sont des features (= toutes sauf les 3 meta).
    n_features = features_df.shape[1] - 3
    print(f"      {n_features} features par ligne (+ 3 colonnes meta: filename, primary_label, window_idx)")

    print("[3/3] Sauvegarde parquet")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    fam_tag = "all" if args.families is None else "_".join(args.families)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = DATA_PROCESSED / f"features_{tag}_w{args.n_windows}_{fam_tag}.parquet"
    features_df.to_parquet(out_path, index=False, compression="snappy")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"      Ecrit: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
