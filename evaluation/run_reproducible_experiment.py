"""
Helper to verify required authoritative inputs and run the experiment in a reproducible way.
Usage:
    python evaluation/run_reproducible_experiment.py

This script checks for the frozen corpus and final ground-truth file, prints checksums,
and instructs or executes the main experiment runner.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CORPUS = Path("data/raw/ogd_metadata_20260306_183841.json")
GT = Path("evaluation/ground_truth_final.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    missing = []
    if not CORPUS.exists():
        missing.append(str(CORPUS))
    if not GT.exists():
        missing.append(str(GT))

    if missing:
        print("Missing authoritative input files:")
        for m in missing:
            print(" - ", m)
        print("\nPlease generate the missing files (see evaluation/build_ground_truth.py) before running the experiment.")
        sys.exit(2)

    semantic_cache = Path("evaluation/embeddings_cache/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2")
    if not semantic_cache.exists():
        print("Missing required cached semantic model artifacts:")
        print(f" - {semantic_cache}")
        print("\nRestore the cached sentence-transformers model under evaluation/embeddings_cache before running the experiment.")
        sys.exit(2)

    print("Corpus file:", CORPUS)
    print(" - size:", CORPUS.stat().st_size, "bytes")
    print(" - sha256:", sha256(CORPUS))
    print()
    print("Ground-truth file:", GT)
    print(" - size:", GT.stat().st_size, "bytes")
    print(" - sha256:", sha256(GT))
    print()

    # Run the experiment
    print("Running evaluation/experiment_runner.py (this will fail if inputs mismatch)...")
    completed = subprocess.run([sys.executable, "evaluation/experiment_runner.py"], check=False)
    if completed.returncode != 0:
        print(f"Evaluation run exited with code {completed.returncode}.")
        sys.exit(completed.returncode)
