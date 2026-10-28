"""
Master Runner Implementasi Algoritma K-Prototypes Clustering
Tugas Pertemuan 9 (Clustering) - Kelompok 6 (Genap) - Person 2 (ISA)

Menjalankan pipeline clustering untuk masing-masing dataset secara terpisah:
- Heart Disease Dataset  : code/03_kprototypes_heart.py -> output/heart/
- Mall Customers Dataset : code/04_kprototypes_mall.py  -> output/mall/
"""

import sys
import argparse
from pathlib import Path

# Menambahkan folder code ke sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "code"))

import importlib

def main():
    parser = argparse.ArgumentParser(description="Runner K-Prototypes Clustering TM 9")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "heart", "mall"],
        help="Pilih dataset: 'heart', 'mall', atau 'all' (default: 'all')",
    )
    args = parser.parse_args()

    if args.dataset in ["heart", "all"]:
        heart_module = importlib.import_module("03_kprototypes_heart")
        heart_module.run()
        print()

    if args.dataset in ["mall", "all"]:
        mall_module = importlib.import_module("04_kprototypes_mall")
        mall_module.run()
        print()


if __name__ == "__main__":
    main()
