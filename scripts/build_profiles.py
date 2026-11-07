import argparse
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.io_utils import load_wikipron
from src.normalize import normalize_words
from src.profiles import (build_profile, subsample, class_vocabs,
                          counters_to_matrix)

ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=["class", "segment"], required=True)
args = ap.parse_args()

meta = pd.read_parquet(config.PROCESSED / "metadata.parquet")
meta = meta[meta["n_words"] >= config.MIN_WORDS]

profiles, isos = [], []
for _, row in tqdm(meta.iterrows(), total=len(meta)):
    words = load_wikipron(row["path"])
    words = normalize_words(words, affricates=config.AFFRICATES)
    words = subsample(words, config.N_WORDS, config.SEED)
    profiles.append(build_profile(words, args.level))
    isos.append(row["iso"])

# vocabularies: fixed for classes, learned from the data for segments
if args.level == "class":
    vocabs = class_vocabs(config.CLASSES)
else:
    vocabs = {b: sorted({k for p in profiles for k in p[b]}, key=str)
              for b in ["unigram", "bigram", "start", "end",
                        "clust_init", "clust_med", "clust_fin"]}

# keep only the most common word shapes overall
shape_total = sum((p["shape"] for p in profiles), Counter())
vocabs["shape"] = [s for s, _ in shape_total.most_common(config.N_SHAPES)]

blocks = {b: counters_to_matrix([p[b] for p in profiles], v)
          for b, v in vocabs.items()}
np.savez(config.PROCESSED / f"profiles_{args.level}.npz", isos=np.array(isos),
         **{f"X_{b}": M for b, M in blocks.items()})