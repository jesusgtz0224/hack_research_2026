import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from src import config
from src.distances import (js_matrix, combine, haversine_matrix,
                           same_family_matrix)

ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=["class", "segment"], required=True)
args = ap.parse_args()

prof = np.load(config.PROCESSED / f"profiles_{args.level}.npz")
isos = prof["isos"]
blocks = [k[2:] for k in prof.files if k.startswith("X_")]

meta = pd.read_parquet(config.PROCESSED / "metadata.parquet")
assert meta["iso"].is_unique, "duplicate ISO codes in metadata: resolve before distances"
meta = meta.set_index("iso").loc[isos].reset_index()

# keep languages with coordinates; isolates become their own family
keep = meta["Latitude"].notna() & meta["Longitude"].notna()
meta = meta[keep].reset_index(drop=True)
idx = np.flatnonzero(keep.to_numpy())
meta["family"] = meta["Family_ID"].where(meta["Family_ID"] != "", meta["Glottocode"])

# phonotactic distances: one matrix per block, plus the combined one
mats = {}
for b in blocks:
    mats[b] = js_matrix(prof[f"X_{b}"][idx])
    np.save(config.PROCESSED / f"dist_{args.level}_{b}.npy", mats[b])
combined = combine(list(mats.values()))
np.save(config.PROCESSED / f"dist_{args.level}_combined.npy", combined)

# ground truth matrices (the same for both levels)
geo = haversine_matrix(meta["Latitude"].to_numpy(float), meta["Longitude"].to_numpy(float))
fam = same_family_matrix(meta["family"])
np.save(config.PROCESSED / f"dist_geo_{args.level}.npy", geo)
np.save(config.PROCESSED / f"same_family_{args.level}.npy", fam)
meta.to_parquet(config.PROCESSED / f"metadata_aligned_{args.level}.parquet")

print(len(meta), "languages;", len(blocks), "blocks:", blocks)