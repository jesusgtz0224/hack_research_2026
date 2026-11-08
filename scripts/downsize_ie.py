import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYSTEM_ROOT))

from src import config
from src.stats_test import mantel, partial_mantel

ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=["class", "segment"], required=True)
ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30])
ap.add_argument("--draws", type=int, default=200)
ap.add_argument("--seed", type=int, default=config.SEED)
args = ap.parse_args()

P = config.PROCESSED
meta = pd.read_parquet(P / f"metadata_aligned_{args.level}.parquet")
geo = np.load(P / f"dist_geo_{args.level}.npy")
fam = 1 - np.load(P / f"same_family_{args.level}.npy")
blocks = ["combined"] + sorted(
    f.stem.split("_", 2)[2] for f in P.glob(f"dist_{args.level}_*.npy")
    if not f.stem.endswith("combined"))
mats = {b: np.load(P / f"dist_{args.level}_{b}.npy") for b in blocks}

ie = np.flatnonzero((meta["family"] == "indo1319").to_numpy())
other = np.flatnonzero((meta["family"] != "indo1319").to_numpy())
print(f"{len(ie)} Indo-European, {len(other)} other languages")


def stats(keep, D):
    """Correlations only; n_perm=0 skips the (slow) permutation p-values."""
    ix = np.ix_(keep, keep)
    d, g, f = D[ix], geo[ix], fam[ix]
    return dict(
        r_geo=mantel(d, g, 0)[0],
        r_fam=mantel(d, f, 0)[0],
        r_fam_given_geo=partial_mantel(d, f, g, 0)[0],
        r_geo_given_fam=partial_mantel(d, g, f, 0)[0],
    )


rng = np.random.default_rng(args.seed)
rows = []
for k in [len(ie)] + args.sizes + [0]:          # full sample, downsampled, no IE
    n_draws = 1 if k in (len(ie), 0) else args.draws
    for _ in range(n_draws):
        pick = ie if k == len(ie) else rng.choice(ie, size=k, replace=False)
        keep = np.sort(np.concatenate([other, pick]))
        for b in blocks:
            rows.append(dict(ie_kept=k, block=b, **stats(keep, mats[b])))
    print("ie_kept =", k, "done")

res = pd.DataFrame(rows)
cols = ["r_geo", "r_fam", "r_fam_given_geo", "r_geo_given_fam"]
summary = (res.groupby(["block", "ie_kept"])[cols]
              .agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)]))
summary.columns = [f"{c}_{s}" for c, s in
                   zip([c for c in cols for _ in range(3)], ["mean", "lo", "hi"] * len(cols))]
summary = summary.round(3).reset_index()
summary.to_csv(config.RESULTS / f"downsample_ie_{args.level}.csv", index=False)
print(summary[summary["block"] == "combined"].to_string(index=False))