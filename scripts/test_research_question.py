import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import sys

SYSTEM_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(SYSTEM_ROOT))

from src import config
from src.stats_test import mantel, partial_mantel

ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=["class", "segment"], required=True)
ap.add_argument("--perms", type=int, default=999)
args = ap.parse_args()

P = config.PROCESSED
geo = np.load(P / f"dist_geo_{args.level}.npy")
fam = 1 - np.load(P / f"same_family_{args.level}.npy")   # 0 = same family, 1 = different

names = ["combined"] + sorted(
    f.stem.split("_", 2)[2] for f in P.glob(f"dist_{args.level}_*.npy")
    if not f.stem.endswith("combined"))

rows = []
for name in names:
    D = np.load(P / f"dist_{args.level}_{name}.npy")
    r_g, p_g = mantel(D, geo, args.perms)
    r_f, p_f = mantel(D, fam, args.perms)
    r_fg, p_fg = partial_mantel(D, fam, geo, args.perms)   # family, controlling for geography
    r_gf, p_gf = partial_mantel(D, geo, fam, args.perms)   # geography, controlling for family
    rows.append(dict(block=name, r_geo=r_g, p_geo=p_g, r_fam=r_f, p_fam=p_f,
                     r_fam_given_geo=r_fg, p_fam_given_geo=p_fg,
                     r_geo_given_fam=r_gf, p_geo_given_fam=p_gf))
    print(name, "done")

out = pd.DataFrame(rows).round(3)
out.to_csv(config.RESULTS / f"tests_{args.level}.csv", index=False)
print(out.to_string(index=False))


# plain-language version of the family effect, on the combined matrix
D = np.load(P / f"dist_{args.level}_combined.npy")
iu = np.triu_indices_from(D, k=1)
same = fam[iu] == 0
print("\nmean distance, same family:", D[iu][same].mean().round(3),
      "| different family:", D[iu][~same].mean().round(3))

d_same, d_diff = D[iu][same], D[iu][~same]

# probability that a random same-family pair is closer than a random different-family pair
u = mannwhitneyu(d_same, d_diff).statistic
print("P(same-family pair closer):", round(1 - u / (len(d_same) * len(d_diff)), 3))

# within-family mean distance, one row per family (3+ members)
meta = pd.read_parquet(P / f"metadata_aligned_{args.level}.parquet")
rows = []
for f, grp in meta.groupby("family"):
    if len(grp) >= 3:
        idx = grp.index.to_numpy()
        sub = D[np.ix_(idx, idx)]
        rows.append((f, len(idx), sub[np.triu_indices_from(sub, k=1)].mean()))
print(pd.DataFrame(rows, columns=["family", "n", "mean_dist"])
        .sort_values("mean_dist").round(3).to_string(index=False))