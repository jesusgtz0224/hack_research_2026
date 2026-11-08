import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path
import sys

SYSTEM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYSTEM_ROOT))

from src import config
from src.io_utils import load_glottolog
from src.stats_test import mantel, partial_mantel

ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=["class", "segment"], required=True)
ap.add_argument("--perms", type=int, default=999)
ap.add_argument("--seed", type=int, default=config.SEED)
args = ap.parse_args()

P = config.PROCESSED
meta = pd.read_parquet(P / f"metadata_aligned_{args.level}.parquet")
geo = np.load(P / f"dist_geo_{args.level}.npy")
same = np.load(P / f"same_family_{args.level}.npy")          # 1 = same family
blocks = ["combined"] + sorted(
    f.stem.split("_", 2)[2] for f in P.glob(f"dist_{args.level}_*.npy")
    if not f.stem.endswith("combined"))
mats = {b: np.load(P / f"dist_{args.level}_{b}.npy") for b in blocks}

glot = load_glottolog(config.GLOTTOLOG_CSV)
fam_name = dict(zip(glot["ID"], glot["Name"]))

rng = np.random.default_rng(args.seed)
n = len(meta)
iu = np.triu_indices(n, k=1)
is_ie = (meta["family"] == "indo1319").to_numpy()
same_u = same[iu] == 1


def pair_in(sel):
    """True for each upper-triangle pair whose two languages are both in sel."""
    return sel[iu[0]] & sel[iu[1]]


def masked_corr(D, G, mask, sel, n_perm):
    """Spearman correlation of D and G over the pairs in mask. The p-value comes from
    shuffling geography among the languages in sel, with the pair mask held fixed."""
    d = rankdata(D[iu][mask])

    def r_of(Gm):
        return np.corrcoef(d, rankdata(Gm[iu][mask]))[0, 1]

    r = r_of(G)
    if n_perm == 0:
        return r, np.nan
    idx = np.flatnonzero(sel)
    hits = 0
    for _ in range(n_perm):
        p = np.arange(n)
        p[idx] = rng.permutation(idx)
        if abs(r_of(G[np.ix_(p, p)])) >= abs(r):
            hits += 1
    return r, (hits + 1) / (n_perm + 1)


# ---- 1. cross-family pairs: does geography predict sound among unrelated languages? ----
everyone = np.ones(n, dtype=bool)
subsets = {
    "cross-family, all languages": (~same_u, everyone),
    "cross-family, no Indo-European": (~same_u & pair_in(~is_ie), ~is_ie),
}
rows = []
for sname, (mask, sel) in subsets.items():
    for b in blocks:
        r, p = masked_corr(mats[b], geo, mask, sel, args.perms)
        rows.append(dict(test=sname, block=b, r=r, p=p, n_pairs=int(mask.sum())))
cross = pd.DataFrame(rows).round(3)
cross.to_csv(config.RESULTS / f"cross_family_{args.level}.csv", index=False)
print("\n== Sound vs geography among UNRELATED pairs ==")
print(cross.to_string(index=False))

# ---- 2. within-family pairs: does sound drift apart as relatives spread out? ----
rows = []
for f, grp in meta.groupby("family"):
    if len(grp) >= 8:
        ix = np.ix_(grp.index.to_numpy(), grp.index.to_numpy())
        r, p = mantel(mats["combined"][ix], geo[ix], args.perms, args.seed)
        rows.append(dict(family=fam_name.get(f, f), n=len(grp), r=r, p=p))
within = pd.DataFrame(rows).round(3)
within.to_csv(config.RESULTS / f"within_family_{args.level}.csv", index=False)
print("\n== Sound vs geography WITHIN each family (8+ members) ==")
print(within.to_string(index=False))

# ---- 3. permutation test for the family effect once Indo-European is removed ----
keep = np.flatnonzero(~is_ie)
ix = np.ix_(keep, keep)
fam_diff = 1 - same
rows = []
for b in blocks:
    D = mats[b][ix]
    r_f, p_f = partial_mantel(D, fam_diff[ix], geo[ix], args.perms, args.seed)
    r_g, p_g = partial_mantel(D, geo[ix], fam_diff[ix], args.perms, args.seed)
    rows.append(dict(block=b, r_fam_given_geo=r_f, p_fam_given_geo=p_f,
                     r_geo_given_fam=r_g, p_geo_given_fam=p_g))
noie = pd.DataFrame(rows).round(3)
noie.to_csv(config.RESULTS / f"no_ie_partial_{args.level}.csv", index=False)
print(f"\n== Partial Mantel without Indo-European ({len(keep)} languages) ==")
print(noie.to_string(index=False))

# ---- 4. unrelated pairs by distance: is there a contact gradient? ----
D = mats["combined"]
bins = [0, 1000, 3000, 6000, 10000, np.inf]
labels = ["<1000 km", "1000-3000", "3000-6000", "6000-10000", ">10000"]
cut = pd.cut(pd.Series(geo[iu]), bins=bins, labels=labels, include_lowest=True)
tab = []
for sname, (mask, _) in subsets.items():
    g = pd.DataFrame({"bin": cut[mask].to_numpy(), "d": D[iu][mask]})
    t = g.groupby("bin", observed=True)["d"].agg(["mean", "count"]).round(3)
    t.insert(0, "subset", sname)
    tab.append(t.reset_index())
print("\n== Mean sound distance of unrelated pairs, by geographic distance ==")
print(pd.concat(tab).to_string(index=False))

# ---- 5. most similar-sounding unrelated pairs that are far apart ----
m = D.sum(axis=1) / (len(D) - 1)
Dn = D / np.sqrt(np.outer(m, m))

EXCLUDE = {"Sanskrit", "Ionic-Attic Ancient Greek", "Egyptian (Ancient)", "Esperanto"}
ok = ~meta["Name"].isin(EXCLUDE).to_numpy()

far = (geo[iu] > 5000) & ~same_u & ok[iu[0]] & ok[iu[1]]
order = np.argsort(np.where(far, Dn[iu], np.inf))[:15]
print("\n== Closest-sounding unrelated pairs >5000 km apart (hubness-adjusted) ==")
for k in order:
    i, j = iu[0][k], iu[1][k]
    print(f"{Dn[i, j]:.3f} (raw {D[i, j]:.3f})  {meta['Name'][i]} "
          f"({fam_name.get(meta['family'][i], '?')})  <->  {meta['Name'][j]} "
          f"({fam_name.get(meta['family'][j], '?')})  {geo[i, j]:.0f} km")