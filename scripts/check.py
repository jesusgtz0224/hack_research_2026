import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.io_utils import load_wikipron
from src.normalize import normalize_words, inventory, tie_bar_share

path = [config.WIKIPRON_DIR / "eng_latn_us_broad.tsv", config.WIKIPRON_DIR / "spa_latn_la_broad_filtered.tsv"]

import numpy as np, pandas as pd
D = np.load(config.PROCESSED / "dist_class_combined.npy")
meta = pd.read_parquet(config.PROCESSED / "metadata_aligned_class.parquet")
print(np.allclose(D, D.T), np.allclose(np.diag(D), 0))

i = meta.index[meta["iso"] == "spa"][0]
for j in np.argsort(D[i])[1:8]:
    print(meta.loc[j, "iso"], meta.loc[j, "Name"], round(D[i, j], 3))