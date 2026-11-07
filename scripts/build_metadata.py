from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_wikipron, load_glottolog, iso_from_filename

glottolog_path = PROJECT_ROOT / "data/raw/glottolog/languages.csv"
wikipron_path = [PROJECT_ROOT / "data/raw/wikipron/eng_latn_us_broad.tsv", PROJECT_ROOT / "data/raw/wikipron/spa_latn_la_broad_filtered.tsv"]

glot = load_glottolog(glottolog_path)

rows = []
for path in wikipron_path:
    words = load_wikipron(path)
    rows.append({"iso": iso_from_filename(path), "n_words": len(words)})
wp = pd.DataFrame(rows)

processed_dir = PROJECT_ROOT / "data/processed"

meta = wp.merge(glot, left_on="iso", right_on="ISO639P3code", how="left")
meta.to_parquet(processed_dir / "metadata.parquet")   # needs pyarrow
meta.to_csv(processed_dir / "metadata.csv", index=False)  # human-readable copy

print(meta)