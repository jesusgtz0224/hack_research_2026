from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.io_utils import load_wikipron, load_glottolog, iso_from_filename



glot = load_glottolog(config.GLOTTOLOG_CSV)

rows = []
for path in config.WIKIPRON_DIR.glob("*.tsv"):
    words = load_wikipron(path)
    rows.append({"iso": iso_from_filename(path), "n_words": len(words), "path": str(path)})
wp = pd.DataFrame(rows)
wp = wp.sort_values("n_words", ascending=False).drop_duplicates("iso")

processed_dir = config.PROCESSED

meta = wp.merge(glot, left_on="iso", right_on="ISO639P3code", how="left")
meta.to_parquet(processed_dir / "metadata.parquet")   # needs pyarrow
meta.to_csv(processed_dir / "metadata.csv", index=False)  # human-readable copy

print(meta)