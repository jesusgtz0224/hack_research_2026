from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import load_wikipron

for path in [
    PROJECT_ROOT / "data/raw/wikipron/spa_latn_la_broad_filtered.tsv",
    PROJECT_ROOT / "data/raw/wikipron/eng_latn_us_broad.tsv",
]:
    words = load_wikipron(path)
    top = Counter(seg for w in words for seg in w).most_common(10)
    print(path, len(words), "words")
    print(top, "\n")