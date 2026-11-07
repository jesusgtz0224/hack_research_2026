from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

WIKIPRON_DIR = RAW / "wikipron"
GLOTTOLOG_CSV = RAW / "glottolog" / "languages.csv"
WIKIPRON_PATTERN = "*_broad.tsv"   # phonemic files only

for d in (PROCESSED, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# ---- reproducibility ----
SEED = 42

# ---- which languages to keep ----
MIN_WORDS = 1000      # drop languages with fewer words than this (after dedup)

# ---- profile building ----
N_WORDS = 1000        # subsample every language to this many words (None = use all)
AFFRICATES = "keep"   # "keep" or "split" (see normalize.py)
N_SHAPES = 50         # number of most common CV word shapes kept in the shape block
SMOOTHING = 0.5       # add-alpha smoothing used in counters_to_matrix

# ---- segment classes (must match classes.py) ----
if AFFRICATES == "keep":
    CLASSES = ["V", "P", "A", "F", "N", "L", "G"]
else:
    CLASSES = ["V", "P", "F", "N", "L", "G"]

# ---- distances and statistics ----
N_PERMUTATIONS = 9999   # for Mantel / partial Mantel tests
MDS_DIMS = 2
UMAP_NEIGHBORS = 15