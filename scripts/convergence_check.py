import re
import pandas as pd
from pathlib import Path
import sys

SYSTEM_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYSTEM_PATH))

from src import config
from src.io_utils import load_wikipron, load_glottolog
from src.normalize import normalize_words
from src.classes import word_to_cv
from src.profiles import subsample

# the same languages, in the same order, as the class-level analysis
meta = pd.read_parquet(config.PROCESSED / "metadata_aligned_class.parquet")
glot = load_glottolog(config.GLOTTOLOG_CSV)
fam_name = dict(zip(glot["ID"], glot["Name"]))

rows = []
for _, r in meta.iterrows():
    words = normalize_words(load_wikipron(r["path"]), affricates=config.AFFRICATES)
    words = subsample(words, config.N_WORDS, config.SEED)   # same sample as the profiles
    shapes = ["".join(word_to_cv(w)) for w in words]
    shapes = [s for s in shapes if s]
    n = len(shapes)
    rows.append(dict(
        name=r["Name"],
        family=fam_name.get(r["family"], r["family"]),
        cv_only=sum(bool(re.fullmatch(r"(CV)+", s)) for s in shapes) / n,
        vowel_final=sum(s.endswith("V") for s in shapes) / n,
        has_cluster=sum("CC" in s for s in shapes) / n,
    ))

prop = pd.DataFrame(rows).sort_values("cv_only", ascending=False)
prop.round(3).to_csv(config.RESULTS / "convergence_properties.csv", index=False)

print(prop.head(20).round(3).to_string(index=False))
print()
print(prop[["cv_only", "vowel_final", "has_cluster"]].describe().round(3))