import csv
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

def load_wikipron(path):
    """Return a list of words and their IPA segments"""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["word", "ipa"],
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )

    df = df[df["ipa"].str.strip() != ""]
    df = df.drop_duplicates(subset=["word", "ipa"])
    return [ipa.split(" ") for ipa in df["ipa"]]

def iso_from_filename(path):
    return Path(path).stem.split("_")[0]

def load_glottolog(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df

def unigram_counts(words, vocab):
    """Counts of each segment, as a numpy vector aligned to vocab (a list)."""
    counts = Counter(seg for w in words for seg in w)
    return np.array([counts[s] for s in vocab], dtype=float)

def haversine_matrix(lat, lon):
    """All-pairs great-circle distance in km. lat, lon are 1D arrays in degrees."""
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))