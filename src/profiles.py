import random
from collections import Counter
from itertools import groupby, product

import numpy as np

from src.classes import word_to_classes, word_to_cv

BOUND = "#"
CLUSTER_CAP = 4  # cluster lengths of 4 or more share one bucket


def subsample(words, n, seed):
    """Same number of words for every language, reproducibly."""
    if n is None or len(words) <= n:
        return words
    return random.Random(seed).sample(words, n)


def to_symbols(words, level):
    """level='segment' keeps raw segments, level='class' maps to class labels."""
    if level == "segment":
        return [w for w in words if w]
    out = [word_to_classes(w) for w in words]
    return [w for w in out if w]


# ---- count blocks: each returns a Counter, not yet normalized ----

def unigram_counts(sym_words):
    return Counter(s for w in sym_words for s in w)


def bigram_counts(sym_words):
    c = Counter()
    for w in sym_words:
        padded = [BOUND] + w + [BOUND]
        c.update(zip(padded, padded[1:]))
    return c


def edge_counts(sym_words):
    start = Counter(w[0] for w in sym_words)
    end = Counter(w[-1] for w in sym_words)
    return start, end


def _runs(cv):
    return [(k, len(list(g))) for k, g in groupby(cv)]


def cluster_counts(words):
    """Consonant cluster lengths in initial, medial, and final position.
    Always computed from the class mapping, so it is comparable across languages."""
    init, med, fin = Counter(), Counter(), Counter()
    for w in words:
        runs = _runs(word_to_cv(w))
        if not runs:
            continue
        init[min(runs[0][1], CLUSTER_CAP) if runs[0][0] == "C" else 0] += 1
        fin[min(runs[-1][1], CLUSTER_CAP) if runs[-1][0] == "C" else 0] += 1
        for i, (k, n) in enumerate(runs):
            if k == "C" and 0 < i < len(runs) - 1:
                med[min(n, CLUSTER_CAP)] += 1
    return init, med, fin


def shape_counts(words):
    """CV word shapes like CVCV. Step 02 keeps only the most common shapes overall."""
    return Counter("".join(word_to_cv(w)) for w in words if word_to_cv(w))


def build_profile(words, level):
    """words: normalized words (lists of segments). Returns {block_name: Counter}."""
    sym = to_symbols(words, level)
    start, end = edge_counts(sym)
    init, med, fin = cluster_counts(words)
    return {
        "unigram": unigram_counts(sym),
        "bigram": bigram_counts(sym),
        "start": start,
        "end": end,
        "clust_init": init,
        "clust_med": med,
        "clust_fin": fin,
        "shape": shape_counts(words),
    }


# ---- turning counters into aligned probability vectors ----

def class_vocabs(classes):
    """Fixed vocabularies for the class level, so every language shares the same space."""
    padded = list(classes) + [BOUND]
    return {
        "unigram": list(classes),
        "bigram": list(product(padded, padded)),
        "start": list(classes),
        "end": list(classes),
        "clust_init": list(range(CLUSTER_CAP + 1)),
        "clust_med": list(range(1, CLUSTER_CAP + 1)),
        "clust_fin": list(range(CLUSTER_CAP + 1)),
    }


def counters_to_matrix(counters, vocab, alpha=0.5):
    """Rows are languages, columns follow vocab. Add-alpha smoothing, then row-normalize."""
    index = {v: i for i, v in enumerate(vocab)}
    M = np.full((len(counters), len(vocab)), alpha, dtype=float)
    for r, c in enumerate(counters):
        for k, n in c.items():
            if k in index:
                M[r, index[k]] += n
    return M / M.sum(axis=1, keepdims=True)