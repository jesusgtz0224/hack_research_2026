import panphon

import panphon

_ft = panphon.FeatureTable()
_cache = {}


def _features(seg):
    """Return {feature_name: value} with values +1/0/-1, or None if seg isn't exactly one segment."""
    vecs = _ft.word_to_vector_list(seg, numeric=True)
    if len(vecs) != 1:
        return None
    return dict(zip(_ft.names, vecs[0]))


def segment_class(seg):
    if seg in _cache:
        return _cache[seg]
    f = _features(seg)
    if f is None:
        c = None
    elif f["syl"] == 1:
        c = "V"
    elif f["nas"] == 1:
        c = "N"
    elif f["lat"] == 1:
        c = "L"
    elif f["son"] == 1:
        c = "G" if f["cons"] == -1 else "L"
    elif f["cont"] == -1:
        c = "A" if f["delrel"] == 1 else "P"
    else:
        c = "F"
    _cache[seg] = c
    return c

def word_to_classes(segments, drop_unknown=True):
    out = [segment_class(s) for s in segments]
    return [c for c in out if c is not None] if drop_unknown else out

def word_to_cv(segments):
    return ["V" if c == "V" else "C" for c in word_to_classes(segments)]