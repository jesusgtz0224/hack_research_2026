import numpy as np
from scipy.spatial.distance import pdist, squareform

_MAX_JS = np.sqrt(np.log(2))   # largest possible JS distance with natural logs


def js_matrix(X):
    """Pairwise Jensen-Shannon distance between rows, scaled to [0, 1]."""
    return squareform(pdist(X, metric="jensenshannon") / _MAX_JS)


def combine(mats):
    """Average several distance matrices after dividing each by its own mean,
    so no block dominates just because its distances are larger."""
    scaled = []
    for m in mats:
        mean = m[np.triu_indices_from(m, k=1)].mean()
        scaled.append(m / mean)
    return np.mean(scaled, axis=0)


def haversine_matrix(lat, lon):
    """All-pairs great-circle distance in km (inputs in degrees)."""
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def same_family_matrix(family):
    """1 where two languages share a family, else 0."""
    f = np.asarray(family)
    return (f[:, None] == f[None, :]).astype(int)