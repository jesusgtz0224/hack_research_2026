import numpy as np
from scipy.stats import rankdata


def _vec(M):
    return M[np.triu_indices_from(M, k=1)]


def _corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def mantel(X, Y, n_perm=999, seed=0):
    """Spearman-style Mantel test. Returns (r, permutation p-value)."""
    rng = np.random.default_rng(seed)
    y = rankdata(_vec(Y))
    r = _corr(rankdata(_vec(X)), y)
    n = X.shape[0]
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if abs(_corr(rankdata(_vec(X[np.ix_(p, p)])), y)) >= abs(r):
            hits += 1
    return r, (hits + 1) / (n_perm + 1)


def partial_mantel(X, Y, Z, n_perm=999, seed=0):
    """Correlation of X and Y after removing what Z explains in both."""
    rng = np.random.default_rng(seed)
    y, z = rankdata(_vec(Y)), rankdata(_vec(Z))
    ryz = _corr(y, z)

    def partial(x):
        rxy, rxz = _corr(x, y), _corr(x, z)
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

    r = partial(rankdata(_vec(X)))
    n = X.shape[0]
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if abs(partial(rankdata(_vec(X[np.ix_(p, p)])))) >= abs(r):
            hits += 1
    return r, (hits + 1) / (n_perm + 1)