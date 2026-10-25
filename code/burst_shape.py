"""Burst waveform shape helpers for the Zhou2016 analyses.

Functions copied verbatim from the thesis-extension scripts step1_burst_drift.py (drift within a session) and
step2_burst_identity.py (identity from shape). Only the Zhou-relevant parts are here; data loading for other
datasets is not.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHANNELS = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"]  # Zhou2016 central channels
N_COMP = 7
N_TYPES = 8
N_PERM_DRIFT = 5000  # step 1 (block permutation for drift)
N_PERM_ID = 1000  # step 2 (label permutation for identity)
BLOCK = 5  # epochs per shuffle block
SEED = 0


# ---- from step1_burst_drift.py (copied verbatim) ----

def zscore_rows(waveforms):
    """Centre and scale each waveform (row) on its own, removing amplitude."""
    w = np.asarray(waveforms, dtype=float)
    return (w - w.mean(axis=1, keepdims=True)) / w.std(axis=1, keepdims=True)


def fit_shared_pca(bursts, n_comp):
    """One PCA on the z-scored bursts of all channels pooled.

    Returns mean (64,), components (n_comp, 64), explained variance ratio (n_comp,).
    """
    pooled = np.vstack([zscore_rows(d["waveform"]) for d in bursts])
    mean = pooled.mean(axis=0)
    _, s, vt = np.linalg.svd(pooled - mean, full_matrices=False)
    ratio = s**2 / (s**2).sum()
    return mean, vt[:n_comp], ratio[:n_comp]


def epoch_channel_scores(bursts, mean, components, n_epochs):
    """Mean PCA score per epoch and channel: (n_epochs, n_channels, n_comp), NaN if no burst."""
    out = np.full((n_epochs, len(bursts), len(components)), np.nan)
    for ch, d in enumerate(bursts):
        scores = (zscore_rows(d["waveform"]) - mean) @ components.T
        trials = np.asarray(d["trial"]).astype(int)
        for t in np.unique(trials):
            out[t, ch] = scores[trials == t].mean(axis=0)
    return out


def spearman_vs_index(scores, order):
    """Spearman rho between epoch position `order` and score, per channel and component.

    scores: (n_epochs, n_channels, n_comp). order: (n_epochs,) position of each epoch.
    NaN epochs are dropped cell by cell. Returns rho (n_channels, n_comp).
    """
    _, n_ch, n_comp = scores.shape
    rho = np.full((n_ch, n_comp), np.nan)
    for ch in range(n_ch):
        for k in range(n_comp):
            y = scores[:, ch, k]
            ok = ~np.isnan(y)
            if ok.sum() < 3:
                continue
            rho[ch, k] = np.corrcoef(rankdata(order[ok]), rankdata(y[ok]))[0, 1]
    return rho


def block_permutation(n, block, rng):
    """Shuffle positions 0..n-1 in blocks of `block` consecutive items; order inside a block kept.

    The last block is shorter when n is not a multiple of `block`.
    """
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def max_stat_permutation(scores, n_perm, seed, block):
    """Observed rho and corrected p by block-shuffling epoch order once for all cells (max |rho|)."""
    rng = np.random.default_rng(seed)
    order = np.arange(scores.shape[0])
    obs = spearman_vs_index(scores, order)
    max_null = np.array([np.nanmax(np.abs(spearman_vs_index(scores, block_permutation(len(order), block, rng))))
                         for _ in range(n_perm)])
    p = (1 + (max_null[None, None, :] >= np.abs(obs)[..., None]).sum(axis=-1)) / (1 + n_perm)
    return obs, p


def pooling_check(bursts, components):
    """|corr| between each channel's own PC1 and its best-matching shared component.

    Returns (best |corr| per channel, index of that shared component).
    """
    best, which = [], []
    for d in bursts:
        w = zscore_rows(d["waveform"])
        _, _, vt = np.linalg.svd(w - w.mean(axis=0), full_matrices=False)
        r = np.abs([np.corrcoef(vt[0], c)[0, 1] for c in components])
        best.append(r.max())
        which.append(int(r.argmax()))
    return np.array(best), np.array(which)


def plot_heatmaps(results, path):
    """One rho heatmap per patient, channels x components, cells with p < 0.05 marked."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7})
    fig, axes = plt.subplots(1, len(results), figsize=(2.0 * len(results), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (subject, rho, p) in zip(axes, results):
        im = ax.imshow(rho, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        for ch, k in zip(*np.where(p < 0.05)):
            ax.text(k, ch, "*", ha="center", va="center")
        ax.set_xticks(range(rho.shape[1]), [str(k + 1) for k in range(rho.shape[1])])
        ax.set_xlabel(f"component (S{subject})")
    axes[0].set_yticks(range(len(CHANNELS)), CHANNELS)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Spearman rho vs epoch")
    fig.savefig(path, dpi=200, bbox_inches="tight")


# ---- from step2_burst_identity.py (copied verbatim) ----

def fit_pca(wave_train, n_comp):
    """PCA on training waveforms. Returns (mean (64,), components (n_comp, 64))."""
    mean = wave_train.mean(axis=0)
    _, _, vt = np.linalg.svd(wave_train - mean, full_matrices=False)
    return mean, vt[:n_comp]


def mean_features(scores, b, n_channels):
    """Mean PCA score per epoch and channel, flattened: (n_epochs, n_channels * n_comp).

    Missing (epoch, channel) cells are filled with the training-epoch mean of that column.
    """
    n_comp = scores.shape[1]
    rows = {(p, e): i for i, (p, e, _) in enumerate(b["epochs"])}
    X = np.full((len(rows), n_channels, n_comp), np.nan)
    keys = np.array([rows[(p, e)] for p, e in zip(b["patient"], b["epoch"])])
    for r in np.unique(keys):
        for ch in range(n_channels):
            m = (keys == r) & (b["channel"] == ch)
            if m.any():
                X[r, ch] = scores[m].mean(axis=0)
    X = X.reshape(len(rows), -1)
    is_train = np.array([t for _, _, t in b["epochs"]])
    fill = np.nanmean(X[is_train], axis=0)
    X[np.isnan(X)] = np.take(fill, np.where(np.isnan(X))[1])
    return X


def hist_features(scores, b, n_types, seed):
    """Fraction of each epoch's bursts in each of n_types k-means shape types (all channels pooled).

    k-means is fitted on training bursts only. Epochs with no burst get all zeros.
    """
    km = KMeans(n_clusters=n_types, n_init=10, random_state=seed).fit(scores[b["train"]])
    types = km.predict(scores)
    rows = {(p, e): i for i, (p, e, _) in enumerate(b["epochs"])}
    X = np.zeros((len(rows), n_types))
    for (p, e, t) in zip(b["patient"], b["epoch"], types):
        X[rows[(p, e)], t] += 1
    totals = X.sum(axis=1, keepdims=True)
    return np.divide(X, totals, out=np.zeros_like(X), where=totals > 0)


def fit_score(X_tr, y_tr, X_te, y_te):
    """Scaler + logistic regression. Returns (balanced accuracy, predictions)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return balanced_accuracy_score(y_te, pred), pred


def permutation_p(X_tr, y_tr, X_te, y_te, observed, n_perm, seed):
    """p of the observed balanced accuracy against shuffled training labels."""
    rng = np.random.default_rng(seed)
    null = np.array([fit_score(X_tr, rng.permutation(y_tr), X_te, y_te)[0] for _ in range(n_perm)])
    return (1 + (null >= observed).sum()) / (1 + n_perm)


def plot_confusions(cells, subjects, path):
    """Grid of confusion matrices (row-normalised), one per (condition, feature set)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7})
    conds = list(dict.fromkeys(c for c, _ in cells))
    feats = ["mean", "hist"]
    fig, axes = plt.subplots(len(conds), 2, figsize=(4.6, 2.3 * len(conds)), squeeze=False)
    labels = [f"S{s}" for s in subjects]
    for i, c in enumerate(conds):
        for j, f in enumerate(feats):
            cm = cells[(c, f)]
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax = axes[i, j]
            ax.imshow(cm, vmin=0, vmax=1, cmap="Greys")
            ax.set_xticks(range(len(subjects)), labels)
            ax.set_yticks(range(len(subjects)), labels)
            ax.set_xlabel(f"predicted ({c}, {f})")
            if j == 0:
                ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
