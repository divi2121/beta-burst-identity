"""Burst-filter fingerprint vs band-filter fingerprint across days, Zhou2016, 4 subjects x 3 sessions.

Question
    Does a person fingerprint built on burst-shape filters identify people across days better than one built on
    frequency bands, and does that change as channels drop?

Inputs
    zhou_2016/sub_X/raw_epochs.npz         all 1199 epochs, 9 channels, -1 to 7 s at 250 Hz
    zhou_2016_norej/sub_X/beta_bursts_...  superlet bursts for every epoch, 64-sample waveforms
    outputs/rejection_masks_zhou.npz       index (subject, session, epoch) + rejected-epoch masks (True = rejected)

Split
    Train: sessions 0 and 1. Test: session 2. Test epochs never shape the dictionary, axes, scaler or classifier.
    'split' mode: 20% of each person's training epochs build the dictionary, the other 80% train the classifier.
    'all' mode (check): all training epochs do both.

Pipeline per channel subset
    1. PCA (10 comp) on z-scored waveforms of dictionary bursts from the subset's channels, 4 people pooled.
    2. Per-epoch mean score per axis (subset channels pooled); Kruskal-Wallis H across the 4 people; top 3 axes.
    3. Per axis, 7 quantile bins of burst scores; mean waveform of lowest and highest bin = 2 filters (6 total),
       zero-mean, unit-norm.
    4. Convolve full epochs with each filter, crop 0 to 6 s, OAS covariance, Riemann tangent space (fit on
       classifier-train epochs), logistic regression for person.
    Band baseline: same, with 6 band-pass filters of 2.5 Hz from 15 to 30 Hz (Butterworth 4, zero phase).
    Combined: both tangent feature sets concatenated into one logistic regression.

Assumptions
    A1 z-scored waveforms (shape only) for PCA and filters; per-subject scaling would remove between-person differences.
    A2 7 quantile bins per axis.
    A3 window 0 to 6 s, as clean_vs_identity.py.
    A4 band baseline 6 x 2.5 Hz bands, 15 to 30 Hz.
    A5 combined = concatenated tangent features.
    A6 subsets: all 9 singles, 20 random pairs, 20 random quads, all 9; plus fixed motor sets.
    A7 dictionary rebuilt per subset.
    A8 axis stability: 5 seeds of the 20% draw on the fixed sets; reported results use seed 0.
    A9 runs: none/split (main), none/all (check), faar/split. Rejected epochs dropped from train and test.
    A11 an epoch with no dictionary-subset bursts is skipped in the axis test.

Outputs
    outputs/burst_filter_fp/results.csv, axes.csv, run.log, figure.png
"""
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from burst_shape import zscore_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/burst_filter_fp"
SUBJECTS = [1, 2, 3, 4]
CHANNELS = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"]
FIXED = {1: ["C3"], 2: ["C3", "C4"], 4: ["C3", "C4", "Cz", "CPz"], 9: CHANNELS}
SFREQ = 250.0
TMIN = -1.0
WIN = (0.0, 6.0)
N_COMP = 10
N_AXES = 3
N_BINS = 7
DICT_FRAC = 0.2
N_RAND = 20
BANDS = [(15 + 2.5 * i, 17.5 + 2.5 * i) for i in range(6)]
SEED = 0
T0 = time.time()


def say(msg):
    line = f"+{(time.time() - T0) / 60:5.1f} min  {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def load_all():
    m = np.load(ROOT / "outputs/rejection_masks_zhou.npz")
    index = m["index"]
    X, wave, channel, row = [], [], [], []
    off = 0
    for s in SUBJECTS:
        d = np.load(ROOT / f"zhou_2016/sub_{s}/raw_epochs.npz", allow_pickle=True)
        assert list(d["ch_names"]) == CHANNELS
        assert len(d["X"]) == (index[:, 0] == s).sum()
        X.append(d["X"])
        meta = np.load(ROOT / f"zhou_2016_norej/sub_{s}/meta.npy", allow_pickle=True)
        assert np.array_equal(index[index[:, 0] == s, 1].astype(int), meta[:, 1].astype(int)), "row order differs"
        bursts = np.load(ROOT / f"zhou_2016_norej/sub_{s}/beta_bursts_superlets_nfs.npy", allow_pickle=True)
        for ch, b in enumerate(bursts):
            tr = np.asarray(b["trial"]).astype(int)
            wave.append(zscore_rows(b["waveform"]))
            channel.append(np.full(len(tr), ch))
            row.append(off + tr)
        off += len(d["X"])
    burst = {"wave": np.vstack(wave).astype(np.float32), "channel": np.concatenate(channel),
             "row": np.concatenate(row)}
    say(f"loaded: {len(index)} epochs, {len(burst['row'])} bursts")
    return {"index": index, "X": np.concatenate(X), "masks": {k: m[k] for k in ["global", "ar", "faar"]},
            "burst": burst}


def dict_mask(index, keep, seed):
    """20% of each person's kept training epochs, seeded."""
    rng = np.random.default_rng(seed)
    dm = np.zeros(len(index), bool)
    for s in SUBJECTS:
        idx = np.flatnonzero((index[:, 0] == s) & (index[:, 1] < 2) & keep)
        dm[rng.choice(idx, int(round(DICT_FRAC * len(idx))), replace=False)] = True
    return dm


def build_filters(burst, index, dmask, chans):
    """PCA, Kruskal-Wallis axis pick, low/high bin mean waveforms. Returns filters (6, 64), axes, H."""
    sel = dmask[burst["row"]] & np.isin(burst["channel"], chans)
    w, rows = burst["wave"][sel], burst["row"][sel]
    pca = PCA(N_COMP, random_state=SEED).fit(w)
    sc = pca.transform(w)
    ep, inv = np.unique(rows, return_inverse=True)
    ep_mean = np.zeros((len(ep), N_COMP))
    np.add.at(ep_mean, inv, sc)
    ep_mean /= np.bincount(inv)[:, None]
    person = index[ep, 0]
    H = np.array([kruskal(*[ep_mean[person == s, a] for s in SUBJECTS]).statistic for a in range(N_COMP)])
    axes = np.argsort(H)[::-1][:N_AXES]
    filters = []
    for a in axes:
        edges = np.quantile(sc[:, a], np.linspace(0, 1, N_BINS + 1))
        for lo, hi in [(edges[0], edges[1]), (edges[-2], edges[-1])]:
            f = w[(sc[:, a] >= lo) & (sc[:, a] <= hi)].mean(0)
            f = f - f.mean()
            filters.append(f / np.linalg.norm(f))
    return np.array(filters), axes, H[axes]


def crop(x):
    i0, i1 = int((WIN[0] - TMIN) * SFREQ), int((WIN[1] - TMIN) * SFREQ)
    return x[..., i0:i1]


def covs_from(signals):
    from pyriemann.estimation import Covariances
    return [Covariances("oas").fit_transform(x.astype(np.float64)) for x in signals]


def tangent(covs, fit_mask):
    from pyriemann.tangentspace import TangentSpace
    return np.hstack([TangentSpace(metric="riemann").fit(C[fit_mask]).transform(C) for C in covs])


def lr_scores(F, y, tr, te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(F[tr], y[tr])
    P = clf.predict_proba(F[te])
    return balanced_accuracy_score(y[te], clf.classes_[P.argmax(1)]), log_loss(y[te], P, labels=SUBJECTS)


def score_subset(D, band_sig, chans, keep, mode, seed):
    index, X = D["index"], D["X"]
    y = index[:, 0]
    dmask = dict_mask(index, keep, seed)
    train = keep & (index[:, 1] < 2)
    clf_tr = train & ~dmask if mode == "split" else train
    te = keep & (index[:, 1] == 2)
    filters, axes, H = build_filters(D["burst"], index, dmask if mode == "split" else train, chans)
    Xs = X[:, chans]
    burst_sig = [crop(fftconvolve(Xs, f[None, None, :], mode="same", axes=-1)) for f in filters]
    F_burst = tangent(covs_from(burst_sig), clf_tr)
    F_band = tangent(covs_from([b[:, chans] for b in band_sig]), clf_tr)
    out = {}
    for name, F in [("burst", F_burst), ("band", F_band), ("combined", np.hstack([F_burst, F_band]))]:
        out[name] = lr_scores(F, y, clf_tr, te)
    return out, axes, H, clf_tr, te


def subsets():
    rng = np.random.default_rng(SEED)
    out = [("random", [c]) for c in range(9)]
    for k in (2, 4):
        combos = list(combinations(range(9), k))
        out += [("random", list(combos[i])) for i in rng.choice(len(combos), N_RAND, replace=False)]
    out.append(("random", list(range(9))))
    out += [("fixed", [CHANNELS.index(c) for c in names]) for names in FIXED.values()]
    return out


def main():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.log").write_text("")
    D = load_all()
    band_sig = []
    for lo, hi in BANDS:
        sos = butter(4, (lo, hi), btype="bandpass", fs=SFREQ, output="sos")
        band_sig.append(crop(sosfiltfilt(sos, D["X"], axis=-1)))
    say("band signals done")
    runs = [("none", "split"), ("none", "all"), ("faar", "split")]
    rows, axrows = [], []
    for clean, mode in runs:
        keep = np.ones(len(D["index"]), bool) if clean == "none" else ~D["masks"][clean]
        for kind, chans in subsets():
            names = "+".join(CHANNELS[c] for c in chans)
            res, axes, H, clf_tr, te = score_subset(D, band_sig, chans, keep, mode, SEED)
            for model, (acc, ll) in res.items():
                rows.append({"clean": clean, "mode": mode, "kind": kind, "n_ch": len(chans), "chans": names,
                             "model": model, "acc": acc, "ll": ll})
            axrows.append({"clean": clean, "mode": mode, "kind": kind, "chans": names, "seed": SEED,
                           "axes": " ".join(map(str, axes)), "H": " ".join(f"{h:.1f}" for h in H)})
            if kind == "fixed":
                say(f"{clean}/{mode} fixed {names}: " +
                    "  ".join(f"{m} acc={a:.3f} ll={l:.3f}" for m, (a, l) in res.items()))
                if clean == "none" and mode == "split":
                    for sd in range(1, 5):
                        _, ax2, H2, _, _ = score_subset(D, band_sig, chans, keep, mode, sd)
                        axrows.append({"clean": clean, "mode": mode, "kind": "stability", "chans": names,
                                       "seed": sd, "axes": " ".join(map(str, ax2)),
                                       "H": " ".join(f"{h:.1f}" for h in H2)})
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
        pd.DataFrame(axrows).to_csv(OUT / "axes.csv", index=False)
        say(f"run {clean}/{mode} done")
    report(pd.DataFrame(rows))


def phase_scramble(f, rng):
    """Same power spectrum as filter f, random phases (DC and Nyquist kept real), zero-mean, unit norm."""
    F = np.fft.rfft(f)
    ph = rng.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0
    if len(f) % 2 == 0:
        ph[-1] = 0
    g = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(f))
    g = g - g.mean()
    return g / np.linalg.norm(g)


def scramble_control(n_scr=5):
    """Step 1 control (added 2026-10-21): burst filters vs the same filters with phases scrambled.
    Main run only (no rejection, 20/80 split, seed 0), fixed channel sets. Covariance depends only on the
    filter power spectrum, so scrambled should match burst if the gain is spectral."""
    import pandas as pd
    D = load_all()
    index, X = D["index"], D["X"]
    y = index[:, 0]
    keep = np.ones(len(index), bool)
    dmask = dict_mask(index, keep, SEED)
    clf_tr = keep & (index[:, 1] < 2) & ~dmask
    te = index[:, 1] == 2
    rng = np.random.default_rng(SEED)
    rows = []
    for names in FIXED.values():
        chans = [CHANNELS.index(c) for c in names]
        filters, _, _ = build_filters(D["burst"], index, dmask, chans)
        sets = [("burst", 0, filters)] + [("scrambled", r, np.array([phase_scramble(f, rng) for f in filters]))
                                          for r in range(n_scr)]
        for name, rep, fs in sets:
            sig = [crop(fftconvolve(X[:, chans], f[None, None, :], mode="same", axes=-1)) for f in fs]
            acc, ll = lr_scores(tangent(covs_from(sig), clf_tr), y, clf_tr, te)
            rows.append({"n_ch": len(chans), "chans": "+".join(names), "model": name, "rep": rep,
                         "acc": acc, "ll": ll})
            say(f"scramble {'+'.join(names)} {name} {rep}: acc={acc:.3f} ll={ll:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scramble.csv", index=False)
    return df


def report(df):
    for (clean, mode), d in df[df.kind == "random"].groupby(["clean", "mode"], sort=False):
        say(f"\n{clean}/{mode}  random subsets, log-loss median [min, max]; acc median")
        for n in sorted(d.n_ch.unique()):
            parts = []
            for m in ["band", "burst", "combined"]:
                x = d[(d.n_ch == n) & (d.model == m)]
                parts.append(f"{m}={x.ll.median():.3f} [{x.ll.min():.3f},{x.ll.max():.3f}] acc={x.acc.median():.3f}")
            g = (d[(d.n_ch == n) & (d.model == "burst")].ll.values - d[(d.n_ch == n) & (d.model == "band")].ll.values)
            say(f"  {n} ch  " + "  ".join(parts) + f"  burst-band median={np.median(g):+.3f}")


if __name__ == "__main__":
    if "--scramble" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        scramble_control()
    else:
        main()
