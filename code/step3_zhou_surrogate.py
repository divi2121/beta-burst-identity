"""Step 3 on Zhou2016: does burst shape carry person information beyond spectrum and covariance?

Stage 'real' (the guard): a simple burst detector on the real data, then step 2. If it scores near
the superlet bursts (step 2: 0.49 to 0.56), it is a fair stand-in. If near chance, stop.
Stage 'surr': phase-randomised surrogates (same random phase on all channels per frequency, which
keeps every channel's spectrum and the cross-channel covariance), same detector, same step 2,
N_SEEDS times. Shape adds information only if real clearly beats the surrogates.

Simple detector, same on real and surrogate, copying the cheap parts of bebopbci/burst_detection.py:
    1. Regress the trial-average evoked response out of each trial, per channel (as line 83 there).
    2. Band-pass 13 to 30 Hz, amplitude envelope (Hilbert).
    3. Threshold: 2 x median envelope, per channel, over all trials, inside 0 to 6 s.
    4. A burst is a stretch above threshold lasting >= 100 ms. One burst per stretch.
    5. Centre: the beta-signal extremum nearest the envelope peak; dropped if > 30 ms away.
    6. Cut 64 samples around the centre from the broadband (evoked-removed) signal, remove the
       mean, and flip it if the centre is a peak, so every burst is trough-centred.
Not copied: the superlet time-frequency peak picking and each burst's own frequency band.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema, butter, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bebopbci"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step12_zhou as z  # noqa: E402
import burst_shape as s2  # noqa: E402

REAL = ROOT / "zhou_2016"
SIMPLE = ROOT / "zhou_2016_simple"
BURST_FILE = "beta_bursts_simple.npy"
SFREQ = 250.0
TMIN = -1.0
BAND = (13.0, 30.0)
WIN = (0.0, 6.0)
HALF = 32
MIN_LEN = int(0.1 * SFREQ)
MAX_ADJ = int(0.03 * SFREQ)
N_SEEDS = 20
SUPERLET_STEP2 = {("a", "mean"): 0.510, ("a", "hist"): 0.485, ("b", "mean"): 0.497, ("b", "hist"): 0.561}


def load_clean(subject):
    """Preprocessed epochs (trials, 9 channels, time) and meta, via the lab functions; cached."""
    cache = REAL / f"sub_{subject}/clean_epochs.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["meta"]
    from help_funcs import load_exp_variables
    from moabb.datasets import Zhou2016
    from preprocess import apply_preprocessing, load_sub
    ev = load_exp_variables(json_filename=str(REAL / "variables.json"))
    epochs, labels, meta = load_sub(subject, Zhou2016(), ev["tmin"], ev["tmax"], ev["exp_time_periods"][:2],
                                    ev["dataset_path"], band_pass=[0, 120])
    epochs, labels, meta, _, _ = apply_preprocessing(epochs, labels, meta, ev["channels"])
    X = epochs.get_data() if hasattr(epochs, "get_data") else np.asarray(epochs)
    meta = np.asarray(meta, dtype=object)
    np.savez(cache, X=X, meta=meta)
    return X, meta


def phase_surrogate(X, rng):
    """Per trial: one random phase per frequency, shared by all channels. Keeps spectra and cross-spectra."""
    F = np.fft.rfft(X, axis=-1)
    phi = rng.uniform(0, 2 * np.pi, size=(X.shape[0], 1, F.shape[-1]))
    phi[..., 0] = 0
    if X.shape[-1] % 2 == 0:
        phi[..., -1] = 0
    return np.fft.irfft(F * np.exp(1j * phi), n=X.shape[-1], axis=-1)


def remove_erf(X):
    """Regress the trial-average evoked response out of every trial, per channel."""
    out = np.empty_like(X)
    for ch in range(X.shape[1]):
        erf = X[:, ch].mean(axis=0)
        for t in range(X.shape[0]):
            slope, icpt = np.polyfit(erf, X[t, ch], 1)
            out[t, ch] = X[t, ch] - (icpt + slope * erf)
    return out


def detect(X):
    """Simple beta burst detector. Returns 9 per-channel dicts with 'waveform' (n, 64) and 'trial'."""
    Xr = remove_erf(X)
    sos = butter(4, BAND, btype="bandpass", fs=SFREQ, output="sos")
    beta = sosfiltfilt(sos, Xr, axis=-1)
    env = np.abs(hilbert(beta, axis=-1))
    i0, i1 = int((WIN[0] - TMIN) * SFREQ), int((WIN[1] - TMIN) * SFREQ)
    n_trials, n_ch, n_t = X.shape
    out = []
    for ch in range(n_ch):
        thr = 2 * np.median(env[:, ch, i0:i1])
        waves, trials = [], []
        for t in range(n_trials):
            sig = beta[t, ch]
            troughs, peaks = argrelextrema(sig, np.less)[0], argrelextrema(sig, np.greater)[0]
            cand = np.r_[troughs, peaks]
            is_peak = np.r_[np.zeros(len(troughs), bool), np.ones(len(peaks), bool)]
            above = np.r_[0, (env[t, ch, i0:i1] > thr).astype(int), 0]
            edges = np.diff(above)
            for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if e - s < MIN_LEN:
                    continue
                pk = i0 + s + int(np.argmax(env[t, ch, i0 + s:i0 + e]))
                j = int(np.argmin(np.abs(cand - pk)))
                c = cand[j]
                if abs(c - pk) > MAX_ADJ or c - HALF < 0 or c + HALF > n_t:
                    continue
                w = Xr[t, ch, c - HALF:c + HALF]
                w = w - w.mean()
                waves.append(-w if is_peak[j] else w)
                trials.append(t)
        out.append({"waveform": np.array(waves, dtype=np.float32).reshape(-1, 2 * HALF), "trial": np.array(trials)})
    return out


def save_bursts(folder, subject, bursts, meta):
    d = folder / f"sub_{subject}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / BURST_FILE, np.array(bursts, dtype=object), allow_pickle=True)
    np.save(d / "meta.npy", meta, allow_pickle=True)


def step2_scores(folder, with_p):
    """Step 2 (both splits, both features) on the bursts in `folder`. Returns {(split, feat): (acc, p)}."""
    z.DATA, z.BURST_FILE = folder, BURST_FILE
    res = {}
    for split in ("a", "b"):
        b = z.collect_split(split)
        mean, comps = s2.fit_pca(b["wave"][b["train"]], s2.N_COMP)
        scores = (b["wave"] - mean) @ comps.T
        y = np.array([p for p, _, _ in b["epochs"]])
        tr = np.array([t for _, _, t in b["epochs"]])
        feats = {"mean": s2.mean_features(scores, b, 9), "hist": s2.hist_features(scores, b, s2.N_TYPES, s2.SEED)}
        for name, X in feats.items():
            acc, _ = s2.fit_score(X[tr], y[tr], X[~tr], y[~tr])
            p = s2.permutation_p(X[tr], y[tr], X[~tr], y[~tr], acc, s2.N_PERM_ID, s2.SEED) if with_p else np.nan
            res[(split, name)] = (acc, p)
    return res


def plot_mean_waveforms(path):
    """Mean z-scored waveform at C3, simple vs superlet detector, one panel per subject."""
    import matplotlib.pyplot as plt
    from burst_shape import zscore_rows
    plt.rcParams.update({"font.size": 7})
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 1.9), sharey=True)
    t = (np.arange(2 * HALF) - HALF) / SFREQ * 1000
    for ax, subject in zip(axes, z.SUBJECTS):
        for folder, fname, label in [(REAL, "beta_bursts_superlets_nfs.npy", "superlet"),
                                     (SIMPLE / "real", BURST_FILE, "simple")]:
            w = np.asarray(np.load(folder / f"sub_{subject}/{fname}", allow_pickle=True)[3]["waveform"], float)
            ax.plot(t, zscore_rows(w).mean(axis=0), label=label, lw=1)
        ax.set_xlabel(f"ms (S{subject}, C3)")
    axes[0].set_ylabel("mean z-scored amplitude")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def run_real():
    for subject in z.SUBJECTS:
        X, meta = load_clean(subject)
        n_superlet = len(np.load(REAL / f"sub_{subject}/meta.npy", allow_pickle=True))
        bursts = detect(X)
        n = [len(d["trial"]) for d in bursts]
        print(f"S{subject}: X={X.shape} trials superlet run={n_superlet}  simple bursts/trial={np.mean(n) / len(X):.1f}",
              flush=True)
        save_bursts(SIMPLE / "real", subject, bursts, meta)
    res = step2_scores(SIMPLE / "real", with_p=True)
    for (split, feat), (acc, p) in res.items():
        print(f"split {split} {feat}: simple={acc:.3f} (p={p:.4f})  superlet={SUPERLET_STEP2[(split, feat)]:.3f}")
    plot_mean_waveforms(ROOT / "outputs/step3_zhou_guard_waveforms.png")


def run_surr():
    rng = np.random.default_rng(0)
    clean = {s: load_clean(s) for s in z.SUBJECTS}
    real = step2_scores(SIMPLE / "real", with_p=False)
    rows = []
    for seed in range(N_SEEDS):
        folder = SIMPLE / f"surr_seed{seed}"
        for subject, (X, meta) in clean.items():
            Xs = phase_surrogate(X, rng)
            if seed == 0:
                d_psd = np.abs(np.abs(np.fft.rfft(Xs)) - np.abs(np.fft.rfft(X))).max() / np.abs(np.fft.rfft(X)).max()
                c_r = np.einsum("nct,ndt->ncd", X - X.mean(-1, keepdims=True), X - X.mean(-1, keepdims=True))
                c_s = np.einsum("nct,ndt->ncd", Xs - Xs.mean(-1, keepdims=True), Xs - Xs.mean(-1, keepdims=True))
                print(f"S{subject} check: max rel spectrum diff={d_psd:.2e}  max rel covariance diff="
                      f"{np.abs(c_s - c_r).max() / np.abs(c_r).max():.2e}", flush=True)
            save_bursts(folder, subject, detect(Xs), meta)
        res = step2_scores(folder, with_p=False)
        rows.append([res[k][0] for k in sorted(res)])
        shutil.rmtree(folder)
        print(f"seed {seed}: " + "  ".join(f"{k[0]}-{k[1]}={res[k][0]:.3f}" for k in sorted(res)), flush=True)
    rows = np.array(rows)
    for i, k in enumerate(sorted(real)):
        r = real[k][0]
        p = (1 + (rows[:, i] >= r).sum()) / (1 + len(rows))
        print(f"split {k[0]} {k[1]}: real={r:.3f}  surrogate mean={rows[:, i].mean():.3f} "
              f"(range {rows[:, i].min():.3f}-{rows[:, i].max():.3f})  diff={r - rows[:, i].mean():+.3f}  p={p:.3f}")
    np.save(SIMPLE / "surrogate_scores.npy", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["real", "surr"])
    run_real() if ap.parse_args().stage == "real" else run_surr()
