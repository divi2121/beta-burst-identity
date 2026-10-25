"""Epoch rejection on Zhou2016: global threshold vs full AutoReject vs FAAR, per subject and session.

Data: lab preprocessing (load_sub + apply_preprocessing) with reject_trials=False, so all three
methods see the same unrejected epochs. Every method is fitted per session.

Methods
    global : autoreject.get_rejection_threshold (one peak-to-peak threshold, as the lab pipeline,
             which fits it per subject, not per session). Epoch rejected if any channel exceeds it.
    ar     : autoreject.AutoReject (per-channel thresholds). Only epochs it marks bad as a whole
             count as rejected; channel repairs are ignored.
    faar   : from the paper's spec (arXiv 2605.12408, II.A), our defaults where it is silent:
             1. Band-pass 8 to 32 Hz. Use 0 to 6 s of each epoch.
             2. Clean reference: cut every epoch into 1 s windows, RMS per window and channel,
                z-score per channel over all windows of the session. A window is clean if |z| < 2
                on at least 90 % of channels (with 9 channels: all of them). Reference epochs are
                those whose windows are all clean.
             3. Five features per epoch and channel: mean |FFT| 8 to 32 Hz, RMS, max |diff|,
                zero-crossing rate, kurtosis. Z-score against the reference epochs.
             4. Severity per feature: |z| < 1 -> 0, 1-2 -> 1, 2-3 -> 2, > 3 -> 3.
                SQI = sum over channels and features (higher is worse).
             5. Knee (Kneedle, kneeliverse) of the sorted SQI curve; epochs above it are rejected.

Output
    Number: percent rejected per subject x session x method, and pairwise overlap.
    Figure: outputs/rejection_rates_zhou.png, one column per method, one dot per session.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bebopbci"))
REAL = ROOT / "zhou_2016"
SUBJECTS = [1, 2, 3, 4]
SFREQ = 250.0
TMIN = -1.0
WIN = (0.0, 6.0)
BAND = (8.0, 32.0)
METHODS = ["global", "ar", "faar"]


def load_raw(subject):
    """Unrejected preprocessed epochs (trials, 9, time), meta, channel names; cached."""
    cache = REAL / f"sub_{subject}/raw_epochs.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["meta"], list(d["ch_names"])
    from help_funcs import load_exp_variables
    from moabb.datasets import Zhou2016
    from preprocess import apply_preprocessing, load_sub
    ev = load_exp_variables(json_filename=str(REAL / "variables.json"))
    epochs, labels, meta = load_sub(subject, Zhou2016(), ev["tmin"], ev["tmax"], ev["exp_time_periods"][:2],
                                    ev["dataset_path"], band_pass=[0, 120])
    X, labels, meta, info, _ = apply_preprocessing(epochs, labels, meta, ev["channels"], reject_trials=False)
    meta, ch = np.asarray(meta, dtype=object), list(info["ch_names"])
    assert len(ch) == X.shape[1], (ch, X.shape)
    np.savez(cache, X=X, meta=meta, ch_names=np.array(ch))
    return X, meta, ch


def to_mne(X, ch_names):
    import mne
    info = mne.create_info(ch_names, SFREQ, "eeg")
    ep = mne.EpochsArray(X, info, tmin=TMIN, verbose=False)
    ep.set_montage("standard_1020")
    return ep


def reject_global(X, ch_names):
    from autoreject import get_rejection_threshold
    thr = get_rejection_threshold(to_mne(X, ch_names), decim=2)["eeg"]
    return np.ptp(X, axis=-1).max(axis=1) > thr


def reject_ar(X, ch_names):
    from autoreject import AutoReject
    ep = to_mne(X, ch_names)
    ar = AutoReject(n_interpolate=np.array([1, 2, 4]), random_state=0, n_jobs=1, verbose=False).fit(ep)
    return np.asarray(ar.get_reject_log(ep).bad_epochs, bool)


def faar_features(x):
    """x (epochs, channels, time) -> (epochs, channels, 5)."""
    spec = np.abs(np.fft.rfft(x, axis=-1))
    f = np.fft.rfftfreq(x.shape[-1], 1 / SFREQ)
    band = (f >= BAND[0]) & (f <= BAND[1])
    zc = (np.diff(np.signbit(x), axis=-1) != 0).mean(axis=-1)
    return np.stack([spec[..., band].mean(-1), np.sqrt((x ** 2).mean(-1)), np.abs(np.diff(x, axis=-1)).max(-1),
                     zc, kurtosis(x, axis=-1)], axis=-1)


def faar_sqi(X):
    """Returns (SQI per epoch, number of reference epochs)."""
    sos = butter(4, BAND, btype="bandpass", fs=SFREQ, output="sos")
    i0, i1 = int((WIN[0] - TMIN) * SFREQ), int((WIN[1] - TMIN) * SFREQ)
    x = sosfiltfilt(sos, X, axis=-1)[..., i0:i1]
    n_ep, n_ch, n_t = x.shape
    w = int(SFREQ)
    n_w = n_t // w
    rms = np.sqrt((x[..., :n_w * w].reshape(n_ep, n_ch, n_w, w) ** 2).mean(-1))  # (ep, ch, win)
    z = (rms - rms.mean(axis=(0, 2), keepdims=True)) / rms.std(axis=(0, 2), keepdims=True)
    clean_win = (np.abs(z) < 2).mean(axis=1) >= 0.9  # (ep, win)
    ref = clean_win.all(axis=1)
    F = faar_features(x)
    zf = (F - F[ref].mean(0)) / F[ref].std(0)
    sev = np.digitize(np.abs(zf), [1, 2, 3])
    return sev.sum(axis=(1, 2)), int(ref.sum())


def reject_faar(X):
    from kneeliverse.kneedle import knee
    sqi, n_ref = faar_sqi(X)
    order = np.argsort(sqi)
    k = knee(np.c_[np.arange(len(sqi), dtype=float), sqi[order].astype(float)])
    bad = np.zeros(len(sqi), bool)
    if k is not None:
        bad = sqi > sqi[order][k]
    return bad, n_ref


def plot(rates, path):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    rng = np.random.default_rng(0)
    colors = {s: c for s, c in zip(SUBJECTS, ["C0", "C1", "C2", "C3"])}
    for (subject, session), r in rates.items():
        for j, m in enumerate(METHODS):
            ax.scatter(j + rng.uniform(-0.12, 0.12), r[m], s=14, color=colors[subject],
                       label=f"S{subject}" if (session == 0 and j == 0) else None)
    ax.set_xticks(range(len(METHODS)), ["global threshold", "AutoReject", "FAAR"])
    ax.set_ylabel("epochs rejected per session (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    rates, masks, index = {}, {m: [] for m in METHODS}, []
    for subject in SUBJECTS:
        X, meta, ch = load_raw(subject)
        sess = meta[:, 1].astype(int)
        for session in np.unique(sess):
            Xs = X[sess == session]
            bad = {"global": reject_global(Xs, ch), "ar": reject_ar(Xs, ch)}
            bad["faar"], n_ref = reject_faar(Xs)
            rates[(subject, session)] = {m: 100 * bad[m].mean() for m in METHODS}
            for m in METHODS:
                masks[m].append(bad[m])
            index += [(subject, session, e) for e in np.flatnonzero(sess == session)]
            print(f"S{subject} ses{session}: epochs={len(Xs)}  faar reference={n_ref}  " +
                  "  ".join(f"{m}={bad[m].sum()} ({rates[(subject, session)][m]:.1f}%)" for m in METHODS), flush=True)
    masks = {m: np.concatenate(v) for m, v in masks.items()}
    np.savez(ROOT / "outputs/rejection_masks_zhou.npz", index=np.array(index), **masks)  # index: subject, session, epoch
    print(f"all sessions: epochs={len(masks['faar'])}")
    for a, b in [("global", "ar"), ("global", "faar"), ("ar", "faar")]:
        both = (masks[a] & masks[b]).sum()
        print(f"   overlap {a} & {b}: {both} epochs rejected by both ({masks[a].sum()} vs {masks[b].sum()})")
    plot(rates, ROOT / "outputs/rejection_rates_zhou.png")


if __name__ == "__main__":
    main()
