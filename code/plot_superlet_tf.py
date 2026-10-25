"""Superlet time-frequency maps on Zhou2016: is the beta power change during motor imagery visible?

Input: zhou_2016_norej/sub_X/tfs_superlets.npy, (epochs, 9 channels, 85 freqs 1-43 Hz step 0.5, 2001 samples -1 to 7 s).
Per person: mean over all epochs and 9 channels (left and right imagery pooled). Grand mean: mean of the 4 people.
Rows: no baseline (log10), percent change vs -0.8 to 0 s, dB vs the same baseline. Baseline applied to the averaged map.
Number: beta (13-30 Hz) mean power 0-5 s vs baseline, percent, per person.
Output: outputs/superlet_tf.png, outputs/superlet_tf_mean.npz
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = [1, 2, 3, 4]
FREQS = np.arange(1.0, 43.25, 0.5)
TIMES = np.linspace(-1.0, 7.0, 2001)
BASE = (-0.8, 0.0)
BETA = (13, 30)
CHUNK = 20


def mean_tf(subject):
    tf = np.load(ROOT / f"zhou_2016_norej/sub_{subject}/tfs_superlets.npy", mmap_mode="r")
    assert tf.shape[2:] == (len(FREQS), len(TIMES)), tf.shape
    acc = np.zeros(tf.shape[2:])
    for i in range(0, len(tf), CHUNK):
        acc += np.asarray(tf[i:i + CHUNK], dtype=np.float64).mean(axis=1).sum(axis=0)
    return acc / len(tf), len(tf)


def main():
    maps, n = {}, {}
    for s in SUBJECTS:
        maps[f"S{s}"], n[s] = mean_tf(s)
        print(f"S{s}: {n[s]} epochs", flush=True)
    maps["mean"] = np.mean([maps[f"S{s}"] for s in SUBJECTS], axis=0)
    np.savez(ROOT / "outputs/superlet_tf_mean.npz", freqs=FREQS, times=TIMES, **maps)

    b = (TIMES >= BASE[0]) & (TIMES < BASE[1])
    fb = (FREQS >= BETA[0]) & (FREQS <= BETA[1])
    task = (TIMES >= 0) & (TIMES < 5)
    for k, m in maps.items():
        ref = m[fb][:, b].mean()
        print(f"{k}: beta 0-5 s vs baseline {100 * (m[fb][:, task].mean() / ref - 1):+.1f}%")

    show = FREQS <= 40
    plt.rcParams.update({"font.size": 7})
    fig, axes = plt.subplots(3, 5, figsize=(11, 5.4), sharex=True, sharey=True)
    for j, (k, m) in enumerate(maps.items()):
        base = m[:, b].mean(axis=1, keepdims=True)
        rows = [np.log10(m), 100 * (m / base - 1), 10 * np.log10(m / base)]
        for i, (img, lab) in enumerate(zip(rows, ["log10 power", "% change", "dB"])):
            ax = axes[i, j]
            if i == 0:
                vmin, vmax, cmap = np.percentile(img[show], [2, 98]).tolist() + ["viridis"]
            else:
                lim = np.percentile(np.abs(img[show][:, TIMES > -0.8]), 98)
                vmin, vmax, cmap = -lim, lim, "RdBu_r"
            pc = ax.pcolormesh(TIMES[::4], FREQS[show], img[show][:, ::4], vmin=vmin, vmax=vmax, cmap=cmap,
                               shading="auto", rasterized=True)
            for t in (0, 5):
                ax.axvline(t, color="k", lw=0.6)
            ax.axvspan(0, 6, color="none", ec="k", ls=":", lw=0.6)
            for f in BETA:
                ax.axhline(f, color="k", lw=0.4, ls="--")
            if j == 4:
                fig.colorbar(pc, ax=ax, label=lab, fraction=0.05)
            if i == 2:
                ax.set_xlabel(f"time (s), {k}")
            if j == 0:
                ax.set_ylabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/superlet_tf.png", dpi=200)


if __name__ == "__main__":
    main()
