"""Bursts per epoch in rejected vs kept epochs, per session and rejection method (Zhou2016).

Inputs
    outputs/rejection_masks_zhou.npz (from faar_vs_ar_rejection.py): one row per epoch,
        index = (subject, session, epoch index within subject), and a bool mask per method.
    zhou_2016_norej/sub_X/beta_bursts_superlets_nfs.npy (from detect_zhou_norej.py): 9 channel
        dicts; 'trial' is the epoch index within subject, same order as the masks.
Subjects whose burst file does not exist yet are skipped, so this runs on partial results.

Per session and method
    rate = bursts per epoch, all 9 channels summed, averaged over the epochs in the group.
    ratio = rate in rejected epochs / rate in kept epochs (nan if nothing rejected).
    ratio > 1: the method throws away burst-rich epochs.

Output
    Number: rejected rate, kept rate and ratio per session x method.
    Figure: outputs/burst_rates_rejected_vs_kept.png, one panel per method, a line per session
            from kept to rejected.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BURSTS = ROOT / "zhou_2016_norej"
METHODS = ["global", "ar", "faar"]
NAMES = {"global": "global threshold", "ar": "AutoReject", "faar": "FAAR"}


def burst_counts(subject, n_epochs):
    """(n_epochs, 9) number of bursts per epoch and channel, or None if not detected yet."""
    f = BURSTS / f"sub_{subject}/beta_bursts_superlets_nfs.npy"
    if not f.exists():
        return None
    bursts = np.load(f, allow_pickle=True)
    return np.stack([np.bincount(np.asarray(d["trial"]).astype(int), minlength=n_epochs) for d in bursts], axis=1)


def main():
    m = np.load(ROOT / "outputs/rejection_masks_zhou.npz")
    index = m["index"]
    rows = []
    for subject in np.unique(index[:, 0]):
        sel = index[:, 0] == subject
        counts = burst_counts(subject, sel.sum())
        if counts is None:
            print(f"S{subject}: bursts not detected yet, skipped")
            continue
        assert counts.shape[0] == sel.sum(), (counts.shape, sel.sum())
        per_epoch = counts.sum(axis=1)
        for session in np.unique(index[sel, 1]):
            s = index[sel, 1] == session
            line = f"S{subject} ses{session} ({s.sum()} epochs):"
            for meth in METHODS:
                bad = m[meth][sel][s]
                kept = per_epoch[s][~bad].mean()
                rej = per_epoch[s][bad].mean() if bad.any() else np.nan
                rows.append((subject, session, meth, kept, rej))
                line += f"  {meth} rej {rej:5.1f} / kept {kept:5.1f} = {rej / kept:4.2f} (n={bad.sum()})"
            print(line)
    if rows:
        plot(rows, ROOT / "outputs/burst_rates_rejected_vs_kept.png")


def plot(rows, path):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), sharey=True)
    colors = dict(zip([1, 2, 3, 4], ["C0", "C1", "C2", "C3"]))
    for ax, meth in zip(axes, METHODS):
        seen = set()
        for subject, session, mm, kept, rej in rows:
            if mm != meth or np.isnan(rej):
                continue
            ax.plot([0, 1], [kept, rej], "-o", ms=3, lw=0.8, color=colors[subject],
                    label=f"S{subject}" if subject not in seen else None)
            seen.add(subject)
        ax.set_xticks([0, 1], ["kept", "rejected"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_xlabel(NAMES[meth])
    axes[0].set_ylabel("bursts per epoch (9 channels)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
