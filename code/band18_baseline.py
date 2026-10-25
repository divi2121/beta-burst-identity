"""Step 2: burst filters vs a strong spectral baseline (18 bands of 2 Hz, 4 to 40 Hz, as in the Track 2 fingerprint).

Same subsets, 80% classifier epochs, no rejection, seed 0 and 10 PCA
components as the main run (burst_filter_fingerprint.py), so numbers compare directly.

Models: band6 (15-30 Hz, existing baseline), band18, burst (6 burst filters), burst+band18 (concatenated).
Assumption C1: band18 = Butterworth order 4, zero phase, edges 4, 6, ..., 40 Hz; signals stored as float32.

Outputs: outputs/burst_filter_fp/band18_results.csv, band18_figure.png, band18.log
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import burst_filter_fingerprint as bf  # noqa: E402

OUT = bf.OUT
BANDS18 = [(4 + 2 * i, 6 + 2 * i) for i in range(18)]
T0 = time.time()


def say(msg):
    line = f"+{(time.time() - T0) / 60:5.1f} min  {msg}"
    print(line, flush=True)
    with open(OUT / "band18.log", "a") as f:
        f.write(line + "\n")


def bank(X, bands):
    out = []
    for lo, hi in bands:
        sos = butter(4, (lo, hi), btype="bandpass", fs=bf.SFREQ, output="sos")
        out.append(bf.crop(sosfiltfilt(sos, X, axis=-1)).astype(np.float32))
    return out


def main():
    import pandas as pd
    (OUT / "band18.log").write_text("")
    D = bf.load_all()
    index, X = D["index"], D["X"]
    y = index[:, 0]
    keep = np.ones(len(index), bool)
    dmask = bf.dict_mask(index, keep, bf.SEED)
    clf_tr = keep & (index[:, 1] < 2) & ~dmask
    te = index[:, 1] == 2
    b6, b18 = bank(X, bf.BANDS), bank(X, BANDS18)
    say("filter banks done")
    rows = []
    for kind, chans in bf.subsets():
        names = "+".join(bf.CHANNELS[c] for c in chans)
        filters, _, _ = bf.build_filters(D["burst"], index, dmask, chans)
        burst_sig = [bf.crop(fftconvolve(X[:, chans], f[None, None, :], mode="same", axes=-1)) for f in filters]
        F = {"burst": bf.tangent(bf.covs_from(burst_sig), clf_tr),
             "band6": bf.tangent(bf.covs_from([s[:, chans] for s in b6]), clf_tr),
             "band18": bf.tangent(bf.covs_from([s[:, chans] for s in b18]), clf_tr)}
        F["burst+band18"] = np.hstack([F["burst"], F["band18"]])
        for model, Fm in F.items():
            acc, ll = bf.lr_scores(Fm, y, clf_tr, te)
            rows.append({"kind": kind, "n_ch": len(chans), "chans": names, "model": model, "acc": acc, "ll": ll})
        if kind == "fixed":
            say(f"fixed {names}: " + "  ".join(f"{r['model']} acc={r['acc']:.3f} ll={r['ll']:.3f}" for r in rows[-4:]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "band18_results.csv", index=False)
    r = df[df.kind == "random"]
    p = r.pivot_table(index=["n_ch", "chans"], columns="model", values="ll")
    a = r.pivot_table(index=["n_ch", "chans"], columns="model", values="acc")
    for n, g in p.groupby(level=0):
        say(f"{n} ch  " + "  ".join(f"{m}={g[m].median():.3f}" for m in ["band6", "band18", "burst", "burst+band18"])
            + f"  acc band18={a.loc[n]['band18'].median():.3f} burst={a.loc[n]['burst'].median():.3f}"
            + f"  burst-band18={(g.burst - g.band18).median():+.3f} (burst<band18 {(g.burst < g.band18).sum()}/{len(g)})"
            + f"  (burst+band18)-band18={(g['burst+band18'] - g.band18).median():+.3f}"
            + f" ({(g['burst+band18'] < g.band18).sum()}/{len(g)})")


if __name__ == "__main__":
    main()
