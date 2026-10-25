"""Figure for burst_filter_fingerprint.py.
Log-loss vs channel count, median over random subsets with min-max band."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs/burst_filter_fp"
df = pd.read_csv(OUT / "results.csv")
runs = [("none", "split"), ("none", "all"), ("faar", "split")]
colors = {"band": "#4c72b0", "burst": "#dd8452", "combined": "#55a868"}
plt.rcParams.update({"font.size": 8})
fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
for ax, (clean, mode) in zip(axs, runs):
    d = df[(df.clean == clean) & (df["mode"] == mode)]
    r = d[d.kind == "random"]
    for m in ["band", "burst", "combined"]:
        g = r[r.model == m].groupby("n_ch").ll
        n = g.median().index
        ax.plot(n, g.median(), "o-", color=colors[m], label=m, ms=3)
        ax.fill_between(n, g.min(), g.max(), color=colors[m], alpha=0.15, lw=0)
    ax.axhline(np.log(4), color="grey", lw=0.6, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 9], ["1", "2", "4", "9"])
    ax.set_yscale("log")
    ax.set_xlabel(f"channels ({clean}, {mode})")
axs[0].set_ylabel("cross-day log-loss")
axs[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "figure.png", dpi=200)
