"""Summary figure for the README: cross-day identification accuracy per burst feature family and the
18-band fingerprint, Zhou2016, 4 people (chance 0.25). Reads outputs/amp_shape/results.csv."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "outputs/amp_shape/results.csv")
d = d[(d.how == "single") & (d.split == "days")]
order = [("F0 shape_z", "burst shape,\nz-scored"), ("F4 amp", "burst\namplitude"),
         ("F1 shape_z+amp", "shape +\namplitude"), ("F5 shape_z_H", "z-scored,\nselected axes"),
         ("F2 raw_H", "raw burst\nwaveform"), ("B18", "18-band\nspectrum")]
sets = [("C3", "C3"), ("C3+C4", "C3 + C4"), ("all9", "9 channels")]
plt.rcParams.update({"font.size": 7})
fig, ax = plt.subplots(figsize=(5.6, 2.6))
w = 0.26
for j, (s, lab) in enumerate(sets):
    v = [d[(d.set == s) & (d.model == m)].acc.iloc[0] for m, _ in order]
    ax.bar([i + (j - 1) * w for i in range(len(order))], v, w, label=lab, color=["#c9d6e8", "#7fa3cc", "#2f5f98"][j])
ax.axhline(0.25, color="grey", lw=0.6, ls=":")
ax.set_xticks(range(len(order)), [o[1] for o in order])
ax.set_ylabel("cross-day accuracy")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "outputs/summary_accuracy.png", dpi=200)
