"""Beta burst detection on Zhou2016 with NO trial rejection, for the rejection-method comparison.

Same lab pipeline and settings as detect_zhou.py, except:
    - apply_preprocessing(..., reject_trials=False), so bursts exist for every epoch;
    - output in zhou_2016_norej/sub_X (a new folder, so the lab code cannot reload the
      tfs_superlets.npy made for the old, rejected trial set: burst_analysis.py:209);
    - progress lines with timestamps in zhou_2016_norej/logs/sub_X.log. The lab functions are
      wrapped from outside (superlet transform, per-channel extract_bursts); lab code is not edited.
One subject per process:  python detect_zhou_norej.py 2
"""
import json
import sys
import time
from os.path import join
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "zhou_2016_norej"
sys.path.insert(0, str(ROOT / "bebopbci"))
import burst_analysis  # noqa: E402
from moabb.datasets import Zhou2016  # noqa: E402

from help_funcs import load_exp_variables  # noqa: E402
from preprocess import apply_preprocessing, load_sub  # noqa: E402

subject = int(sys.argv[1])
(OUT / "logs").mkdir(parents=True, exist_ok=True)
(OUT / f"sub_{subject}").mkdir(exist_ok=True)
LOG = OUT / f"logs/sub_{subject}.log"
T0 = time.time()


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  +{(time.time() - T0) / 60:5.1f} min  {msg}\n")


def wrap(fn, before, after):
    def inner(*a, **k):
        if before:
            log(before(*a, **k))
        out = fn(*a, **k)
        log(after(*a, **k))
        return out
    return inner


burst_analysis.superlets_mne_epochs = wrap(burst_analysis.superlets_mne_epochs,
                                           lambda e, *a, **k: f"transform start ({e.shape[0]} epochs)",
                                           lambda *a, **k: "transform done")
burst_analysis.extract_bursts = wrap(burst_analysis.extract_bursts, None,
                                     lambda *a, **k: f"channel {a[8]} done")

cfg = json.load(open(ROOT / "zhou_2016/variables.json"))
cfg["dataset_path"] = str(OUT) + "/"
json.dump(cfg, open(OUT / "variables.json", "w"), indent=4)
ev = load_exp_variables(json_filename=str(OUT / "variables.json"))

freqs = np.arange(1.0, 43.25, 0.5)
bm = burst_analysis.TfBursts(ev, freqs, np.array([1, 40]), np.where((freqs >= 1) & (freqs <= 40))[0],
                             tf_method="superlets", plot_format="png", remove_fooof=False)

log("start")
sub_dir = join(ev["dataset_path"], f"sub_{subject}/")
epochs, labels, meta = load_sub(subject, Zhou2016(), ev["tmin"], ev["tmax"], ev["exp_time_periods"][:2],
                                ev["dataset_path"], band_pass=[0, 120])
epochs, labels, meta, _, ntrials = apply_preprocessing(epochs, labels, meta, ev["channels"], reject_trials=False)
np.save(sub_dir + "ntrials", ntrials)
np.save(sub_dir + "labels", labels)
np.save(sub_dir + "meta", meta)  # columns: subject, session, run
log(f"data loaded: {epochs.shape[0]} epochs")
bm.burst_extraction(subject, epochs, labels, band="beta")
log("FINISHED")
