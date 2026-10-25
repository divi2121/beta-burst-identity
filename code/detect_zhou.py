"""Beta burst detection on Zhou2016 with the bebopbci pipeline, same settings as 01_detect_bursts.py.

Runs the lab code unchanged; only the config path is local
(zhou_2016/variables.json, a copy of bebopbci/variables.json with dataset_path changed).
Also saves the MOABB metadata (session, run per trial) needed for the cross-day steps.
"""
import sys
from os.path import join
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bebopbci"))
from moabb.datasets import Zhou2016  # noqa: E402

from burst_analysis import TfBursts  # noqa: E402
from help_funcs import load_exp_variables  # noqa: E402
from preprocess import apply_preprocessing, load_sub  # noqa: E402

experimental_vars = load_exp_variables(json_filename=str(ROOT / "zhou_2016/variables.json"))
experimental_vars["dataset_path"] = str(ROOT / "zhou_2016") + "/"  # config holds a machine-specific path
savepath = experimental_vars["dataset_path"]
subjects = [int(s) for s in sys.argv[1:]] or list(range(1, experimental_vars["n_subjects"] + 1))

freqs = np.arange(1.0, 43.25, 0.5)
upto_gamma_band = np.array([1, 40])
upto_gamma_range = np.where((freqs >= 1) & (freqs <= 40))[0]

bm = TfBursts(experimental_vars, freqs, upto_gamma_band, upto_gamma_range,
              tf_method="superlets", plot_format="png", remove_fooof=False)

for subject in subjects:
    print(f"Subject {subject}", flush=True)
    sub_dir = join(savepath, f"sub_{subject}/")
    epochs, labels, meta = load_sub(subject, Zhou2016(), experimental_vars["tmin"], experimental_vars["tmax"],
                                    experimental_vars["exp_time_periods"][:2], savepath, band_pass=[0, 120])
    epochs, labels, meta, _, ntrials = apply_preprocessing(epochs, labels, meta, experimental_vars["channels"])
    np.save(sub_dir + "ntrials", ntrials)
    np.save(sub_dir + "labels", labels)
    np.save(sub_dir + "meta", meta)  # columns: subject, session, run
    bm.burst_extraction(subject, epochs, labels, band="beta")
