"""Steps 1 and 2 on Zhou2016 bursts (4 healthy subjects, 3 sessions each).

Reuses the step 1 and step 2 functions unchanged; only the
loading differs, because Zhou has no movement/rest split and has 3 sessions per subject.

Data
    zhou_2016/sub_X/beta_bursts_superlets_nfs.npy: 9 channel dicts, 'trial' indexes the kept trials.
    zhou_2016/sub_X/meta.npy: one row per kept trial, columns subject, session, run.
    Left and right hand imagery are pooled. Channels FC3, FCz, FC4, C3, Cz, C4, CP3, CPz, CP4.
    Trial order within a session is assumed to be recording order (run 0 then run 1). Not checked.

Step 1 (drift): per subject AND per session, so the epoch index never crosses a day.
Step 2 (identity), two splits:
    a  within session: first half of every session trains, second half tests (as Claassen).
    b  across days:   sessions 0 and 1 train, session 2 tests.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import burst_shape as s1  # noqa: E402
import burst_shape as s2  # noqa: E402
from burst_shape import zscore_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "zhou_2016"
SUBJECTS = [1, 2, 3, 4]
BURST_FILE = "beta_bursts_superlets_nfs.npy"
s1.CHANNELS = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"]


def load_zhou(subject):
    """Return (bursts, session per kept trial)."""
    bursts = list(np.load(DATA / f"sub_{subject}/{BURST_FILE}", allow_pickle=True))
    meta = np.load(DATA / f"sub_{subject}/meta.npy", allow_pickle=True)
    return bursts, meta[:, 1].astype(int)


def session_bursts(bursts, session_of_trial, session):
    """Keep one session's bursts; re-index 'trial' to 0..n-1 within that session.

    Returns (bursts of that session, number of trials in it).
    """
    trials = np.flatnonzero(session_of_trial == session)
    new_index = {t: i for i, t in enumerate(trials)}
    out = []
    for d in bursts:
        tr = np.asarray(d["trial"]).astype(int)
        keep = np.isin(tr, trials)
        out.append({"waveform": np.asarray(d["waveform"])[keep],
                    "trial": np.array([new_index[t] for t in tr[keep]])})
    return out, len(trials)


def run_step1():
    results = []
    for subject in SUBJECTS:
        bursts, sess = load_zhou(subject)
        for session in np.unique(sess):
            b, n_epochs = session_bursts(bursts, sess, session)
            mean, comps, ratio = s1.fit_shared_pca(b, s1.N_COMP)
            scores = s1.epoch_channel_scores(b, mean, comps, n_epochs)
            rho, p = s1.max_stat_permutation(scores, s1.N_PERM_DRIFT, s1.SEED, s1.BLOCK)
            best, _ = s1.pooling_check(b, comps)
            ch, k = np.unravel_index(np.nanargmax(np.abs(rho)), rho.shape)
            print(f"S{subject} ses{session}: epochs={n_epochs}  PC1={ratio[0] * 100:.1f}%  "
                  f"min pooling |corr|={best.min():.2f}  largest |rho|={abs(rho[ch, k]):.2f} "
                  f"at {s1.CHANNELS[ch]} comp {k + 1}, corrected p={p[ch, k]:.4f}", flush=True)
            results.append((f"{subject} ses{session}", rho, p))
    s1.plot_heatmaps(results, str(ROOT / "outputs/step1_burst_drift_zhou.png"))


def collect_split(split):
    """Pool all bursts in the step 2 format, with the train flag set by `split` ('a' or 'b')."""
    wave, patient, channel, epoch, train, epochs = [], [], [], [], [], []
    for subject in SUBJECTS:
        bursts, sess = load_zhou(subject)
        is_train = np.zeros(len(sess), bool)
        for session in np.unique(sess):
            idx = np.flatnonzero(sess == session)
            is_train[idx] = (np.arange(len(idx)) < len(idx) // 2) if split == "a" else (session < 2)
        epochs += [(subject, e, bool(is_train[e])) for e in range(len(sess))]
        for ch, d in enumerate(bursts):
            tr = np.asarray(d["trial"]).astype(int)
            wave.append(zscore_rows(d["waveform"]))
            patient.append(np.full(len(tr), subject))
            channel.append(np.full(len(tr), ch))
            epoch.append(tr)
            train.append(is_train[tr])
    return {"wave": np.vstack(wave), "patient": np.concatenate(patient),
            "channel": np.concatenate(channel), "epoch": np.concatenate(epoch),
            "train": np.concatenate(train), "epochs": epochs}


def run_step2():
    cells = {}
    for split, name in [("a", "within session"), ("b", "across days")]:
        b = collect_split(split)
        mean, comps = s2.fit_pca(b["wave"][b["train"]], s2.N_COMP)
        scores = (b["wave"] - mean) @ comps.T
        y = np.array([p for p, _, _ in b["epochs"]])
        is_train = np.array([t for _, _, t in b["epochs"]])
        feats = {"mean": s2.mean_features(scores, b, 9), "hist": s2.hist_features(scores, b, s2.N_TYPES, s2.SEED)}
        print(f"{name}: epochs train={is_train.sum()} test={(~is_train).sum()}  bursts={len(scores)}", flush=True)
        for fname, X in feats.items():
            acc, pred = s2.fit_score(X[is_train], y[is_train], X[~is_train], y[~is_train])
            p = s2.permutation_p(X[is_train], y[is_train], X[~is_train], y[~is_train], acc, s2.N_PERM_ID, s2.SEED)
            print(f"   {fname}: balanced accuracy={acc:.3f} (chance 0.250), p={p:.4f}", flush=True)
            cells[(name, fname)] = confusion_matrix(y[~is_train], pred, labels=SUBJECTS)
    s2.plot_confusions(cells, SUBJECTS, str(ROOT / "outputs/step2_burst_identity_zhou.png"))


if __name__ == "__main__":
    run_step1()
    run_step2()
