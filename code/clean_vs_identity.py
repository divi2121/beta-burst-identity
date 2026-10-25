"""Does epoch rejection change identity and task classification? Zhou2016, 4 subjects x 3 sessions.

Inputs
    zhou_2016/sub_X/raw_epochs.npz         unrejected epochs (from faar_vs_ar_rejection.py)
    zhou_2016_norej/sub_X/beta_bursts_...  superlet bursts for every epoch (detect_zhou_norej.py)
    outputs/rejection_masks_zhou.npz       rejected epochs per method, fitted per session

Conditions
    none, global, ar, faar: drop rejected epochs from train AND test.
    random_<m>: per session, drop as many epochs as method m did, chosen at random; N_RAND repeats.
Splits
    within: first half of each session trains, second half tests.
    days:   sessions 0 and 1 train, session 2 tests.
Readouts
    identity, covariance: alpha (8-13) and beta (13-30) Hz on 0-6 s, OAS covariance, tangent space
        fitted on train, scaler + logistic regression. Balanced accuracy and log-loss (4 classes).
    identity, burst shape: step 2 features. One PCA (7 comps) on training bursts of all subjects,
        then 'mean' (mean score per channel x comp) and 'hist' (8 k-means types, fraction per epoch).
    task, covariance: left vs right per subject, same covariance pipeline. Balanced accuracy.
Output
    Number: table of scores per condition x split (random: min to max over repeats).
    Figure: outputs/clean_vs_identity.png.
    All scores: outputs/clean_vs_identity.csv. Progress lines go to stdout as it runs.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import burst_shape as s2  # noqa: E402
from burst_shape import zscore_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = [1, 2, 3, 4]
SFREQ = 250.0
TMIN = -1.0
WIN = (0.0, 6.0)
BANDS = [(8, 13), (13, 30)]
METHODS = ["global", "ar", "faar"]
N_RAND = 20
SEED = 0
T0 = time.time()


def say(msg):
    print(f"+{(time.time() - T0) / 60:5.1f} min  {msg}", flush=True)


def load_all():
    """Per-epoch arrays over all subjects, in the order of the rejection masks."""
    m = np.load(ROOT / "outputs/rejection_masks_zhou.npz")
    index = m["index"]
    X, task, waves = [], [], []
    for subject in SUBJECTS:
        d = np.load(ROOT / f"zhou_2016/sub_{subject}/raw_epochs.npz", allow_pickle=True)
        labels = np.load(ROOT / f"zhou_2016_norej/sub_{subject}/labels.npy", allow_pickle=True)
        assert len(labels) == len(d["X"]) == (index[:, 0] == subject).sum()
        X.append(d["X"])
        task.append(labels)
        bursts = np.load(ROOT / f"zhou_2016_norej/sub_{subject}/beta_bursts_superlets_nfs.npy", allow_pickle=True)
        waves.append(bursts)
        say(f"loaded S{subject}")
    X = np.concatenate(X)
    i0, i1 = int((WIN[0] - TMIN) * SFREQ), int((WIN[1] - TMIN) * SFREQ)
    filt = []
    for band in BANDS:
        sos = butter(4, band, btype="bandpass", fs=SFREQ, output="sos")
        filt.append(sosfiltfilt(sos, X, axis=-1)[..., i0:i1])
    covs = [cov_oas(x) for x in filt]
    say("covariances done")
    # bursts as flat arrays, with a global epoch row index matching `index`
    offs = np.r_[0, np.cumsum([(index[:, 0] == s).sum() for s in SUBJECTS])]
    wave, channel, row = [], [], []
    for k, bursts in enumerate(waves):
        for ch, b in enumerate(bursts):
            tr = np.asarray(b["trial"]).astype(int)
            wave.append(zscore_rows(b["waveform"]))
            channel.append(np.full(len(tr), ch))
            row.append(offs[k] + tr)
    burst = {"wave": np.vstack(wave).astype(np.float32), "channel": np.concatenate(channel),
             "row": np.concatenate(row)}
    say(f"bursts: {len(burst['row'])}")
    return {"index": index, "masks": {k: m[k] for k in METHODS}, "covs": covs,
            "task": np.concatenate(task), "burst": burst}


def cov_oas(x):
    from pyriemann.estimation import Covariances
    return Covariances("oas").fit_transform(x.astype(np.float64))


def split_train(index, split):
    """True for training epochs."""
    tr = np.zeros(len(index), bool)
    for s in SUBJECTS:
        for ses in np.unique(index[index[:, 0] == s, 1]):
            idx = np.flatnonzero((index[:, 0] == s) & (index[:, 1] == ses))
            tr[idx] = (np.arange(len(idx)) < len(idx) // 2) if split == "within" else (ses < 2)
    return tr


def tangent(covs, tr, keep):
    from pyriemann.tangentspace import TangentSpace
    F = []
    for C in covs:
        ts = TangentSpace(metric="riemann").fit(C[keep & tr])
        F.append(ts.transform(C[keep]))
    return np.hstack(F)


def lr_scores(F_tr, y_tr, F_te, y_te, labels):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(F_tr, y_tr)
    P = clf.predict_proba(F_te)
    acc = balanced_accuracy_score(y_te, clf.classes_[P.argmax(1)])
    return acc, log_loss(y_te, P, labels=labels)


def burst_features(burst, keep, tr, index):
    """Step 2 features for the kept epochs. Returns {'mean': X, 'hist': X} with rows = kept epochs."""
    rows_kept = np.flatnonzero(keep)
    sel = keep[burst["row"]]
    epochs = [(int(index[r, 0]), int(r), bool(tr[r])) for r in rows_kept]
    b = {"wave": burst["wave"][sel], "channel": burst["channel"][sel], "epoch": burst["row"][sel],
         "patient": index[burst["row"][sel], 0], "train": tr[burst["row"][sel]], "epochs": epochs}
    mean, comps = s2.fit_pca(b["wave"][b["train"]], s2.N_COMP)
    scores = (b["wave"] - mean) @ comps.T
    return {"mean": s2.mean_features(scores, b, 9), "hist": s2.hist_features(scores, b, s2.N_TYPES, SEED)}


def evaluate(D, keep, split):
    """All readouts for one kept-epoch mask and split. Returns a dict of scores."""
    index, tr = D["index"], split_train(D["index"], split)
    y = index[keep, 0]
    trk = tr[keep]
    out = {}
    F = tangent(D["covs"], tr, keep)
    out["id_cov_acc"], out["id_cov_ll"] = lr_scores(F[trk], y[trk], F[~trk], y[~trk], SUBJECTS)
    for name, Fb in burst_features(D["burst"], keep, tr, index).items():
        out[f"id_{name}_acc"], out[f"id_{name}_ll"] = lr_scores(Fb[trk], y[trk], Fb[~trk], y[~trk], SUBJECTS)
    task = D["task"][keep]
    for s in SUBJECTS:
        m = y == s
        Fs = tangent([C[np.flatnonzero(keep)[m]] for C in D["covs"]], trk[m], np.ones(m.sum(), bool))
        out[f"task_S{s}"], _ = lr_scores(Fs[trk[m]], task[m][trk[m]], Fs[~trk[m]], task[m][~trk[m]],
                                         np.unique(task))
    return out


def random_mask(index, n_drop_per_session, rng):
    keep = np.ones(len(index), bool)
    for (s, ses), n in n_drop_per_session.items():
        idx = np.flatnonzero((index[:, 0] == s) & (index[:, 1] == ses))
        keep[rng.choice(idx, n, replace=False)] = False
    return keep


def main():
    import pandas as pd
    D = load_all()
    index = D["index"]
    rng = np.random.default_rng(SEED)
    rows = []
    for split in ["within", "days"]:
        conds = [("none", np.ones(len(index), bool))] + [(m, ~D["masks"][m]) for m in METHODS]
        for name, keep in conds:
            rows.append({"split": split, "cond": name, "rep": 0, **evaluate(D, keep, split)})
            say(f"{split} {name}: " + "  ".join(f"{k}={v:.3f}" for k, v in rows[-1].items()
                                                  if k.startswith(("id_cov", "id_mean_acc", "id_hist_acc"))))
        for m in METHODS:
            n_drop = {(s, ses): int(D["masks"][m][(index[:, 0] == s) & (index[:, 1] == ses)].sum())
                      for s, ses in {(int(a), int(b)) for a, b in index[:, :2]}}
            for rep in range(N_RAND):
                rows.append({"split": split, "cond": f"random_{m}", "rep": rep,
                             **evaluate(D, random_mask(index, n_drop, rng), split)})
                if rep % 5 == 4:
                    say(f"{split} random_{m}: {rep + 1}/{N_RAND}")
            pd.DataFrame(rows).to_csv(ROOT / "outputs/clean_vs_identity.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/clean_vs_identity.csv", index=False)
    report(df)
    plot(df, ROOT / "outputs/clean_vs_identity.png")


def report(df):
    cols = ["id_cov_acc", "id_cov_ll", "id_mean_acc", "id_hist_acc"] + [f"task_S{s}" for s in SUBJECTS]
    for split in ["within", "days"]:
        print(f"\n{split}")
        d = df[df.split == split]
        for cond in ["none"] + METHODS:
            r = d[d.cond == cond].iloc[0]
            print(f"  {cond:7s} " + "  ".join(f"{c}={r[c]:.3f}" for c in cols))
            if cond in METHODS:
                rr = d[d.cond == f"random_{cond}"]
                print(f"  {'random':7s} " + "  ".join(f"{c}={rr[c].min():.3f}-{rr[c].max():.3f}" for c in cols))


def plot(df, path):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7})
    panels = [("id_cov_acc", "identity, covariance"), ("id_cov_ll", "identity, covariance (log-loss)"),
              ("id_mean_acc", "identity, burst mean"), ("id_hist_acc", "identity, burst hist"),
              ("task", "task, covariance")]
    conds = ["none"] + METHODS
    fig, axes = plt.subplots(2, len(panels), figsize=(9, 3.6), sharex=True)
    for i, split in enumerate(["within", "days"]):
        d = df[df.split == split]
        for j, (col, label) in enumerate(panels):
            ax = axes[i, j]
            for x, cond in enumerate(conds):
                r = d[d.cond == cond]
                if cond in METHODS:
                    rr = d[d.cond == f"random_{cond}"]
                    vals = rr[[f"task_S{s}" for s in SUBJECTS]].values.ravel() if col == "task" else rr[col]
                    ax.plot([x, x], [vals.min(), vals.max()], color="0.8", lw=5, solid_capstyle="butt")
                if col == "task":
                    for k, s in enumerate(SUBJECTS):
                        ax.scatter(x + (k - 1.5) * 0.08, r[f"task_S{s}"].iloc[0], s=8, color=f"C{k}", zorder=3,
                                   label=f"S{s}" if (x == 0 and i == 0) else None)
                else:
                    ax.scatter(x, r[col].iloc[0], s=12, color="k", zorder=3)
            ax.set_xticks(range(len(conds)), ["none", "global", "AR", "FAAR"])
            if i == 1:
                ax.set_xlabel(label)
            if j == 0:
                ax.set_ylabel(f"{split}\nscore")
    axes[0, -1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
