"""Amplitude back into burst shape: does it carry stable identity, and does it add to the band-18 fingerprint?

Why: beta burst amplitude is the most stable burst feature across sessions (Pauls et al. 2024, Clin Neurophysiol),
and z-scoring the waveforms removes it.

Data: step 2 bursts (zhou_2016/, lab global rejection, all peak times), 923 epochs; band-18 on the matching
zhou_2016/sub_X/clean_epochs.npz rows, 0 to 6 s.
Cross-day: train sessions 0+1, test session 2. Tuning (C, k) and stacking weights: fit on session 0, judge on
session 1. Session 2 is scored once.

Families, per epoch, mean over the epoch's bursts per channel (missing cells = training mean, as step 2):
    F0 shape_z     z-scored waveforms, PCA 7                      (step 2 baseline)
    F1 shape_z+amp F0 plus mean log(std of raw waveform) per channel
    F2 raw_H       raw waveforms, PCA 15, top k axes by Kruskal-Wallis H across people, k in {3, 5, 7}
    F3 raw_stab    same PCA, top k axes by stability ratio, k in {3, 5, 7}
    F4 amp         mean log(std of raw waveform) per channel only
    B18            band-18 covariance tangent features
Combinations B18 + F*: stacking (logistic regression on the experts' session-1 log-probabilities) and plain
concatenation (reference).

Assumptions
    E1 PCA (raw and z-scored), axis rankings and the missing-cell fill are fitted once on sessions 0+1 bursts.
       PCA is unsupervised; the H and stability rankings use labels from sessions 0+1, so the session-1 tuning is
       somewhat optimistic for F2/F3. Session 2 never enters any fit.
    E2 stability ratio per axis = variance across people of each person's mean score (averaged over training
       sessions) / mean over people of the variance of that person's per-session means. Large = people differ
       and each person stays put across sessions. (A cell-mean version of a two-way ANOVA ratio.)
    E3 the H and stability rankings use the per-epoch score averaged over the set's channels.
    E4 within-session check (first half / second half of each session) uses the C and k tuned cross-day.
    E5 stacking: multinomial logistic regression, C = 1, on concatenated log-probabilities of the two experts.

Outputs: outputs/amp_shape/results.csv, stack_weights.csv, run.log (the figure is drawn by plot_summary.py)
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import burst_filter_fingerprint as bf  # noqa: E402
from band18_baseline import BANDS18  # noqa: E402
from burst_shape import zscore_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/amp_shape"
SUBJECTS = [1, 2, 3, 4]
CHANNELS = bf.CHANNELS
SETS = {"C3": ["C3"], "C3+C4": ["C3", "C4"], "all9": CHANNELS}
N_Z, N_RAW = 7, 15
KS = [3, 5, 7]
CS = [0.001, 0.01, 0.1, 1, 10]
T0 = time.time()


def say(msg):
    line = f"+{(time.time() - T0) / 60:5.1f} min  {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def load():
    rows, X, wave, chan, row = [], [], [], [], []
    for s in SUBJECTS:
        d = np.load(ROOT / f"zhou_2016/sub_{s}/clean_epochs.npz", allow_pickle=True)
        meta = np.load(ROOT / f"zhou_2016/sub_{s}/meta.npy", allow_pickle=True)
        assert len(d["X"]) == len(meta)
        off = len(rows)
        rows += [(s, int(ses)) for ses in meta[:, 1]]
        X.append(d["X"])
        for ch, b in enumerate(np.load(ROOT / f"zhou_2016/sub_{s}/beta_bursts_superlets_nfs.npy", allow_pickle=True)):
            wave.append(np.asarray(b["waveform"], float))
            chan.append(np.full(len(b["trial"]), ch))
            row.append(off + np.asarray(b["trial"]).astype(int))
    rows = np.array(rows)
    b = {"raw": np.vstack(wave), "chan": np.concatenate(chan), "row": np.concatenate(row)}
    b["z"] = zscore_rows(b["raw"])
    b["amp"] = np.log(b["raw"].std(axis=1))[:, None]
    say(f"loaded {len(rows)} epochs, {len(b['row'])} bursts")
    return {"person": rows[:, 0], "session": rows[:, 1], "X": np.concatenate(X), "b": b}


def epoch_means(values, b, sel, chans, n_rows, train_rows):
    """(n_bursts, d) values -> (n_rows, len(chans) * d): mean per epoch and channel, missing = training mean."""
    d = values.shape[1]
    out = np.zeros((n_rows, len(chans), d))
    cnt = np.zeros((n_rows, len(chans)))
    for j, ch in enumerate(chans):
        m = sel & (b["chan"] == ch)
        np.add.at(out[:, j], b["row"][m], values[m])
        np.add.at(cnt[:, j], b["row"][m], 1)
    with np.errstate(invalid="ignore"):
        out = out / cnt[..., None]
    out = out.reshape(n_rows, -1)
    fill = np.nanmean(out[train_rows], axis=0)
    idx = np.where(np.isnan(out))
    out[idx] = fill[idx[1]]
    return out


def pca(w):
    mean = w.mean(axis=0)
    _, _, vt = np.linalg.svd(w - mean, full_matrices=False)
    return mean, vt


def rank_axes(ep_axis, person, session, train_rows):
    """Per axis: Kruskal-Wallis H across people and the stability ratio (E2), on training epochs."""
    p, s, v = person[train_rows], session[train_rows], ep_axis[train_rows]
    H = np.array([kruskal(*[v[p == q, a] for q in SUBJECTS]).statistic for a in range(v.shape[1])])
    sess = np.unique(s)
    cell = np.array([[v[(p == q) & (s == t)].mean(axis=0) for t in sess] for q in SUBJECTS])  # person x sess x axis
    stab = cell.mean(axis=1).var(axis=0) / cell.var(axis=1).mean(axis=0)
    return np.argsort(H)[::-1], np.argsort(stab)[::-1], H, stab


def families(D, chans, train_rows):
    b, n = D["b"], len(D["person"])
    sel = np.isin(b["chan"], chans)
    tr_b = sel & train_rows[b["row"]]
    mz, vz = pca(b["z"][tr_b])
    f0 = epoch_means((b["z"] - mz) @ vz[:N_Z].T, b, sel, chans, n, train_rows)
    amp = epoch_means(b["amp"], b, sel, chans, n, train_rows)
    mr, vr = pca(b["raw"][tr_b])
    sr = (b["raw"] - mr) @ vr[:N_RAW].T
    per_ch = epoch_means(sr, b, sel, chans, n, train_rows).reshape(n, len(chans), N_RAW)
    order_h, order_s, H, stab = rank_axes(per_ch.mean(axis=1), D["person"], D["session"], train_rows)
    fam = {"F0 shape_z": {0: f0}, "F1 shape_z+amp": {0: np.hstack([f0, amp])}, "F4 amp": {0: amp}}
    fam["F2 raw_H"] = {k: per_ch[:, :, order_h[:k]].reshape(n, -1) for k in KS}
    fam["F3 raw_stab"] = {k: per_ch[:, :, order_s[:k]].reshape(n, -1) for k in KS}
    # control: the F2/F3 recipe (PCA 15, ranked axes, k) on the z-scored waveforms
    mz15, vz15 = pca(b["z"][tr_b])
    sz = (b["z"] - mz15) @ vz15[:N_RAW].T
    per_ch_z = epoch_means(sz, b, sel, chans, n, train_rows).reshape(n, len(chans), N_RAW)
    zh, zs, _, _ = rank_axes(per_ch_z.mean(axis=1), D["person"], D["session"], train_rows)
    fam["F5 shape_z_H"] = {k: per_ch_z[:, :, zh[:k]].reshape(n, -1) for k in KS}
    fam["F6 shape_z_stab"] = {k: per_ch_z[:, :, zs[:k]].reshape(n, -1) for k in KS}
    info = {"axes_H": order_h[:max(KS)].tolist(), "axes_stab": order_s[:max(KS)].tolist(),
            "H": np.round(H[order_h[:max(KS)]], 1).tolist(), "stab": np.round(stab[order_s[:max(KS)]], 2).tolist()}
    return fam, info


def band18(D, chans, train_rows, sig):
    return {0: bf.tangent(bf.covs_from([x[:, [CHANNELS.index(c) for c in chans]] for x in sig]), train_rows)}


def fit_proba(F, y, tr, te, C):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(F[tr], y[tr])
    return clf.predict_proba(F[te])


def score(P, y):
    return balanced_accuracy_score(y, np.array(SUBJECTS)[P.argmax(1)]), log_loss(y, P, labels=SUBJECTS)


def tune(opts, y, s0, s1):
    """opts {k: F}. Returns best (k, C) by session-1 log-loss after fitting on session 0."""
    best = None
    for k, F in opts.items():
        for C in CS:
            ll = log_loss(y[s1], fit_proba(F, y, s0, s1, C), labels=SUBJECTS)
            if best is None or ll < best[0]:
                best = (ll, k, C)
    return best[1], best[2]


def main():
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.log").write_text("")
    D = load()
    y, ses = D["person"], D["session"]
    s0, s1, s2, s01 = ses == 0, ses == 1, ses == 2, ses < 2
    sig = []
    for lo, hi in BANDS18:
        sos = butter(4, (lo, hi), btype="bandpass", fs=bf.SFREQ, output="sos")
        sig.append(bf.crop(sosfiltfilt(sos, D["X"], axis=-1)).astype(np.float32))
    say("band-18 signals done")
    # within-session split: first half of each person x session trains
    half = np.zeros(len(y), bool)
    for q in SUBJECTS:
        for t in np.unique(ses):
            idx = np.flatnonzero((y == q) & (ses == t))
            half[idx[: len(idx) // 2]] = True
    rows, weights = [], []
    for set_name, chans in SETS.items():
        ch = [CHANNELS.index(c) for c in chans]
        fam, info = families(D, ch, s01)
        fam["B18"] = band18(D, chans, s01, sig)
        say(f"{set_name}: axes by H {info['axes_H']} (H {info['H']}); by stability {info['axes_stab']} "
            f"(ratio {info['stab']})")
        best, P_val, P_test = {}, {}, {}
        for name, opts in fam.items():
            k, C = tune(opts, y, s0, s1)
            best[name] = (k, C)
            P_val[name] = fit_proba(opts[k], y, s0, s1, C)
            P_test[name] = fit_proba(opts[k], y, s01, s2, C)
            acc, ll = score(P_test[name], y[s2])
            rows.append({"set": set_name, "model": name, "how": "single", "k": k, "C": C, "split": "days",
                         "acc": acc, "ll": ll})
        fam_w, _ = families(D, ch, half)
        fam_w["B18"] = band18(D, chans, half, sig)
        for name, (k, C) in best.items():
            acc, ll = score(fit_proba(fam_w[name][k], y, half, ~half, C), y[~half])
            rows.append({"set": set_name, "model": name, "how": "single", "k": k, "C": C, "split": "within",
                         "acc": acc, "ll": ll})
        for name in [n for n in fam if n != "B18"]:
            Zv = np.log(np.clip(np.hstack([P_val["B18"], P_val[name]]), 1e-12, 1))
            Zt = np.log(np.clip(np.hstack([P_test["B18"], P_test[name]]), 1e-12, 1))
            stk = LogisticRegression(C=1.0, max_iter=5000).fit(Zv, y[s1])
            acc, ll = score(stk.predict_proba(Zt), y[s2])
            rows.append({"set": set_name, "model": f"B18 + {name}", "how": "stacked", "k": best[name][0],
                         "C": best[name][1], "split": "days", "acc": acc, "ll": ll})
            w = np.abs(stk.coef_)
            weights.append({"set": set_name, "model": name, "weight_B18": w[:, :4].sum(),
                            "weight_other": w[:, 4:].sum()})
            k = best[name][0]
            cat = {kk: np.hstack([fam["B18"][0], fam[name][kk]]) for kk in fam[name]}
            kc, Cc = tune({k: cat[k]}, y, s0, s1)
            acc, ll = score(fit_proba(cat[kc], y, s01, s2, Cc), y[s2])
            rows.append({"set": set_name, "model": f"B18 + {name}", "how": "concat", "k": kc, "C": Cc,
                         "split": "days", "acc": acc, "ll": ll})
        d = pd.DataFrame([r for r in rows if r["set"] == set_name])
        say(f"{set_name}:\n" + d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
        pd.DataFrame(weights).to_csv(OUT / "stack_weights.csv", index=False)
    say("stacking weights (sum |coef|):\n" + pd.DataFrame(weights).to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
