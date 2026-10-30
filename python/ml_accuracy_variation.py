"""
Review 3 - Executable code
One Run Crowns a Model. Does the Next Run Agree?
Seed, test-set, and split variation in ML accuracy (Probability & Statistics project)

Install:   python -m pip install numpy scipy scikit-learn pandas matplotlib
Run:       python ml_accuracy_variation.py            (full design, ~3-6 minutes)
Demo:      python ml_accuracy_variation.py --quick    (100 A/A pairs, ~30 seconds)

Pre-committed design (unchanged from Reviews 1 and 2):
  Dataset   : Breast Cancer Wisconsin Diagnostic (569 rows, 30 features, 2 classes)
  Exp A     : 50 seeds, decision tree + random forest, ONE fixed 455/114 split
  Exp B     : 30 shared stratified 80/20 splits, three models  -> CRD vs RBD ANOVA
  Exp C     : 1,000 A/A pairs  D1 (same split, new seed)  D2 (new split + new seed)
  Checks    : residual normality (chi-square GOF, Shapiro-Wilk), CI half-width vs runs
No tuning is done. Every seed and every accuracy is saved in results.csv.

The ANOVA is computed directly from the sum-of-squares formulas
(SST = SS_model + SS_split + SSE), so no statsmodels install is needed.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- settings
BASE = 20260928          # master seed: every other seed is derived from this
TEST_SIZE = 0.20         # 80/20 split -> 455 train / 114 test
N_SEEDS = 50             # Experiment A
N_SPLITS = 30            # Experiment B (blocks)
N_AA = 1000              # Experiment C (pairs per condition)
N_TREES = 25             # forest size fixed before any result was seen
Z = 1.959963984540054    # z_{0.975}
MODELS = ["Logistic regression", "Decision tree", "Random forest"]


# ----------------------------------------------------------------- helpers
def build_models(seed: int) -> dict:
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
        "Decision tree": DecisionTreeClassifier(random_state=seed),
        "Random forest": RandomForestClassifier(
            n_estimators=N_TREES, random_state=seed, n_jobs=1),
    }


def accuracy(model, xtr, ytr, xte, yte) -> float:
    """Fit, predict, return proportion correct  K/n."""
    model.fit(xtr, ytr)
    return float(np.mean(model.predict(xte) == yte))


def t_ci(values, conf=0.95):
    """t interval for a mean:  x_bar +/- t_{n-1} * s / sqrt(n)."""
    a = np.asarray(values, dtype=float)
    half = stats.t.ppf((1 + conf) / 2, len(a) - 1) * stats.sem(a)
    return float(a.mean() - half), float(a.mean() + half)


def z_ci(k, n):
    """Normal-approximation interval for a Binomial proportion."""
    p = k / n
    half = Z * np.sqrt(p * (1 - p) / n)
    return max(0.0, p - half), min(1.0, p + half)


def anova_tables(wide: pd.DataFrame):
    """One-way CRD and additive two-way RBD ANOVA from first principles.

    wide: rows = splits (blocks, b), columns = models (treatments, t).
    Y_ij = mu + tau_i + beta_j + e_ij
    """
    Y = wide.to_numpy()                       # shape (b, t)
    b, t = Y.shape
    N = b * t
    grand = Y.mean()
    ss_total = ((Y - grand) ** 2).sum()
    ss_model = b * ((Y.mean(axis=0) - grand) ** 2).sum()
    ss_split = t * ((Y.mean(axis=1) - grand) ** 2).sum()

    # CRD: ignore split identity
    sse_crd = ss_total - ss_model
    df_m, df_e_crd = t - 1, N - t
    ms_m, mse_crd = ss_model / df_m, sse_crd / df_e_crd
    f_crd = ms_m / mse_crd
    crd = pd.DataFrame(
        {"SS": [ss_model, sse_crd], "df": [df_m, df_e_crd],
         "MS": [ms_m, mse_crd], "F": [f_crd, np.nan],
         "p": [stats.f.sf(f_crd, df_m, df_e_crd), np.nan]},
        index=["model", "Residual"])

    # RBD: remove split (block) variation from the error term
    sse_rbd = ss_total - ss_model - ss_split
    df_b, df_e_rbd = b - 1, (t - 1) * (b - 1)
    ms_b, mse_rbd = ss_split / df_b, sse_rbd / df_e_rbd
    f_m, f_b = ms_m / mse_rbd, ms_b / mse_rbd
    rbd = pd.DataFrame(
        {"SS": [ss_model, ss_split, sse_rbd], "df": [df_m, df_b, df_e_rbd],
         "MS": [ms_m, ms_b, mse_rbd], "F": [f_m, f_b, np.nan],
         "p": [stats.f.sf(f_m, df_m, df_e_rbd), stats.f.sf(f_b, df_b, df_e_rbd), np.nan]},
        index=["model", "split", "Residual"])

    # residuals of the additive model: e_ij = Y_ij - row_mean - col_mean + grand
    fitted = Y.mean(axis=1, keepdims=True) + Y.mean(axis=0, keepdims=True) - grand
    resid = Y - fitted
    return crd, rbd, mse_crd, mse_rbd, resid.ravel(), fitted.ravel()


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# ----------------------------------------------------------------- main
def main():
    global N_AA
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="demo mode: 100 A/A pairs per condition instead of 1,000")
    args = parser.parse_args()
    if args.quick:
        N_AA = 100

    out = Path(__file__).resolve().parent / "ml_luck_results"
    out.mkdir(exist_ok=True)

    banner("SETUP")
    print("Python", platform.python_version(), "| NumPy", np.__version__,
          "| SciPy", scipy.__version__, "| scikit-learn", sklearn.__version__,
          "| pandas", pd.__version__, "| Matplotlib", matplotlib.__version__)
    print(f"Mode: {'QUICK DEMO' if args.quick else 'FULL'}  (A/A pairs per condition = {N_AA})")

    X, y = load_breast_cancer(return_X_y=True)
    seed_rng = np.random.default_rng(BASE)
    fixed_split_seed = int(seed_rng.integers(1, 2**31 - 1))
    xtr, xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=fixed_split_seed)
    n_test = len(yte)
    print(f"Dataset: {len(y)} rows, {X.shape[1]} features | fixed split "
          f"train={len(ytr)} test={n_test}")
    print(f"One extra correct test case moves accuracy by 1/{n_test} = "
          f"{100 / n_test:.3f} percentage points")

    rows = []
    result = {"design": {"base_seed": BASE, "fixed_split_seed": fixed_split_seed,
                         "seed_runs": N_SEEDS, "blocked_splits": N_SPLITS,
                         "aa_pairs_each": N_AA, "forest_trees": N_TREES,
                         "test_n": n_test, "quick_mode": args.quick}}

    # ------------------------------------------------ Experiment A: seeds only
    banner("EXPERIMENT A - 50 seeds on ONE fixed split")
    seeds = 10000 + seed_rng.choice(2**31 - 1 - 10000, size=N_SEEDS * 2, replace=False)
    seed_scores = {}
    for m_idx, name in enumerate(["Decision tree", "Random forest"]):
        vals = []
        for i in range(N_SEEDS):
            sd = int(seeds[m_idx * N_SEEDS + i])
            est = (DecisionTreeClassifier(random_state=sd) if name == "Decision tree"
                   else RandomForestClassifier(n_estimators=N_TREES, random_state=sd, n_jobs=1))
            acc = accuracy(est, xtr, ytr, xte, yte)
            vals.append(acc)
            rows.append({"experiment": "A_seed", "model": name, "run": i + 1,
                         "split_seed": fixed_split_seed, "model_seed": sd, "accuracy": acc})
        v = np.asarray(vals)
        seed_scores[name] = v
        lo, hi = t_ci(v)
        print(f"{name:14s} mean={100*v.mean():6.2f}%  SD={100*v.std(ddof=1):.2f} pp  "
              f"95% CI=[{100*lo:.2f}%, {100*hi:.2f}%]  range={100*v.min():.2f}-"
              f"{100*v.max():.2f}%  unique scores={len(np.unique(v))}")
        result.setdefault("seed_variation", {})[name] = {
            "mean": v.mean(), "sd": v.std(ddof=1), "ci95": [lo, hi],
            "min": v.min(), "max": v.max(), "unique": int(len(np.unique(v)))}

    # ------------------------------------------------ Experiment B: 30 blocks
    banner("EXPERIMENT B - three models on the SAME 30 splits (blocks)")
    split_generator_seed = int(seed_rng.integers(1, 2**31 - 1))
    splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE,
                                      random_state=split_generator_seed)
    split_rows = []
    for split_id, (itr, ite) in enumerate(splitter.split(X, y), start=1):
        anchor = BASE + 1000 + split_id
        model_seeds = {MODELS[0]: anchor * 10 + 1, MODELS[1]: anchor * 10 + 2,
                       MODELS[2]: anchor * 10 + 3}
        for name, est in build_models(anchor).items():
            if name == "Logistic regression":
                est.set_params(logisticregression__random_state=model_seeds[name])
            else:
                est.set_params(random_state=model_seeds[name])
            acc = accuracy(est, X[itr], y[itr], X[ite], y[ite])
            r = {"experiment": "B_split", "split": split_id, "model": name,
                 "model_seed": model_seeds[name], "accuracy": acc}
            split_rows.append(r)
            rows.append(r)

    df = pd.DataFrame(split_rows)
    wide = df.pivot(index="split", columns="model", values="accuracy")[MODELS]
    top = wide.eq(wide.max(axis=1), axis=0).sum()
    print(f"{'Model':20s} {'Mean':>7s} {'SD':>8s} {'Binom SD':>9s} {'Ratio':>6s} {'Top':>6s}")
    for name in MODELS:
        v = wide[name].to_numpy()
        p = v.mean()
        bsd = np.sqrt(p * (1 - p) / n_test)
        print(f"{name:20s} {100*p:6.2f}% {100*v.std(ddof=1):6.2f}pp {100*bsd:7.2f}pp "
              f"{v.std(ddof=1)/bsd:5.2f}x {top[name]:3d}/{N_SPLITS}")
        result.setdefault("split_comparison", {})[name] = {
            "mean": p, "sd": v.std(ddof=1), "binomial_plugin_sd": bsd,
            "top_split_count": int(top[name])}
    print("(Top counts can sum to more than 30 because tied splits count for each model.)")

    # ANOVA
    crd, rbd, mse_c, mse_b, resid, fitted = anova_tables(wide)
    re = 100 * mse_c / mse_b
    eta = rbd.loc["model", "SS"] / (rbd.loc["model", "SS"] + rbd.loc["Residual", "SS"])
    pd.set_option("display.float_format", lambda x: f"{x:.6g}")
    print("\nCRD one-way ANOVA (split identity ignored)\n", crd.to_string())
    print("\nRBD additive two-way ANOVA (split = block)\n", rbd.to_string())
    print(f"\nRelative efficiency  MSE_CRD / MSE_RBD x 100 = {re:.1f}%")
    print(f"Partial eta-squared (model, RBD) = {eta:.4f}")
    result["anova"] = {"crd": crd.to_dict(), "rbd": rbd.to_dict(), "mse_crd": mse_c,
                       "mse_rbd": mse_b, "relative_efficiency_pct": re,
                       "partial_eta_sq": eta}

    # Paired t-tests
    print("\nPaired t-tests on matched splits (naive = ignores split overlap)")
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b_ = MODELS[i], MODELS[j]
            d = (wide[a] - wide[b_]).to_numpy()
            tt = stats.ttest_rel(wide[a], wide[b_])
            lo, hi = t_ci(d)
            # Nadeau-Bengio corrected resampled SE
            cse = np.sqrt((1 / N_SPLITS + TEST_SIZE / (1 - TEST_SIZE)) * np.var(d, ddof=1))
            ch = stats.t.ppf(0.975, N_SPLITS - 1) * cse
            print(f"  {a} - {b_}: diff={100*d.mean():+.3f} pp  naive CI=[{100*lo:.2f}, "
                  f"{100*hi:.2f}]  corrected CI=[{100*(d.mean()-ch):.2f}, "
                  f"{100*(d.mean()+ch):.2f}]  t={tt.statistic:.3f}  p={tt.pvalue:.3g}")
            pairs.append({"a": a, "b": b_, "diff": d.mean(), "t": tt.statistic,
                          "p": tt.pvalue, "naive_ci": [lo, hi],
                          "corrected_ci": [d.mean() - ch, d.mean() + ch]})
    result["paired_tests"] = pairs

    # ------------------------------------------------ Experiment C: A/A
    banner(f"EXPERIMENT C - A/A test, {N_AA} pairs per condition")
    aa_rng = np.random.default_rng(BASE + 777)
    aa_seeds = 100000 + aa_rng.choice(2**31 - 1 - 100000, size=4 * 1000, replace=False)
    aa_summary = {}
    for version in ["D1_same_split", "D2_different_splits"]:
        gaps = []
        for k in range(N_AA):
            s1, s2 = int(aa_seeds[2 * k]), int(aa_seeds[2 * k + 1])
            if version == "D1_same_split":
                sets = [(xtr, ytr, xte, yte, fixed_split_seed)] * 2
            else:
                ps1, ps2 = s1 + 31, s2 + 31
                a1 = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=ps1)
                a2 = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=ps2)
                sets = [(a1[0], a1[2], a1[1], a1[3], ps1), (a2[0], a2[2], a2[1], a2[3], ps2)]
            sc = [accuracy(RandomForestClassifier(n_estimators=N_TREES, random_state=s,
                                                  n_jobs=1), xt, yt, xv, yv)
                  for s, (xt, yt, xv, yv, _) in zip((s1, s2), sets)]
            gap = abs(sc[0] - sc[1])
            gaps.append(gap)
            rows.append({"experiment": version, "model": "Random forest", "run": k + 1,
                         "model_seed_1": s1, "model_seed_2": s2, "split_seed_1": sets[0][4],
                         "split_seed_2": sets[1][4], "accuracy_1": sc[0],
                         "accuracy_2": sc[1], "absolute_gap": gap})
        g = np.asarray(gaps)
        aa_summary[version] = {}
        for thr in (0.01, 0.02):
            cnt = int(np.sum(g >= thr - 1e-12))
            lo, hi = z_ci(cnt, N_AA)
            aa_summary[version][f">={int(thr*100)}pp"] = {"count": cnt, "prop": cnt / N_AA,
                                                          "z_ci95": [lo, hi]}
            print(f"  {version:20s} gap >= {int(thr*100)} pp: {cnt:4d}/{N_AA}  "
                  f"p_hat={cnt/N_AA:.3f}  95% Z CI=[{lo:.3f}, {hi:.3f}]")
    print("These are threshold-gap frequencies, NOT Type I error rates.")
    result["aa_test"] = aa_summary

    # ------------------------------------------------ Diagnostics
    banner("ASSUMPTION CHECKS - RBD residuals")
    sh = stats.shapiro(resid)
    zres = (resid - resid.mean()) / resid.std(ddof=1)
    edges = stats.norm.ppf(np.linspace(0, 1, 7))           # 6 equal-probability bins
    observed, _ = np.histogram(zres, bins=edges)
    expected = np.full(6, len(zres) / 6)
    chisq = float(((observed - expected) ** 2 / expected).sum())
    chidf = 6 - 1 - 2                                      # bins - 1 - estimated (mean, sd)
    chip = float(stats.chi2.sf(chisq, chidf))
    print(f"Chi-square GOF (course test): X^2={chisq:.4f}, df={chidf}, p={chip:.4f}  "
          f"observed bins={observed.tolist()}")
    print(f"Shapiro-Wilk (supplementary): W={sh.statistic:.4f}, p={sh.pvalue:.4f}")
    result["diagnostics"] = {"chi_sq": chisq, "chi_df": chidf, "chi_p": chip,
                             "shapiro_w": sh.statistic, "shapiro_p": sh.pvalue}

    # CI half-width as runs accumulate
    f = seed_scores["Random forest"]
    hw = [stats.t.ppf(0.975, n - 1) * stats.sem(f[:n]) * 100 for n in range(2, N_SEEDS + 1)]
    stable = next((i + 2 for i in range(len(hw)) if max(hw[i:]) <= 0.5), None)
    print(f"Forest 95% CI half-width at 50 runs = {hw[-1]:.3f} pp; "
          f"stays <= 0.5 pp from n = {stable}")
    result["ci_halfwidth"] = {"at_50": hw[-1], "stable_from": stable}

    # ------------------------------------------------ Save
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    with (out / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=float)

    # ------------------------------------------------ Figures
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(f * 100, bins="auto", color="#326891", edgecolor="white")
    ax.set(xlabel="Random-forest test accuracy (%)", ylabel="Count",
           title="Seed variation on one fixed train/test split")
    fig.tight_layout(); fig.savefig(out / "fig1_seed_histogram.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.boxplot([wide[n].to_numpy() * 100 for n in MODELS])
    ax.set_xticks([1, 2, 3], MODELS)
    ax.set(ylabel="Accuracy (%)", title="Three models on the same 30 splits")
    fig.tight_layout(); fig.savefig(out / "fig2_model_boxplot.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(range(2, N_SEEDS + 1), hw, marker=".")
    ax.axhline(0.5, color="firebrick", linestyle="--", label="0.5 percentage points")
    ax.set(xlabel="Number of random-forest seeds", ylabel="95% CI half-width (pp)",
           title="Precision of the fixed-split mean as runs accumulate")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "fig3_ci_halfwidth.png", dpi=160); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    stats.probplot(zres, dist="norm", plot=axs[0]); axs[0].set_title("RBD residual Q-Q plot")
    axs[1].scatter(fitted, resid, alpha=0.65)
    axs[1].axhline(0, color="firebrick", linestyle="--")
    axs[1].set(xlabel="Fitted accuracy", ylabel="Residual", title="Residuals versus fitted")
    fig.tight_layout(); fig.savefig(out / "fig4_rbd_diagnostics.png", dpi=160); plt.close(fig)

    banner("DONE")
    print(f"Saved results.csv, results.json and 4 figures in: {out}")


if __name__ == "__main__":
    main()
