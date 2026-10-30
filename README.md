# Accuracy Variation Lab

Probability & Statistics project companion: an interactive web app that trains logistic regression,
a decision tree and a random forest in the browser and shows how accuracy varies with the random seed,
the number of train/test splits and the test-set size.

## Run the web app in VS Code

1. Open this folder in VS Code (File → Open Folder → `accuracy-lab`).
2. Right-click `index.html` → **Open with Live Server** (or click **Go Live** in the status bar).
3. The browser opens at `http://127.0.0.1:5500` and Experiment B starts automatically.

Live Server lets the models run on several parallel workers. Double-clicking `index.html`
also works, but then everything runs on one thread and is slower.

## Run the Python experiment (Review 3 code)

```bash
cd python
python -m pip install numpy scipy scikit-learn pandas matplotlib
python ml_accuracy_variation.py --quick   # ~25-second demo
python ml_accuracy_variation.py           # full design, reproduces the paper
```

Results and figures are written to `python/ml_luck_results/`.

## Files

| Path | What it is |
|---|---|
| `index.html` | Page layout |
| `css/style.css` | Styles, light and dark themes |
| `js/engine.js` | Seeded RNG, the three models, t / F / Z statistics, ANOVA |
| `js/app.js` | Experiments, charts and controls |
| `js/worker.js` | Background worker that trains models in parallel |
| `js/data.js` | Breast Cancer Wisconsin (569 × 30) and Digits (1,797 × 64) datasets |
| `python/ml_accuracy_variation.py` | scikit-learn version that reproduces the paper's numbers |

## Tabs

- **Split count (Experiment B):** 5–100 shared splits, running mean with 95% CI, CRD vs RBD ANOVA, relative efficiency, paired t-tests.
- **Seed only (Experiment A):** one fixed split, 10–200 seeds, score histogram, CI half-width vs runs.
- **A/A test (Experiment C):** identical models on the same split (D1) and different splits (D2), gap frequencies with Z intervals.
- **Test-set size:** observed spread vs the Binomial √(p(1−p)/n) reference.

The browser models are JavaScript re-implementations, so numbers are close to but not identical to
scikit-learn. Use the Python script when results must match the paper.
