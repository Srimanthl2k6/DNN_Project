<<<<<<< HEAD
# DNN_Project
=======
# DNN Project

This repository contains the DNN Project for CIFAR-10H uncertainty and robustness evaluation.

Contents:
- `src/` - project source code (training, evaluation, robustness, visualization, finalization)
- `checkpoints/` - model checkpoints (ignored from git)
- `plots/` - generated visualizations
- `results.csv`, `results_final.csv`, `FINAL_COMPARISON_MATRIX.csv`, `FINAL_REPORT.md`

Quick start

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run evaluation and finalization pipeline:

```bash
python src/benchmark_robustness.py
python src/best_model_selection.py
python src/visualize.py
python src/finalize_project.py
```

Notes
- Checkpoints and large binary datasets are intentionally ignored in `.gitignore` to keep the repository lightweight.
- If you want to include checkpoints, move them to a release or upload to a separate storage.
>>>>>>> 7baa03f (chore: prepare repository for GitHub (README, .gitignore, requirements))
