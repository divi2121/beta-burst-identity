# bebopbci (vendored subset)

These files are copied from [bebopbci](https://gitlab.com/sotpapad/bebopbci) (commit 6362c8c), the beta burst detection and preprocessing code of the Bonaiuto lab. They are included so the analysis runs without a separate download. bebopbci is licensed under GPL-3.0 (see `LICENSE` in this folder).

Only the modules the analysis imports are included. One change: `burst_analysis.py` lines 759 to 760 index `[0][0]` instead of `[0]`, needed with recent numpy. The exact change is in `numpy_fix.patch`.
