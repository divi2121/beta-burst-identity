"""Progress of detect_zhou_norej.py: stage, channels done out of 9, elapsed, per subject."""
from pathlib import Path

LOGS = Path(__file__).resolve().parents[1] / "zhou_2016_norej/logs"

for s in [1, 2, 3, 4]:
    f = LOGS / f"sub_{s}.log"
    if not f.exists():
        print(f"S{s}: not started")
        continue
    lines = f.read_text().splitlines()
    n_ch = sum("channel" in l for l in lines)
    print(f"S{s}: channels {n_ch}/9 | last: {lines[-1]}")
