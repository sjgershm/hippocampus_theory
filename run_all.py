"""Run all simulations and regenerate every figure (PDF) in figures/."""

import subprocess
import sys
import os

SIMS = [
    "sims/sim1_theta_phase.py",
    "sims/sim2_predictive_inference.py",   # backward skew + reward clustering + inference
    "sims/sim3_multiscale_discounting.py",
    "sims/sim4_context_dg.py",
    "sims/sim5_partial_remapping.py",
    "sims/sim6_splitter_cells.py",
]

here = os.path.dirname(os.path.abspath(__file__))
for s in SIMS:
    print(f"\n=== {s} ===")
    subprocess.run([sys.executable, os.path.join(here, s)], check=True)
print("\nAll figures written to figures/.")
