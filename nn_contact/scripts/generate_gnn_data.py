#!/usr/bin/env python3
"""GNN Newton data generation — parameter sweep with SLURM array support.

Designed for HPC array jobs:
    # Run single config by index
    python3 nn_contact/scripts/generate_gnn_data.py --index 0

    # SLURM array job (runs all configs in parallel)
    sbatch --array=0-215 nn_contact/scripts/launch_gnn_datagen.sh

    # List all configs without running
    python3 nn_contact/scripts/generate_gnn_data.py --list

    # Run locally (sequential, all configs)
    python3 nn_contact/scripts/generate_gnn_data.py --all

Parameter grid (2 * 3 * 3 * 3 * 2 * 2 = 216 combinations):
    n:         [5, 10]
    nsteps:    [50, 100, 200]
    E:         [0.03, 0.05, 0.1]
    nu:        [0.2, 0.3, 0.4]
    kn_factor: [10.0, 20.0]
    plastic:   [true, false]
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter


# ── Parameter grid ──────────────────────────────────────────────────────────
GRID = {
    "n":         [5, 10],
    "nsteps":    [50, 100, 200],
    "E":         [0.03, 0.05, 0.1],
    "nu":        [0.2, 0.3, 0.4],
    "kn_factor": [10.0, 20.0],
    "plastic":   ["true", "false"],
}


def build_configs() -> list[dict]:
    """Generate all parameter combinations from grid (deterministic order)."""
    keys = list(GRID.keys())
    values = list(GRID.values())
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


ALL_CONFIGS = build_configs()


def config_tag(cfg: dict) -> str:
    """Short string tag for a configuration."""
    pl = "pl" if cfg["plastic"] == "true" else "el"
    return (f"n{cfg['n']}_ns{cfg['nsteps']}_"
            f"E{cfg['E']}_nu{cfg['nu']}_"
            f"kn{cfg['kn_factor']}_{pl}")


def run_config(index: int, output_dir: str) -> bool:
    """Run a single data collection simulation by config index.

    Returns True on success, False on failure.
    """
    if index < 0 or index >= len(ALL_CONFIGS):
        print(f"ERROR: index {index} out of range [0, {len(ALL_CONFIGS)-1}]")
        return False

    cfg = ALL_CONFIGS[index]
    tag = config_tag(cfg)

    # Locate the collect script
    script_dir = Path(__file__).resolve().parent
    collect_script = str(script_dir / "collect_gnn_newton_data.py")

    cmd = [
        sys.executable, collect_script,
        "--n", str(cfg["n"]),
        "--nsteps", str(cfg["nsteps"]),
        "--E", str(cfg["E"]),
        "--nu", str(cfg["nu"]),
        "--kn-factor", str(cfg["kn_factor"]),
        "--plastic", cfg["plastic"],
        "--output-dir", output_dir,
        "--tag", tag,
    ]

    print(f"[{index}/{len(ALL_CONFIGS)}] Config: {tag}")
    print(f"  n={cfg['n']}  nsteps={cfg['nsteps']}  E={cfg['E']}  "
          f"nu={cfg['nu']}  kn={cfg['kn_factor']}  plastic={cfg['plastic']}")
    print(f"  Command: {' '.join(cmd)}")

    t0 = perf_counter()
    result = subprocess.run(cmd)
    dt = perf_counter() - t0

    if result.returncode == 0:
        print(f"  OK — {dt:.1f}s")
        return True
    else:
        print(f"  FAILED (code={result.returncode}) — {dt:.1f}s")
        return False


def list_configs():
    """Print all configs with their indices."""
    print(f"Total configurations: {len(ALL_CONFIGS)}")
    print(f"{'Index':>5s}  {'Tag':<50s}  {'n':>2s}  {'ns':>3s}  "
          f"{'E':>5s}  {'nu':>4s}  {'kn':>5s}  {'plastic':>7s}")
    print("-" * 90)
    for i, cfg in enumerate(ALL_CONFIGS):
        tag = config_tag(cfg)
        print(f"{i:5d}  {tag:<50s}  {cfg['n']:2d}  {cfg['nsteps']:3d}  "
              f"{cfg['E']:5.2f}  {cfg['nu']:4.2f}  {cfg['kn_factor']:5.1f}  "
              f"{cfg['plastic']:>7s}")


def main():
    parser = argparse.ArgumentParser(
        description="GNN Newton data generation — parameter sweep"
    )
    parser.add_argument("--index", type=int, default=None,
                        help="Config index (for SLURM array: $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--list", action="store_true",
                        help="List all configs and exit")
    parser.add_argument("--all", action="store_true",
                        help="Run all configs sequentially (local mode)")
    parser.add_argument("--output-dir", type=str,
                        default="nn_contact/data/gnn_newton_raw",
                        help="Output directory for .npz files")
    args = parser.parse_args()

    if args.list:
        list_configs()
        return

    # Resolve output dir to absolute before any chdir
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.index is not None:
        # Single config (SLURM array mode)
        ok = run_config(args.index, output_dir)
        sys.exit(0 if ok else 1)
    elif args.all:
        # Run all configs sequentially
        t_total = perf_counter()
        n_ok, n_fail = 0, 0
        for i in range(len(ALL_CONFIGS)):
            ok = run_config(i, output_dir)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        dt = perf_counter() - t_total
        print(f"\nSweep complete: {n_ok} ok, {n_fail} failed, {dt:.1f}s total")
    else:
        parser.print_help()
        print(f"\nTotal configs: {len(ALL_CONFIGS)}  "
              f"(use --index N, --all, or --list)")


if __name__ == "__main__":
    main()
