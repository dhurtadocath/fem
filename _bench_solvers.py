"""Benchmark UMFPACK vs Pardiso vs MUMPS on ContactPotato_NGSolve.

Runs 3 load steps at each mesh size with each solver,
captures profiling output, and compares timings.
"""
import subprocess, sys, re, os

SCRIPT = "1_Minimization_solvers/ContactPotato_NGSolve.py"
NSTEPS = 3
MESHES = [10, 15]
SOLVERS = ["umfpack", "pardiso", "mumps"]

all_results = {}  # (mesh, solver) -> info dict

for MESH in MESHES:
    for slv in SOLVERS:
        key = (MESH, slv)
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {slv}  (n={MESH}, {NSTEPS} steps)")
        print(f"{'='*60}")

        runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_runner.py")
        script_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPT)
        with open(runner, "w") as f:
            f.write(f"import mpi4py.MPI\n")
            f.write(f"import re, sys, os\n")
            f.write(f"__file__ = {script_abs!r}\n")
            f.write(f"code = open(__file__).read()\n")
            f.write(f"code = re.sub(r'^n\\s*=.*$', 'n = {MESH}', code, count=1, flags=re.MULTILINE)\n")
            f.write(f"code = re.sub(r'^nsteps\\s*=.*$', 'nsteps = {NSTEPS}', code, count=1, flags=re.MULTILINE)\n")
            f.write(f"code = re.sub(r'^plot\\s*=.*$', 'plot = 0', code, count=1, flags=re.MULTILINE)\n")
            f.write(f"code = re.sub(r'^linear_solver\\s*=.*$', 'linear_solver = \"{slv}\"', code, count=1, flags=re.MULTILINE)\n")
            f.write(f"exec(compile(code, __file__, 'exec'))\n")

        proc = subprocess.run(
            [sys.executable, runner],
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        )
        output = proc.stdout + proc.stderr
        # Print last portion of output
        print(output[-2000:] if len(output) > 2000 else output)

        info = {"solver": slv, "mesh": MESH}

        # Extract per-call average for linear_solve
        m = re.search(r'linear_solve\s+([\d.]+)s\s+\(\s*[\d.]+%\)\s+\[(\d+)\s+calls,\s+avg\s+([\d.]+)ms\]', output)
        if m:
            info["solve_total_s"] = float(m.group(1))
            info["solve_calls"] = int(m.group(2))
            info["solve_avg_ms"] = float(m.group(3))

        # Extract assemble_lin
        m = re.search(r'assemble_lin\s+([\d.]+)s\s+\(\s*[\d.]+%\)\s+\[(\d+)\s+calls,\s+avg\s+([\d.]+)ms\]', output)
        if m:
            info["asm_total_s"] = float(m.group(1))
            info["asm_avg_ms"] = float(m.group(3))

        # Total wall time
        m = re.search(r'Total wall time:\s+([\d.]+)', output)
        if m:
            info["wall_s"] = float(m.group(1))

        all_results[key] = info

# Cleanup
try:
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_runner.py"))
except OSError:
    pass

# Summary
for MESH in MESHES:
    print(f"\n\n{'='*60}")
    print(f"  SOLVER COMPARISON — n={MESH} ({MESH**3} hex elements, {NSTEPS} steps)")
    print(f"{'='*60}")
    print(f"  {'Solver':<12s} {'Wall(s)':>8s} {'Solve(ms)':>10s} {'Asm(ms)':>9s} {'Solve tot':>10s} {'Calls':>6s}")
    print(f"  {'-'*57}")
    for slv in SOLVERS:
        r = all_results.get((MESH, slv), {})
        def fmt(val, w, dec=1):
            return f"{val:>{w}.{dec}f}" if isinstance(val, float) else f"{'?':>{w}s}"
        wall = fmt(r.get('wall_s'), 8)
        avg = fmt(r.get('solve_avg_ms'), 10)
        asm = fmt(r.get('asm_avg_ms'), 9)
        stot = fmt(r.get('solve_total_s'), 10)
        calls = f"{r.get('solve_calls', '?'):>6}" if isinstance(r.get('solve_calls'), int) else f"{'?':>6s}"
        print(f"  {slv:<12s} {wall} {avg} {asm} {stot} {calls}")
