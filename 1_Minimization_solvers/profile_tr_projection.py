#!/usr/bin/env python3
"""
Profiling script for TR projection contact mechanics.
Runs a shorter simulation to identify bottlenecks.
"""

import meshio, sys, os, pickle
import numpy as np
import cProfile
import pstats
import io
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyClasses.FEAssembly import *
from PyClasses.Contacts import *
from PyClasses.FEModel import *

os.chdir(sys.path[0])

def setup_model():
    """Set up the contact model for profiling."""
    # Use smaller mesh for faster profiling
    mesh = 5
    plastic = 0

    # BLOCK
    mesh_blk = meshio.read(f"../Meshes/Cubes/cube{mesh}x{mesh}x{mesh}.msh")
    X_blk = mesh_blk.points
    hexas_blk = mesh_blk.cells_dict['hexahedron']
    blk = FEAssembly(X_blk, hexas_blk, name="BLOCK", recOuters=False)
    blk.Youngsmodulus = 0.05
    blk.Translate([0.0, 0.0, 3.5])

    # POTATO
    [ptt] = pickle.load(open("Dat/PotatoAssembly.dat", "rb"))
    if hasattr(ptt, 'hexas') and not hasattr(ptt, 'elements'):
        ptt.elements = ptt.hexas
    ptt.isRigid = True

    # Selections
    blk_bottom = blk.SelectFlatSide("-z")
    blk_top = blk.SelectFlatSide("+z")
    ptt_highernodes = ptt.SelectHigherThan("z", val=0.5, Strict=True, OnSurface=True)

    # Boundary conditions
    cond_bd1 = [ptt, ptt.SelectAll(), "dirichlet", "xyz", [0.0, 0.0, 0.0]]
    cond_bd2 = [blk, blk_top, "dirichlet", "xyz", [12.0, 0.0, 0.0]]
    BCs = [cond_bd1, cond_bd2]

    # Contact with TR projection
    slave = [blk, blk_bottom]
    master = [ptt, ptt_highernodes]
    E = blk.Youngsmodulus
    base_kn = 20.0 * E
    maxGN = 0.001

    contact1 = Contact(slave, master, kn=base_kn, C1Edges=False,
                       maxGN=maxGN, f0=0.1, use_TR_projection=True,
                       TR_init=0.1, TR_min=1e-15, TR_max=1.0)

    # Model
    model = FEModel([blk, ptt], [contact1], BCs, subname="_profile_test")

    # Initialize patches
    ndofs = 3 * (len(X_blk) + len(ptt.X))
    ptt.surf.ComputeGrgPatches(np.zeros(ndofs), range(len(ptt.surf.nodes)))

    return model

def run_profile():
    """Run profiling on the simulation."""
    print("=" * 60)
    print("PROFILING TR PROJECTION CONTACT MECHANICS")
    print("=" * 60)

    # Setup
    print("\n[1/4] Setting up model...")
    t0 = time.time()
    model = setup_model()
    print(f"      Setup completed in {time.time()-t0:.2f}s")

    # Profile the solve
    print("\n[2/4] Running profiled simulation (10 time steps)...")
    pr = cProfile.Profile()

    t0 = time.time()
    pr.enable()
    model.Solve(TimeSteps=10, max_iter=20, recover=False, minimethod="BFGS", plot=0)
    pr.disable()
    solve_time = time.time() - t0
    print(f"      Solve completed in {solve_time:.2f}s")

    # Generate stats
    print("\n[3/4] Generating profiling statistics...")

    # Sort by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)  # Top 50 functions
    cumulative_stats = s.getvalue()

    # Sort by total time (self time)
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(50)
    tottime_stats = s2.getvalue()

    # Save to file
    print("\n[4/4] Saving results...")
    with open("profile_results.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PROFILING RESULTS - TR PROJECTION CONTACT MECHANICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Mesh: 5x5x5\n")
        f.write(f"  Time steps: 10\n")
        f.write(f"  Solver: BFGS\n")
        f.write(f"  Total solve time: {solve_time:.2f}s\n\n")

        f.write("=" * 80 + "\n")
        f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME\n")
        f.write("=" * 80 + "\n")
        f.write(cumulative_stats)

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP 50 FUNCTIONS BY SELF TIME (tottime)\n")
        f.write("=" * 80 + "\n")
        f.write(tottime_stats)

    # Save binary stats for further analysis
    pr.dump_stats("profile_tr.pstats")

    # Print summary to console
    print("\n" + "=" * 60)
    print("TOP 20 BY CUMULATIVE TIME")
    print("=" * 60)
    ps.print_stats(20)

    print("\n" + "=" * 60)
    print("TOP 20 BY SELF TIME")
    print("=" * 60)
    ps2.print_stats(20)

    print("\n" + "=" * 60)
    print(f"Results saved to: profile_results.txt")
    print(f"Binary stats saved to: profile_tr.pstats")
    print("=" * 60)

    return pr

if __name__ == "__main__":
    run_profile()
