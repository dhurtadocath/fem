# Verification & Optimization Checklist — ContactPotato_NGSolve.py

**Created**: 2026-02-07
**Source**: Comprehensive verification against academic literature, NGSolve API source, and reference implementation (PyClasses/).
**Script**: `1_Minimization_solvers/ContactPotato_NGSolve.py`

---

## Tier 1 — High Impact (precision + performance)

### [x] 1. Add `TaskManager()` for parallel NGSolve operations
- **Type**: PERF | **Expected**: 2-8x on large meshes
- **Status**: DONE — exposed via `--taskmanager` CLI flag
- **Notes from testing**:
  - mesh=5 (648 DOFs): TaskManager adds overhead, **slower** (serial is faster for micro-problems)
  - mesh=15 (12k DOFs): beneficial, NGSolve operations dominate at ~105ms/eval
  - Follows canonical `with TaskManager():` pattern (NGSolve `howto_parallel.rst`)
  - The flag design is correct: users enable it for production meshes (>~5k DOFs)

### [x] 2. Use `realcompile=True` on energy form
- **Type**: PERF | **Expected**: 2-5x on assembly for large meshes
- **Status**: DONE — exposed via `--realcompile` CLI flag
- **Notes from testing**:
  - mesh=5: large startup cost (~5-10s JIT C++ compilation), **slower overall**
  - mesh=15: compilation cost amortized over ~200 evals/step, net positive
  - Pattern: `Variation(psi_sym.Compile(realcompile=True, wait=True) * dx)`
  - `wait=True` ensures C++ compilation finishes before first `Apply()`/`Energy()` call

### [x] 3. Add energy-based linesearch to Newton solver
- **Type**: PRECISION/ROBUSTNESS | **Impact**: prevents divergence and singular matrix crashes
- **Status**: DONE — backtracking linesearch following NGSolve `NewtonSolver.Solve()` pattern
- **Implementation**: `_total_energy_at()` evaluates material + contact energy at trial point.
  Step halved while `E(u - tau*w) > E(u) + tol` (same criterion as `nonlinearsolvers.py:66`).
- **Fallback**: on singular tangent matrix (UMFPACK failure), falls back to scaled gradient
  descent step with linesearch.
- **Results**:
  - mesh=5: Newton no longer crashes at step 2 (was fatal before). All 10 steps complete.
  - mesh=5: Newton is ~4x faster per step than L-BFGS-B (2.4s vs 9s)
  - mesh=10: still hits singular tangent occasionally — fallback handles it but convergence
    degrades. Root cause is likely element inversion (J<0) during large load steps, needs
    further investigation.
- **Open issue**: active-set oscillation (steps 3-8 on mesh=5 don't fully converge in 5 AS
  cycles). Increasing `--max_as 15` helps but the contact equilibrium is hard to reach.
  Consider augmented Lagrangian or better active-set strategy.

### [x] 4. Cache `AssembleLinearization` in `hessp`
- **Type**: PERF | **Expected**: 2-10x for Newton-CG solver
- **Status**: DONE — `_hessp_cache` dict caches both tangent assembly and contact state
- **Implementation**: `_hessp_cache["x_free"]` stores the last linearization point; only
  reassembles when `np.array_equal(x_free, cached)` is False. Also caches contact state
  (gn, normals, active_idx, dndxs_all) and precomputes slave-DOF-to-free-DOF mapping.
- **Results** (mesh=5, 30 steps, newton-cg):
  - ~95% of hessp calls reuse cached tangent (e.g., 801 calls / 37 assemblies in step 1)
  - Each assembly costs ~22ms (mesh=5) vs ~0.12ms for mat*vec → 190x ratio
  - Step 1 alone saves ~17s of assembly time (764 avoided × 22ms)
- **Comparison** (mesh=10, 30 steps):
  - Newton-CG: 199.4s, all 30 steps complete, stable with up to 50 active nodes
  - L-BFGS-B: diverges at step 7, crashes at step 17 (NaN propagation)

---

## Tier 2 — Medium Impact (robustness + performance)

### [x] 5. Compile stress/VM CoefficientFunctions for VTK
- **Type**: PERF | **Expected**: 1.5-3x faster VTK export
- **Status**: DONE — all stress tensors and von Mises CFs now use `.Compile()`
- **Notes**: only affects VTK export cost, not solver performance

### [x] 6. Reuse symbolic factorization (UMFPACK `.Update()`)
- **Type**: PERF | **Expected**: ~30-50% faster linear solve in Newton
- **Status**: DONE — cache UMFPACK inverse, reuse symbolic factorization via `inv.Update()`
- **Implementation**: `_cached_inv` created on first Newton iteration via `mat.Inverse()`.
  On subsequent iterations, after `AssembleLinearization` + contact Hessian + regularization
  modify matrix values in-place, `_cached_inv.Update()` redoes only numeric factorization
  (reuses the symbolic factorization — elimination tree, permutation, etc.).
  Cache invalidated on exception (singular matrix) and re-created on next attempt.
- **Note**: Used UMFPACK (not sparsecholesky) because `UmfpackInverse.SupportsUpdate()==True`
  and it handles non-symmetric matrices (contact Hessian breaks symmetry). SparseCholesky
  also supports Update but only for symmetric matrices.
- **Results** (newton, 30 steps):
  - mesh=5: 18.7s (was 17.9s — within noise, too few DOFs for symbolic to matter)
  - mesh=10: **112.6s (was 142.3s — 21% faster)**
  - Residuals at steps 1-2 match exactly: |r|=2.99e-10 and 1.78e-05

### [x] 7. Mesh-dependent penalty `kn = factor * E / h`
- **Type**: PRECISION | **Impact**: correct scaling across mesh densities
- **Status**: DONE — `h_contact = 4.0 / n; kn = kn_factor * E / h`
- **Standard**: `kn ~ alpha * E / h` (Wriggers 2006), kn_factor=20 default
- **Results** (newton-cg, 30 steps):
  - mesh=5: kn 1.0→1.25, maxpen 11.0mm→9.5mm, time 45.1s→29.8s
  - mesh=10: kn 1.0→2.5, maxpen 4.7mm→1.9mm (**2.5x reduction**), time 199.4s→150.8s
  - Penetration now scales consistently across mesh densities

### [x] 8. Vectorize contact force assembly
- **Type**: PERF | **Expected**: minor for small active sets, noticeable for large ones
- **Status**: DONE — vectorized residual scatter and hessp Hv product
- **Changes**:
  - Precomputed `_slave_free_x/y/z` index arrays (replace dict lookup)
  - `hessp()`: batch `kn*(n·p)*n` via numpy broadcasting + `np.add.at` scatter
  - Newton inner loop: collect arrays during projection, vectorized force scatter
  - Hessian `mat[i,j]` writes left sequential (NGSolve sparse matrix API limitation)
- **Results**: mesh=5 29.5s→27.6s (7%), mesh=10 150.8s→147.9s (2%).
  Marginal on current mesh sizes — benefit grows with larger active sets.

---

## Tier 3 — Low Impact (code quality + safety)

### [x] 9. Deduplicate contact force computation
- **Type**: CODE | **Impact**: maintainability
- **Status**: DONE — extracted `compute_contact_forces(gn, normals, active)` helper
- **Changes**: replaced 3 identical 7-line blocks in `objective()`, `finalize_contact_state()`,
  and Newton post-processing with single-line calls to `compute_contact_forces()`.
  Newton inner loop kept separate (different pattern: per-node from cached projections + Hessian).

### [x] 10. Fix sparsity assumption on `mat[i,j]` writes for contact Hessian
- **Type**: CODE/ROBUSTNESS | **Impact**: prevent silent bugs with different energy models
- **Status**: DONE — one-time assertion using `mat.COO()` on first contact vertex
- **Investigation findings** (test_sparsity.py):
  - `mat[i,j]` read on missing entry returns 0 silently (no error)
  - `mat[i,j] = val` write on missing entry **raises** `"sparse matrix row full"` (not silent)
  - VectorH1 hyperelastic form always couples all 3 components at same vertex (confirmed via COO)
  - So the current code already crashes loudly on wrong sparsity, but with a cryptic message
- **Implementation**: On first Newton iteration with contact, extract COO from `a_form.mat`,
  build (row,col) set, assert all 9 entries of the 3×3 block `[v, v+nv, v+2*nv]²` exist.
  Clear error message explains the coupling requirement. Check runs once, costs ~0.

### [x] 11. Guard `hessp` side-effect on `gfu.vec`
- **Type**: CODE | **Impact**: defensive robustness
- **Status**: DONE — save/restore `gfu.vec` free DOFs around the reassembly in `hessp()`
- **Problem**: `hessp()` modified `gfu.vec.FV().NumPy()[free_dofs] = x_free` as a global
  side effect. Safe for current scipy Newton-CG usage but fragile if other code reads
  `gfu.vec` between hessp calls (e.g., logging, energy evaluation, postprocessing).
- **Fix**: Save `_saved_free = vec[free_dofs].copy()` before modifying, restore after
  `AssembleLinearization` and contact evaluation. Cost: one small array copy per reassembly
  (only on the `need_reassemble` path, not on cached CG inner iterations).

### [x] 12. Vectorize KD-tree surface sampling setup
- **Type**: PERF | **Impact**: marginal (one-time init cost)
- **Status**: DONE — pre-allocate arrays, meshgrid parametric grid, call C++ backend directly
- **Changes**: Replaced triple-nested Python loop with list.append by:
  (1) `np.meshgrid` + `ravel` for the (u,v) sample grid (2500 pairs, computed once)
  (2) Pre-allocated `surf_pts` and `surf_pids` arrays (no append overhead)
  (3) Direct `gb.Grg(ctrlpts, u, v, eps)` call bypassing `Grg0` → `Grg` Python dispatch
  (4) `_flatCtrlPts_array()` and `eps` cached per-patch (one lookup instead of 2500)

---

## Critical Finding: Contact Projection is the Bottleneck (L-BFGS-B)

**Profiling results** (mesh=5, L-BFGS-B, 200 iterations/step):

| Component | Time | % |
|-----------|------|---|
| NGSolve (Energy + Apply) | 3.9s | 2% |
| Contact (TR projection) | 194s | **98%** |
| Other | 0.1s | <1% |

**Root cause**: `ContactCache.evaluate()` with `tol_reuse=0.05` — when L-BFGS-B takes steps
larger than 0.05 (in any slave node position), the cache misses and ALL slave nodes undergo
full TR projection (~5ms/node, 36 nodes = 180ms/eval).

**Cache behavior observed**:
- eval #1 after `reset()`: 0/36 reused (expected — forced full projection at step start)
- eval #2-#5 (same step, small L-BFGS-B steps): 36/36 reused (cache working)
- eval #100 (later in step, larger steps): 0/36 reused (cache failing)

**On mesh=15** (your output): contact is only 35% of time because:
- More DOFs → smaller per-node displacement increments → better cache hit rate
- NGSolve operations scale with mesh size and become the dominant cost

**Possible improvements** (not yet implemented):
- Increase `tol_reuse` adaptively (start tight, loosen if cache miss rate is high)
- Project only slave nodes near the surface (skip nodes far from potato)
- Re-project only the nodes that moved more than threshold, not all-or-nothing

---

## Critical Finding: Singular Tangent on mesh=10+ — FIXED

The Newton solver previously crashed on mesh=10 at step 3 due to singular tangent matrix
(UMFPACK failure). NaN from the failed solve propagated to all subsequent steps.

**Root causes identified**:
1. UMFPACK has two failure modes: (a) raises `NgException` on hard singular, (b) silently
   returns NaN on near-singular (only prints WARNING). The original `try/except` only
   caught case (a).
2. Gradient descent fallback produced NaN when state was already corrupted.
3. Linesearch accepted NaN steps (`energy_new > energy_old + tol` evaluates to False when
   both are NaN).
4. No recovery mechanism — once NaN entered `gfu.vec`, all subsequent steps were corrupted.

**Fixes implemented** (multiple safeguards):
1. `_total_energy_at()` returns `np.inf` for NaN/inf states (rejects bad trial points)
2. Linesearch: explicit `np.isfinite(energy_new)` check before accepting; if no step is
   accepted, keeps previous state (never writes NaN to `gfu.vec`)
3. Steepest descent fallback: physically scaled step size (`0.1 * h_contact / max|r|`),
   NaN guard on residual
4. NaN detection at active-set loop entry — breaks early if state is corrupted
5. Step-level NaN recovery: if Newton produces NaN, restores `_u_prev_vec` (state from
   before the step) + re-applies Dirichlet increment

**Results**:
- mesh=5, 30 steps: 17.9s, all complete, no crashes (same as before)
- mesh=10, 30 steps: **142.3s, all 30 steps complete** (previously crashed at step 3)
- Contact steps (3-25) have residuals ~0.3-3.6 and penetration ~0.1-0.4 due to active-set
  not converging in 5 iterations (fundamental limitation, not a bug)
- Newton-CG on same problem: 149.0s, |grad|<1e-7, maxpen~0.002 (much better convergence)

**Remaining limitation**: Active-set oscillation during contact. The direct Newton solver
doesn't fully converge the contact equilibrium. Newton-CG converges 100-200x better
penetration because it minimizes the combined energy. Augmented Lagrangian or semi-smooth
Newton would be needed for true active-set convergence in the direct Newton path.

---

## Academic Literature Comparison

| Aspect | This Code | Standard Reference | Status |
|--------|-----------|-------------------|--------|
| Energy | Ciarlet Neo-Hookean | Bonet & Wood (2008), Holzapfel (2000) | CORRECT |
| Contact | Node-to-surface penalty | Wriggers (2006), Laursen (2002) | CORRECT |
| Active set | Outer loop, fixed inner | Wriggers (2006) Ch. 9 | CORRECT |
| Hessian curvature | dn/dxs via IFT, capped | Laursen (2002) Ch. 3 | CORRECT |
| kn selection | kn = alpha*E/h (mesh-dep) | kn ~ alpha*E/h | DONE |
| Newton | Linesearch added | Bonet & Wood damped Newton | DONE |
| Variation() | Auto-diff | NGSolve standard (Schoberl et al.) | CORRECT |
| TaskManager | CLI flag | NGSolve howto_parallel.rst | DONE |
| realcompile | CLI flag | NGSolve i-tutorials | DONE |

---

## Sources
- [Neo-Hookean solid](https://en.wikipedia.org/wiki/Neo-Hookean_solid)
- [FreeFEM Compressible Neo-Hookean](https://doc.freefem.org/models/compressible-neo-hookean-materials.html)
- [Bower Ch3.5 Hyperelasticity](https://solidmechanics.org/text/Chapter3_5/Chapter3_5.htm)
- [NGSolve py_tutorials/elasticity.py](https://github.com/NGSolve/ngsolve)
- [NGSolve i-tutorials](https://docu.ngsolve.org/latest/i-tutorials/index.html)
- Bonet & Wood, *Nonlinear Continuum Mechanics for FEA*, 2nd ed., Cambridge, 2008
- Holzapfel, *Nonlinear Solid Mechanics*, Wiley, 2000
- Wriggers, *Computational Contact Mechanics*, 2nd ed., Springer, 2006
- Laursen, *Computational Contact and Impact Mechanics*, Springer, 2002
