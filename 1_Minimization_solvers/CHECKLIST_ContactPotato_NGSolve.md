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

### [ ] 6. Reuse symbolic factorization (`sparsecholesky` + `.Update()`)
- **Type**: PERF | **Expected**: ~30-50% faster linear solve in Newton
- **Status**: TODO
- **Problem**: `mat.Inverse(fes.FreeDofs(), inverse="umfpack")` is called every Newton
  iteration, recomputing the symbolic factorization each time.
- **Fix**: use `"sparsecholesky"` with `.Update()` method that only does numeric refactorization
  when sparsity pattern is unchanged (which it is — only values change between Newton steps).
- **Note**: the sparsity pattern includes contact entries only if the hyperelastic energy
  already couples those DOFs. Need to verify pattern stability.

### [x] 7. Mesh-dependent penalty `kn = factor * E / h`
- **Type**: PRECISION | **Impact**: correct scaling across mesh densities
- **Status**: DONE — `h_contact = 4.0 / n; kn = kn_factor * E / h`
- **Standard**: `kn ~ alpha * E / h` (Wriggers 2006), kn_factor=20 default
- **Results** (newton-cg, 30 steps):
  - mesh=5: kn 1.0→1.25, maxpen 11.0mm→9.5mm, time 45.1s→29.8s
  - mesh=10: kn 1.0→2.5, maxpen 4.7mm→1.9mm (**2.5x reduction**), time 199.4s→150.8s
  - Penetration now scales consistently across mesh densities

### [ ] 8. Vectorize contact force assembly
- **Type**: PERF | **Expected**: minor for small active sets, noticeable for large ones
- **Status**: TODO
- **Problem**: Python loops over active nodes for both residual and Hessian modification
- **Fix for residual** (easy): already partially vectorized (lines 504-510 use fancy indexing)
- **Fix for Hessian** (harder): the 9 `mat[dofs[a], dofs[b]]` writes per node are inherently
  sequential due to NGSolve sparse matrix access pattern

---

## Tier 3 — Low Impact (code quality + safety)

### [ ] 9. Deduplicate contact force computation
- **Type**: CODE | **Impact**: maintainability
- **Status**: TODO
- **Problem**: contact force assembly is written 3 times: in `objective()`, in
  `newton_active_set_solve()` post-processing, and in `finalize_contact_state()`
- **Fix**: extract to a single helper function

### [ ] 10. Fix sparsity assumption on `mat[i,j]` writes for contact Hessian
- **Type**: CODE/ROBUSTNESS | **Impact**: prevent silent bugs with different energy models
- **Status**: TODO
- **Problem**: `mat[dofs[a], dofs[b]] = mat[dofs[a], dofs[b]] + K_con[a,b]` assumes the entry
  exists in the sparsity pattern. This is true for hyperelastic energy (couples all components
  of the same vertex) but would silently produce wrong results for decoupled/linear elastic
  models where off-diagonal blocks (x-y, x-z, y-z) are absent from the sparsity.
- **Fix (preferred)**: use `BaseMatrix.AddElementMatrix()` which safely handles sparsity:
  ```python
  # dof_ids = [v, v+nv, v+2*nv] as IntRange or list
  # K_con_3x3 = FlatMatrix(3, 3, ...) or create via Matrix(3,3)
  mat.AddElementMatrix(dof_ids, dof_ids, K_con_mat, replace=False)
  ```
  Requires verifying AddElementMatrix Python binding signature in NGSolve source.
- **Fix (fallback)**: add a runtime assertion on first call to detect missing entries:
  ```python
  if _first_contact_hessian_write:
      for a in range(3):
          for b in range(3):
              if mat[dofs[a], dofs[b]] == 0.0 and K_con[a,b] != 0.0:
                  assert False, f"Sparsity pattern missing entry ({dofs[a]},{dofs[b]})"
  ```

### [ ] 11. Guard `hessp` side-effect on `gfu.vec`
- **Type**: CODE | **Impact**: defensive robustness
- **Status**: TODO
- **Problem**: `hessp()` modifies `gfu.vec.FV().NumPy()[free_dofs] = x_free` which is a global
  side effect. Safe for current scipy Newton-CG usage but fragile.

### [ ] 12. Vectorize KD-tree surface sampling setup
- **Type**: PERF | **Impact**: marginal (one-time init cost ~0.5s)
- **Status**: TODO — low priority since it only runs once at startup

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

## Critical Finding: Singular Tangent on mesh=10+

The Newton solver hits singular tangent matrix (UMFPACK failure) on mesh=10 at step 2. The
linesearch prevents divergence but cannot fix a genuinely singular tangent.

**Likely causes**:
1. Element inversion (J<0) during large load steps — the linesearch should prevent this but
   the energy tolerance `max(1e-14*|E|, gtol)` may be too permissive
2. Near-zero stiffness in lateral directions — the block is only constrained on "top" face,
   with "bottom" slave nodes free to slide. Without contact, the structure has near-zero
   lateral stiffness.
3. Contact Hessian `kn*(n x n)` adds rank-1 updates per active node — if contact is lost
   between active-set iterations, the stiffness suddenly drops.

**Possible fixes** (not yet implemented):
- Add small regularization to diagonal: `mat[i,i] += eps` for slave DOFs
- Use `"sparsecholesky"` which handles near-singular better than UMFPACK
- Check for J<0 before accepting linesearch step
- Reduce load increment size (increase `--nsteps`)

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
