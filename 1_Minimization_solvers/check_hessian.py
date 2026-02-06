"""
Diagnostic script to check the contact Hessian computation.

The contact energy is: E = (1/2) * kn * g^2
where g = (x_s - x_c(t)) · n(t) and t = t(x_s) is the projection parameter.

Gradient: ∂E/∂x_s = kn * g * ∂g/∂x_s

Hessian: ∂²E/∂x_s² = kn * (∂g/∂x_s ⊗ ∂g/∂x_s + g * ∂²g/∂x_s²)

Key question: Is ∂n/∂x_s the same as ∂²g/∂x_s²? NO!

Let's derive ∂g/∂x_s properly:
g = (x_s - x_c) · n
∂g/∂x_s = n - (∂x_c/∂x_s)ᵀ n + (∂n/∂x_s)ᵀ (x_s - x_c)
        = n - (∂x_c/∂x_s)ᵀ n + g * (∂n/∂x_s)ᵀ n

Since n is a unit vector: nᵀ (∂n/∂x_s) = 0 (derivative perpendicular to n)
But (∂n/∂x_s)ᵀ n is NOT zero in general!

Let's check this numerically.
"""

import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyClasses import gregory_patch_backend as gb

# Load potato
[ptt] = pickle.load(open("Dat/PotatoAssembly.dat", "rb"))
if hasattr(ptt, 'hexas') and not hasattr(ptt, 'elements'):
    ptt.elements = ptt.hexas
ptt.isRigid = True
n_ptt_nodes = len(ptt.X)
ndofs_ptt = 3 * n_ptt_nodes
ptt.DoFs = np.array([[3*i, 3*i+1, 3*i+2] for i in range(n_ptt_nodes)])
ptt.surf.ComputeGrgPatches(np.zeros(ndofs_ptt), range(len(ptt.surf.nodes)))
patches = ptt.surf.patches

print("Checking contact Hessian derivation...")
print("="*60)

# Pick a test point (penetrating into the surface)
pid = 10  # arbitrary patch
t = np.array([0.5, 0.5])  # center of patch
patch = patches[pid]

# Get surface point and derivatives
xc, dxcdt, d2xcd2t = patch.Grg(t, deriv=2)
n_raw = patch.D3Grg(t)
n = n_raw / np.linalg.norm(n_raw)

# Create a test slave position (slightly penetrating)
g = -0.1  # penetration
xs = xc + g * n  # slave position

print(f"Patch {pid}, t = {t}")
print(f"Surface point xc = {xc}")
print(f"Normal n = {n}")
print(f"Gap g = {g}")
print(f"Slave position xs = {xs}")
print()

# Compute ∂t/∂x_s via implicit function theorem
delta = xs - xc
dfdt = -2 * np.tensordot(delta, d2xcd2t, axes=1) + 2 * (dxcdt.T @ dxcdt)
dfdxs = -2 * dxcdt.T
dtdxs = np.linalg.solve(-dfdt, dfdxs)  # (2, 3)

print("∂t/∂x_s (2×3):")
print(dtdxs)
print()

# Compute ∂x_c/∂x_s
dxcdxs = dxcdt @ dtdxs  # (3,2) @ (2,3) = (3,3)
print("∂x_c/∂x_s (3×3):")
print(dxcdxs)
print(f"Is symmetric? {np.allclose(dxcdxs, dxcdxs.T)}")
print()

# Compute ∂n/∂x_s
dndt = patch.dndt(t)  # (3, 2)
dndxs = dndt @ dtdxs  # (3, 3)
print("∂n/∂x_s (3×3):")
print(dndxs)
print(f"Is symmetric? {np.allclose(dndxs, dndxs.T)}")
eigs = np.linalg.eigvalsh(0.5*(dndxs + dndxs.T))
print(f"Eigenvalues of symmetric part: {eigs}")
print()

# Check: nᵀ (∂n/∂x_s) should be zero (derivative perpendicular to n)
nT_dndxs = n @ dndxs
print(f"nᵀ @ (∂n/∂x_s) = {nT_dndxs} (should be ~0)")
print()

# Check: (∂n/∂x_s)ᵀ @ n
dndxs_T_n = dndxs.T @ n
print(f"(∂n/∂x_s)ᵀ @ n = {dndxs_T_n}")
print()

# Now compute the FULL ∂g/∂x_s
# g = (x_s - x_c) · n
# ∂g/∂x_s = n - (∂x_c/∂x_s)ᵀ @ n + (∂n/∂x_s)ᵀ @ (x_s - x_c)
dgdxs_full = n - dxcdxs.T @ n + dndxs.T @ delta
print(f"Full ∂g/∂x_s = {dgdxs_full}")
print(f"Simple approximation (just n) = {n}")
print(f"Difference = {dgdxs_full - n}")
print()

# The Hessian should be:
# K = kn * (∂g/∂x_s ⊗ ∂g/∂x_s + g * ∂²g/∂x_s²)

# Using the simple approximation ∂g/∂x_s ≈ n:
kn = 1.0  # penalty parameter
K_simple = kn * np.outer(n, n)
print("Simple Hessian (n ⊗ n):")
print(K_simple)
print(f"Eigenvalues: {np.linalg.eigvalsh(K_simple)}")
print()

# Using the formula K = kn * (n ⊗ n + g * ∂n/∂x_s) [CURRENT IMPLEMENTATION]
K_current = kn * (np.outer(n, n) + g * dndxs)
print("Current implementation (n ⊗ n + g * ∂n/∂x_s):")
print(K_current)
print(f"Is symmetric? {np.allclose(K_current, K_current.T)}")
print(f"Eigenvalues: {np.linalg.eigvalsh(0.5*(K_current + K_current.T))}")
print()

# Using the CORRECT formula with full ∂g/∂x_s
K_correct_1st = kn * np.outer(dgdxs_full, dgdxs_full)
print("First term (∂g/∂x_s ⊗ ∂g/∂x_s):")
print(K_correct_1st)
print(f"Eigenvalues: {np.linalg.eigvalsh(K_correct_1st)}")
print()

# For the second term, we need ∂²g/∂x_s² which is complex
# Let's compute it numerically
eps = 1e-6
d2gdxs2_numerical = np.zeros((3, 3))
for i in range(3):
    xs_plus = xs.copy()
    xs_plus[i] += eps
    xs_minus = xs.copy()
    xs_minus[i] -= eps

    # Recompute gap at perturbed positions
    # For simplicity, use the same projection (frozen t)
    g_plus = (xs_plus - xc) @ n
    g_minus = (xs_minus - xc) @ n
    dgdxs_i = (g_plus - g_minus) / (2 * eps)

    # This gives us the i-th column of ∂g/∂x_s (with frozen projection)
    # For ∂²g/∂x_s², we need to differentiate again...

# Actually, with frozen projection, ∂g/∂x_s = n and ∂²g/∂x_s² = 0
# The curvature term only appears when we account for the moving projection

print("="*60)
print("CONCLUSION:")
print("="*60)
print()
print("The current implementation uses: K = kn * (n ⊗ n + g * ∂n/∂x_s)")
print()
print("Issues found:")
print("1. ∂n/∂x_s is NOT symmetric in general")
print("2. The formula g * ∂n/∂x_s is NOT the same as g * ∂²g/∂x_s²")
print("3. The correct first term should use ∂g/∂x_s, not just n")
print()
print("For the Hessian to be symmetric and positive semi-definite,")
print("we should either:")
print("  a) Use only n ⊗ n (frozen projection, always PSD)")
print("  b) Symmetrize the curvature term: g * 0.5*(∂n/∂x_s + (∂n/∂x_s)ᵀ)")
print("  c) Use the full correct formula with ∂g/∂x_s and ∂²g/∂x_s²")
