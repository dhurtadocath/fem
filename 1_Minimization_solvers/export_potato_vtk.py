"""
export_potato_vtk.py — Export potato Gregory-patch surface as VTK for ParaView
===============================================================================
Tessellates all Gregory patches into a triangulated surface mesh and writes
a .vtu file (VTK XML UnstructuredGrid) that can be loaded in ParaView
alongside the simulation VTK output.

Usage:
    python export_potato_vtk.py                        # default: potato.vtu, res=30
    python export_potato_vtk.py --output my_potato.vtu --resolution 50
"""
import os, sys, pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Configuration ─────────────────────────────────────────────────────────────
output_file = "potato.vtu"      # output filename
resolution  = 30                # parametric grid resolution per patch (30x30)
# ──────────────────────────────────────────────────────────────────────────────

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load potato and compute Gregory patches
[ptt] = pickle.load(open("Dat/PotatoAssembly.dat", "rb"))
if hasattr(ptt, 'hexas') and not hasattr(ptt, 'elements'):
    ptt.elements = ptt.hexas
ptt.isRigid = True
n_nodes = len(ptt.X)
ptt.DoFs = np.array([[3*i, 3*i+1, 3*i+2] for i in range(n_nodes)])
ptt.surf.ComputeGrgPatches(np.zeros(3 * n_nodes), range(len(ptt.surf.nodes)))
patches = ptt.surf.patches
npatches = len(patches)

print(f"Potato: {npatches} Gregory patches")

# Tessellate
t_vals = np.linspace(0, 1, resolution)
npts_per_patch = resolution * resolution
ntri_per_patch = 2 * (resolution - 1) * (resolution - 1)

all_points = []
all_cells = []
all_patch_ids = []
offset = 0

for pid, patch in enumerate(patches):
    # Sample surface points on parametric grid
    pts = np.empty((npts_per_patch, 3))
    k = 0
    for i, u in enumerate(t_vals):
        for j, v in enumerate(t_vals):
            pts[k] = patch.Grg0(np.array([u, v], dtype=np.float64))
            k += 1
    all_points.append(pts)

    # Create triangles from quad grid
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            p00 = offset + i * resolution + j
            p10 = offset + (i + 1) * resolution + j
            p01 = offset + i * resolution + (j + 1)
            p11 = offset + (i + 1) * resolution + (j + 1)
            all_cells.append([p00, p10, p01])
            all_cells.append([p10, p11, p01])
    all_patch_ids.extend([pid] * ntri_per_patch)
    offset += npts_per_patch

points = np.vstack(all_points)
cells = np.array(all_cells, dtype=np.int32)
patch_ids = np.array(all_patch_ids, dtype=np.int32)

npts = len(points)
ncells = len(cells)

print(f"Tessellation: {npts} points, {ncells} triangles")

# Write VTK XML UnstructuredGrid (.vtu)
with open(output_file, "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <UnstructuredGrid>\n')
    f.write(f'    <Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">\n')

    # Cell data: patch ID
    f.write('      <CellData Scalars="patch_id">\n')
    f.write('        <DataArray type="Int32" Name="patch_id" format="ascii">\n')
    for i in range(0, ncells, 20):
        chunk = patch_ids[i:i+20]
        f.write("          " + " ".join(str(x) for x in chunk) + "\n")
    f.write('        </DataArray>\n')
    f.write('      </CellData>\n')

    # Points
    f.write('      <Points>\n')
    f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
    for pt in points:
        f.write(f"          {pt[0]:.10e} {pt[1]:.10e} {pt[2]:.10e}\n")
    f.write('        </DataArray>\n')
    f.write('      </Points>\n')

    # Cells (triangles = VTK type 5)
    f.write('      <Cells>\n')
    f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
    for tri in cells:
        f.write(f"          {tri[0]} {tri[1]} {tri[2]}\n")
    f.write('        </DataArray>\n')
    f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
    for i in range(0, ncells, 20):
        chunk = range(i, min(i + 20, ncells))
        f.write("          " + " ".join(str(3 * (j + 1)) for j in chunk) + "\n")
    f.write('        </DataArray>\n')
    f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
    for i in range(0, ncells, 20):
        chunk_size = min(20, ncells - i)
        f.write("          " + " ".join(["5"] * chunk_size) + "\n")
    f.write('        </DataArray>\n')
    f.write('      </Cells>\n')

    f.write('    </Piece>\n')
    f.write('  </UnstructuredGrid>\n')
    f.write('</VTKFile>\n')

print(f"Written: {output_file} ({os.path.getsize(output_file) / 1024:.0f} KB)")
