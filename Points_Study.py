from PyClasses.FEAssembly import *
from PyClasses.Contacts import *
from PyClasses.FEModel import *
import sys, os, pickle
import numpy as np
from time import time
import matplotlib.pyplot as plt

# POTATO
[ptt] = pickle.load(open("PointsCreation/PotatoAssembly.dat","rb"))
ptt.isRigid = True     # faster solving when True
ndofs = 3*len(ptt.X)

# MODEL
model = FEModel([ptt], [],[])           # [bodies, contacts, BCs, opts*]
ptt.Translate([-6.0, 0.0, 0.0])
model.X = ptt.X.ravel()
ptt.surf.ComputeGrgPatches(np.zeros(ndofs),range(len(ptt.surf.nodes)))

model_X = model.X.reshape(-1,3)
xa,xb = min(model_X[:,0]), max(model_X[:,0])
ya,yb = min(model_X[:,1]), max(model_X[:,1])
za,zb = min(model_X[:,2]), max(model_X[:,2])


offset = 0.1
dx, dy, dz = offset*np.array([xb-xa,yb-ya,zb-za])
xmin, xmax = xa-dx, xb+dx
ymin, ymax = ya-dy, yb+dy
zmin, zmax = za-dz, zb+dz

bounding_box = np.array([[xmin, ymin, zmin],
                         [xmin, ymin, zmax],
                         [xmin, ymax, zmin],
                         [xmin, ymax, zmax],
                         [xmax, ymin, zmin],
                         [xmax, ymin, zmax],
                         [xmax, ymax, zmin],
                         [xmax, ymax, zmax]])

# Connect the bounding box points
edges = [
    [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7],
    [4, 5], [4, 6], [5, 7], [6, 7]
]

# small visualization of the bounding box 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for edge in edges:
#     ax.plot3D(*zip(*bounding_box[edge]), color="r")
# plt.show()




####################################################-------------------------

# Testing points 
points_per_side = 2
xs = np.zeros((points_per_side*points_per_side*points_per_side , 3))

# Generate set of points in space
locator_index = 0
for x in np.linspace(xmin,xmax, points_per_side):
    for y in np.linspace(ymin,ymax, points_per_side):
        for z in np.linspace(zmin,zmax, points_per_side):
            xs[locator_index] = np.array([x,y,z])
            locator_index += 1

patches = model.bodies[0].surf.patches
npatches = len(patches)

# Trust-region parameters for projection
TR_init = 0.1   # initial radius as in original Python experiment
TR_min = 1e-12
TR_max = 1.0

# Precompute BS centers and flattened CtrlPts for all patches
xm_matrix = np.array([patch.BS.x for patch in patches])                # (npatches, 3)
CtrlPts_all = np.vstack([np.array(patch.flatCtrlPts()) for patch in patches])  # (npatches*20, 3)
radii = np.array([patch.BS.r for patch in patches], dtype=np.float64)  # (npatches,)

# Fully vectorized distance matrix for candidate selection
distance_matrix = np.linalg.norm(xm_matrix[:, np.newaxis, :] - xs[np.newaxis, :, :], axis=2)  # (npatches, npoints)

OUTMAT_final = np.zeros((len(xs),23))

stt = time()
for i, xsi in enumerate(xs):
    # Candidate patches for this point (nearest BS centers)
    sorted_indices = np.argsort(distance_matrix[:, i])
    candidates_i = sorted_indices[:5].astype(np.int32)  # choose nearest 45 patches

    # Use C++ backend to get best patch and parameters among candidates
    from PyClasses import gregory_patch_backend as gb
    best_patch, t1, t2, m_best = gb.find_projection_tr_multi(
        CtrlPts_all,
        xsi.astype(np.float64),
        candidates_i,
        radii,
        patches[0].eps,
        TR_init,
        TR_min,
        TR_max,
    )

    p_id = int(best_patch)
    print("Closest patch id:", p_id)
    t1t2_min = np.array([t1, t2])
    xc_min, dxcdt, d2xcd2t = patches[p_id].Grg(t1t2_min, deriv=2)
    D1p, D2p = dxcdt.T
    D3p = np.cross(D1p,D2p)
    nor = D3p/np.linalg.norm(D3p)
    normal_min = patches[p_id].D3Grg(t1t2_min)/np.linalg.norm(patches[p_id].D3Grg(t1t2_min))
    #
    gns_min = (xsi - xc_min) @ nor
    gns_min_check = (xsi - xc_min) @ normal_min
    #
    dfdt =  -2*np.tensordot(xsi-xc_min,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
    dfdxs = -2*dxcdt.T
    dtdxs = np.linalg.solve(-dfdt,dfdxs)

    # Vars for K---------------------------------------------------
    dndt = patches[p_id].dndt(t1t2_min)
    dndxs = dndt@dtdxs
    OUTMAT_final[i,:] = [int(p_id), t1t2_min[0], t1t2_min[1], xc_min[0], xc_min[1], xc_min[2], gns_min, gns_min_check, normal_min[0], normal_min[1], normal_min[2], \
                         nor[0], nor[1], nor[2], \
                         dndxs[0,0], dndxs[0,1], dndxs[0,2], dndxs[1,0], dndxs[1,1], dndxs[1,2], dndxs[2,0], dndxs[2,1], dndxs[2,2] ]
print("Total time for point projection:", time() - stt)
