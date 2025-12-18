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

# Iterate over points generated to get closest patch and surface parameters
patches = model.bodies[0].surf.patches
npatches = len(patches)
raw_distances = np.zeros(npatches)

xm_matrix = np.zeros((npatches,3))
for ip,patch in enumerate(patches):
    xm_matrix[ip] = patch.BS.x
distance_matrix = np.linalg.norm(xm_matrix[:, np.newaxis, :] - xs, axis=2)


candidates = []
for i,xsi in enumerate(xs):
    
    raw_distances = distance_matrix[:,i]
    sorted_indices = np.argsort(raw_distances)
    candidates.append(sorted_indices[0:5])
candidates = np.array(candidates)

# TR method to find closest point on surface patch
t1t2_init = [[0,0],[0.5,0],[1,0],[0,0.5],[0.5,0.5],[1,0.5],[0,1.0],[0.5,1.0],[1,1.0]]
eps_min = 1e-15

TR_max = 1.0
TR_min = 1e-12 #1e-2 #1e-12
TR_init = 0.1
OUTMAT_final = np.zeros((len(xs),23))

# Diagnostics: count total outer TR iterations (for profiling)
TR_outer_iters_total = 0

stt = time()
for i,xsi in enumerate(xs):
    # t1t2 = np.zeros((len(candidates[i]),2))
    # gns = np.zeros(len(candidates[i])) 
    OUTMAT=np.zeros((len(candidates[i]),4))
    for ip,patch in enumerate(candidates[i]):
        outmat = np.zeros((9,3))
        for j in range(9):
            t1t2 = t1t2_init[j]
            xc = patches[patch].Grg0(t1t2)
            m_new = np.linalg.norm(xsi - xc)**2
            m_old = 1e10
            flag_new_u = 1.0
            cnt1 = 0
            
            TR_radius = TR_init #0.000001#0.01
            while m_new < m_old:
                cnt1 += 1
                TR_outer_iters_total += 1
                #print("cnt1", cnt1)
                # print("t1t2", t1t2)
                if flag_new_u ==1:
                    xc, dxcdt, d2xcd2t = patches[patch].Grg(t1t2, deriv=2)
                    f = 2*(xc - xsi) @ dxcdt
                    # print(xc.shape)
                    # print((xc - xsi) @ d2xcd2t)
                    # print(np.tensordot(xsi-xc,d2xcd2t,axes=1).shape)
                    # print(d2xcd2t.shape)
                    # print(dxcdt.shape)
                    # print((dxcdt @ dxcdt).shape)
                    # print(dxcdt.T @ dxcdt)
                    # print(np.einsum('i,ijk->jk', (xc - xsi), d2xcd2t).shape)
                    K = 2*(dxcdt.T @ dxcdt + np.einsum('i,ijk->jk', (xc - xsi), d2xcd2t))
                r = -f
                p = r
                q= p 
                h = np.zeros(2)
                flag_boundary_reached = 0
                bla = 1.0
                # cnt2 = 0
                while bla > 0:
                    # cnt2 += 1
                    # print("cnt2", cnt2)
                    if np.dot(np.dot(p.T, K),p) <= 0:
                        flag_boundary_reached = 1
                        a = np.dot(p.T, p)
                        b = 2*np.dot(p.T, h)
                        c = np.dot(h.T, h) - TR_radius**2
                        alpha = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                        h = h + alpha*p
                        break
                    alpha = np.dot(r.T, q)/ np.dot(np.dot(p.T, K),p)
                    if np.dot((h + alpha*p).T, (h + alpha*p)) >= TR_radius**2:
                        flag_boundary_reached = 1
                        a = np.dot(p.T, p)
                        b = 2*np.dot(p.T, h)
                        c = np.dot(h.T, h) - TR_radius**2
                        alpha = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                        h = h + alpha*p
                        break
                    h = h + alpha*p
                    phi = np.dot(r.T, p)
                    r = r - alpha * np.dot(K, p)
                    # print(np.max([1.0e-15, 1.0e-5 * np.linalg.norm(f)]))
                    # print("ffffff", np.linalg.norm(f))
                    # print("ffffff", np.linalg.norm(r))
                    # print("ffffff", 1.0e-5 * np.linalg.norm(f))
                    if np.linalg.norm(r) < np.max([1e-15, 1.0e-5 * np.linalg.norm(f)]):
                        break 
                    q = r
                    p = q + (np.dot(r.T, q))/(phi) * p
                m_new_plus_h = np.linalg.norm(xsi - patches[patch].Grg0(t1t2 + h))**2
                ratio  = (m_new - m_new_plus_h) / (-np.dot(f.T, h) - 0.5 *np.dot(np.dot(h.T, K),h))
                if ratio < 0.25:
                    TR_radius = 0.25 * TR_radius
                    flag_new_u = 0.0
                else: 
                    t1t2 += h
                    if ratio > 0.75 and flag_boundary_reached == 1:
                        TR_radius = min(2.0 * TR_radius, TR_max) 
                    flag_new_u = 1.0
                if (TR_radius < TR_min) | (t1t2[0] < -0.5) | (t1t2[0] > 1.5) | (t1t2[1] < -0.5) | (t1t2[1] > 1.5): #np.linalg.norm(f)< 1e-12: #TR_radius < TR_min:
                    #print("TR radius below minimum","\n", 
                     #     "xsi", xsi, "patch_id", patch, "t1t2_init", t1t2_init[j])
                    break
                if flag_new_u == 1.0:
                    m_old = m_new
                    m_new = m_new_plus_h   

            outmat[j,:] = [m_new, t1t2[0], t1t2[1]]
        #
        # print(((outmat[:,1]>=eps_min) and (outmat[:,1]<=1.0)) and ((outmat[:,2]>=eps_min) and (outmat[:,2]<=1.0)))
        # outmat_filtered = outmat[((outmat[:,1]>=eps_min) and (outmat[:,1]<=1.0)) and ((outmat[:,2]>=eps_min) and (outmat[:,2]<=1.0))]
        outmat_filtered = outmat[ ((outmat[:, 1] >= eps_min) & (outmat[:, 1] <= 1.0)) &     ((outmat[:, 2] >= eps_min) & (outmat[:, 2] <= 1.0))]
        # print("outmat_filtered", outmat_filtered)
        if len(outmat_filtered) ==0:
            OUTMAT[ip,:] = [1e10, -1.0, -1.0, candidates[i][ip]]
        else:    
            sorted_outmat = outmat_filtered[np.argsort(outmat_filtered[:,0])]
            # OUTMAT[ip,:] = [sorted_outmat[0,:],candidates[ip]]
            OUTMAT[ip,:] = [sorted_outmat[0,0], sorted_outmat[0,1], sorted_outmat[0,2], candidates[i][ip]]
            
    idx_min = np.argmin(OUTMAT[:,0])
    p_id = int(OUTMAT[idx_min,3])
    print("Closest patch id:", p_id)
    t1t2_min = OUTMAT[idx_min,1:3]
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
print("Total TR outer iterations (v0):", TR_outer_iters_total)
