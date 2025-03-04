import meshio, sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
from math import pi
from scipy import sparse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyClasses.FEAssembly import *
from PyClasses.Contacts import *
from PyClasses.FEModel import *

from time import time
import argparse


# Argument parsing
parser = argparse.ArgumentParser(description='Process a data for the 2d contact model.')
parser.add_argument('--min_method', type=str, required=True, help='minimization method: BFGS, LBFGSNN')
parser.add_argument('--mesh', type=int, required=True, help='choose mesh 5, 10 or 15')
parser.add_argument('--plastic', type=int, required=True, help='boolean for plastic')

args = parser.parse_args()

# Calculate subspace bounds
minimization_method = args.min_method
mesh = args.mesh
plastic = args.plastic

# minimization_method = "BFGS"
# mesh = 15
# plastic = 1


####################
### GETTING DATA ###
####################
os.chdir(sys.path[0])

# # BLOCK
# mesh_blk   = meshio.read("../Meshes/Block_pseudo2d_"+str(mesh)+".msh")
# X_blk     = mesh_blk.points
# hexas_blk = mesh_blk.cells_dict['hexahedron']
nx=6
ny=2
nz=6
clx=0.4
cly=1.0
clz=0.4


LEN_node=nx*ny*nz
X_blk=np.zeros((LEN_node,3))
LEN_el=(nx-1)*(ny-1)*(nz-1)*6
hexas_blk=np.zeros((LEN_el,4))
cnt=0
cnt2=0
cntlb=-1
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            X_blk[cnt]=[i*clx, j*cly, k*clz]
            cnt=cnt+1
            cntlb=cntlb+1
            if i==nx-1 or j==ny-1 or k==nz-1:
                pass
            else:
                cn1=cntlb
                cn2=cntlb+1
                cn3=cntlb+1+nx
                cn4=cntlb+nx
                cn5=cntlb+nx*ny
                cn6=cntlb+nx*ny+1
                cn7=cntlb+nx*ny+nx+1
                cn8=cntlb+nx*ny+nx

                hexas_blk[cnt2:cnt2+6,:]=np.array([[cn4, cn1, cn6, cn5],
                                          [cn4, cn6, cn7, cn8],
                                          [cn4, cn5, cn6, cn8],
                                          [cn1, cn2, cn4, cn6],
                                          [cn4, cn6, cn3, cn7],
                                          [cn2, cn3, cn4, cn6]])
                print(cnt2)
                cnt2=cnt2+6








if plastic:
    blk = FEAssembly(X_blk,hexas_blk, name= "BLOCK",recOuters=False,plastic_param=[0.01,0.05,1.0])
else:
    blk = FEAssembly(X_blk,hexas_blk, name= "BLOCK",recOuters=False)

import pdb; pdb.set_trace()

blk.Youngsmodulus = 0.05
blk.Translate([-3,0.0,2.6])
blk.Resize(1.01)    # To avoid matching meshes with base


# # 2d-Base
mesh_base   = meshio.read("../Meshes/Base_pseudo2d.msh")
X_base = mesh_base.points
hexas_base = mesh_base.cells_dict['hexahedron']
base = FEAssembly(X_base,hexas_base, name = "BASE")
base.Resize(2.0,dir='y')
base.Resize(1.1,dir='x')
pickle.dump([base],open("Base2dAssembly.dat","wb"))
base.isRigid = True     # faster solving when True
base_top = base.SelectFlatSide("+z")
# Bump in the middle
base.X[21,2] += 0.5
base.X[39,2] += 0.5

ndofs = 3*(len(X_blk)+len(base.X))


######################
### BUILDING MODEL ###
######################

### Selections (nodes) ###
blk_bottom  = blk.SelectFlatSide("-z")
blk_top     = blk.SelectFlatSide("+z")
lead_face = blk.SelectFlatSide("-y")
front_face = blk.SelectFlatSide("+y")

slave_nodes = list(set(blk_bottom).intersection(set(lead_face)))

if mesh == 10 and "TR" in minimization_method:
    slave_nodes = slave_nodes+[143,152]   # Adding internal nodes to the slave nodes for the TR method mesh 10. Avoids unwanted deformations

blk_top     = list(set(blk_top   ).intersection(set(lead_face)))

same_pairs = np.zeros((len(lead_face),2),dtype=int)
for ii,node in enumerate(lead_face):
    x,y,z = blk.X[node]
    node_mirror = np.where((abs(blk.X[:,0]-x)<1e-10) & (abs(blk.X[:,2]-z)<1e-10) & (blk.X[:,1]!=y))[0][0]
    same_pairs[ii,:] = [node,node_mirror]


# Construction of Transformation matrices for pseudo-2d out of the 3d implementation
# Ns : Matrix used to enforce symmetry between nodes at -Y and +Y faces
# Nt : Matrix used to extract X and Z of the leader face (at -Y) in the 3d model as the pseudo-2d model
rs = np.zeros(2*(len(blk.X)))
cs = np.zeros(2*(len(blk.X)))
vs = np.ones((2*len(blk.X)))
rt = np.zeros((len(blk.X)))
ct = np.zeros((len(blk.X)))
vt = np.ones((len(blk.X)))

for i_p,pair in enumerate(same_pairs):
    node_1 = pair[0]
    node_2 = pair[1]
    rt[[2*i_p,2*i_p+1]]=[3*node_1,3*node_1+2]
    ct[[2*i_p,2*i_p+1]]=[2*i_p,2*i_p+1]
    rs[[4*i_p,4*i_p+1,4*i_p+2,4*i_p+3]]=[3*node_1,3*node_1+2,3*node_2,3*node_2+2]
    cs[[4*i_p,4*i_p+1,4*i_p+2,4*i_p+3]]=[3*node_1,3*node_1+2,3*node_1,3*node_1+2]

Ns = sparse.csr_matrix((vs,(rs,cs)),shape=(ndofs,ndofs))
Nt = sparse.csr_matrix((vt,(rt,ct)),shape=(ndofs,len(blk.X)))
N = [Ns,Nt]


### BOUNDARY CONDITIONS ###  [body, nodes, type, directions, values, times(*)]
cond_bd1 = [base, base.SelectAll(), "dirichlet", "xyz", [0.0, 0.0, 0.0]              ]      # Base: static and rigid
cond_bd2 = [blk , blk_top         , "dirichlet",  "xz", [0.0, 0.0,-0.25], [0.0, 0.1] ]      # Block: Indentation
cond_bd3 = [blk , blk_top         , "dirichlet",  "xz", [6.0, 0.0, 0.0 ], [0.1, 0.9] ]      # Block: Displacement
cond_bd4 = [blk , blk_top         , "dirichlet",  "xz", [0.0, 0.0, 0.25], [0.9, 1.0 ] ]      # Block: Indentation

cond_bd5 = [blk , blk.SelectAll() , "dirichlet",   "y", [0.0, 0.0,0.0]               ]      # Block: Symmetry

BCs = [cond_bd1, cond_bd2,cond_bd3,cond_bd4, cond_bd5]


### CONTACTS ###            # [body, nodes]
# For cases where sides enter into contact, uncomment this:
slave_x_plus  = list(set(blk.SelectFlatSide("+x")+blk.SelectFlatSide("-x")).intersection(set(lead_face)))
slave_x_plus  = list(set(blk.SelectLowerThan("z",3.0)).intersection(set(slave_x_plus)))
slave_nodes = list(set(slave_nodes+slave_x_plus))
slave  = [blk , slave_nodes    ]
master = [base, base_top]

contact1 = Contact(slave, master, kn=1e2, C1Edges = True, maxGN = 1e-5)       # (slave, master) inputs can be surfaces as well


### MODEL ###
subname = "_"+("plastic" if plastic else "elastic")+"_"+minimization_method+"_"+str(mesh)
model = FEModel([blk, base], [contact1], BCs,transform_2d=N,subname =subname )           # [bodies, contacts, BCs, opts*]

base.surf.ComputeGrgPatches(np.zeros(ndofs),base_top,exactNodesGiven=True)
# model.plotNow(as2D=True,OnlyMasterSurf=True)       # Uncomment to see and verify geometry


#############
## Running ##
#############

# import cProfile
# import pstats
# import io
# pr = cProfile.Profile()
# pr.enable()

t0 = time()

recov = "OUTPUT_202411302302pseudo2d_plastic_TR-icho_10/"+"RecoveryData.dat"
model.Solve(TimeSteps=100,max_iter=15, recover=False ,minimethod=minimization_method,plot=2,IterUpdate=1)

print("this took",time()-t0,"seconds to compute")


# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
# ps.print_stats()
# print(s.getvalue())
