
import meshio, sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
from math import pi

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

# BLOCK
mesh_blk   = meshio.read("../Meshes/Cubes/cube"+str(mesh)+"x"+str(mesh)+"x"+str(mesh)+".msh")

X_blk     = mesh_blk.points
hexas_blk = mesh_blk.cells_dict['hexahedron']
if plastic:
    blk = FEAssembly(X_blk,hexas_blk, name= "BLOCK",recOuters=False,plastic_param=[0.01,0.05,1.0])
else:
    blk = FEAssembly(X_blk,hexas_blk, name= "BLOCK",recOuters=False)
blk.Youngsmodulus = 0.05
blk.Translate([0.0,0.0,3.5])


# # POTATO
[ptt] = pickle.load(open("Dat/PotatoAssembly.dat","rb"))
# Fix compatibility issue from hexas->elements rename
if hasattr(ptt, 'hexas') and not hasattr(ptt, 'elements'):
    ptt.elements = ptt.hexas
## OR ##
# mesh_ptt = meshio.read("../Meshes/QuadSpheres/QuadSphere4.msh")
# # mesh_ptt = meshio.read("Meshes/Cubes/cube843.msh")
# X_ptt = mesh_ptt.points
# hexas_ptt = mesh_ptt.cells_dict['hexahedron']
# ptt = FEAssembly(X_ptt,hexas_ptt, name = "POTATO",recOuters=recOut)
# ptt.Resize(4.0,dir='x')
# ptt.Resize(3.0,dir='y')
# ptt.Resize(2.0,dir='z')
# ptt.RandDistort(0.5)
# pickle.dump([ptt],open("Dat/PotatoAssembly.dat","wb"))
ptt.isRigid = True     # faster solving when True

######################
### BUILDING MODEL ###
######################

### Selections (nodes) ###
blk_bottom  = blk.SelectFlatSide("-z")
blk_top     = blk.SelectFlatSide("+z")
ptt_highernodes = ptt.SelectHigherThan("z", val = 0.5, Strict = True,OnSurface = True)


### BOUNDARY CONDITIONS ###  [body, nodes, type, directions, values, times(*)]
cond_bd1 = [ptt, ptt.SelectAll(), "dirichlet", "xyz"  , [0.0, 0.0, 0.0] ]
cond_bd2 = [blk, blk_top   , "dirichlet", "xyz"  , [12.0, 0.0,0.0] ]


BCs = [cond_bd1, cond_bd2]

### CONTACTS ###            # [body, nodes]
slave   = [blk , blk_bottom]
master = [ptt,ptt_highernodes]

# CONVERGENCE FIX: Mesh-adaptive contact parameters to prevent oscillations
element_size = 4.0 / mesh  # Approximate element characteristic length

# Conservative contact stiffness scaling - maintain reasonable ratio to material stiffness
E = blk.Youngsmodulus  # Material Young's modulus
base_kn = 20.0 * E  # Conservative base stiffness (20x material stiffness)

# Scale contact stiffness y with mesh characteristic parameter L/h 
# mesh_adapted_kn = 0.1*base_kn*mesh
mesh_adapted_kn = base_kn

# penetration tolerance 
maxGN = 0.0001


print(f"Mesh-adaptive contact parameters for mesh={mesh}:")
print(f"  Element size: {element_size:.3f}")
print(f"  Material E: {E:.3f}")
print(f"  Contact stiffness kn: {mesh_adapted_kn:.2f} (was 1e2={1e2:.0f})")
print(f"  Max penetration maxGN: {maxGN:.4f} (was 0.001)")

# contact1 = Contact(slave, master, kn=mesh_adapted_kn, C1Edges = False,
#                     maxGN = maxGN, f0=0.1)

contact1 = Contact(slave, master, kn=mesh_adapted_kn, C1Edges = False,
                    maxGN = maxGN, f0=0.1, use_TR_projection=True,
                    TR_init=0.1, TR_min=1e-15, TR_max=1.0,
                    tr_base_ncand=24, tr_min_ncand=10, tr_max_ncand=96,
                    tr_radius_factor_initial=1.5, tr_k_surf=15)


### MODEL ###
subname = "_"+("plastic" if plastic else "elastic")+"_"+minimization_method+"_"+str(mesh)
model = FEModel([blk, ptt], [contact1], BCs, subname=subname)           # [bodies, contacts, BCs, opts*]

ndofs = 3*(len(X_blk)+len(ptt.X))
ptt.surf.ComputeGrgPatches(np.zeros(ndofs),range(len(ptt.surf.nodes)))
model.plotNow()       # Uncomment to see and verify geometry

#############
## Running ##
#############

# import cProfile
# import pstats
# import io
# pr.enable(# pr = cProfile.Profile())

t0 = time()


# recov = "OUTPUT_202410290908ContactPotato_slideX_elastic_BFGS_10/"+"RecoveryData.dat"
model.Solve(TimeSteps=100,max_iter=20, recover=False ,minimethod=minimization_method,plot=0)



print("this took",time()-t0,"seconds to compute")


# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
# ps.print_stats()

# print(s.getvalue())


