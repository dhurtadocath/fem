from PyClasses.FEAssembly import *
from PyClasses.Contacts import *
from PyClasses.FEModel import *

import meshio, sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
from math import pi


####################
### GETTING DATA ###
####################
os.chdir(sys.path[0])

# BLOCK
mesh_blk   = meshio.read("../Meshes/Cubes/cube5x5x5side4.msh")
# mesh_blk   = meshio.read("../Meshes/Cubes/cube3x3x3side4.msh")
X_blk     = mesh_blk.points
hexas_blk = mesh_blk.cells_dict['hexahedron']
blk = FEAssembly(X_blk,hexas_blk, name= "BLOCK",recOuters=False)
blk.Youngsmodulus = 0.05
blk.Translate([6.0,0.0,4.5])


# # POTATO
[ptt] = pickle.load(open("PotatoAssembly.dat","rb"))
## OR ##
# mesh_ptt = meshio.read("../Meshes/QuadSpheres/QuadSphere4.msh")
# # mesh_ptt = meshio.read("Meshes/Cubes/cube843.msh")
# X_ptt = mesh_ptt.points
# hexas_ptt = mesh_ptt.cells_dict['hexahedron']
# ptt = FEAssembly(X_ptt,hexas_ptt, name = "POTATO",recOuters=False)
# ptt.Resize(4.0,dir='x')
# ptt.Resize(3.0,dir='y')
# ptt.Resize(2.0,dir='z')
# ptt.RandDistort(0.5)
# # pickle.dump([ptt],open("PotatoAssembly.dat","wb"))
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
cond_bd2 = [blk, blk_top   , "dirichlet", "xyz"  , [0.0, 0.0,-1.5], [0.0,0.5] ]
cond_bd3 = [blk, blk_top   , "dirichlet", "xyz"  , [4.0, 0.0, 0.0], [0.4,1.0] ]

BCs = [cond_bd1, cond_bd2,cond_bd3]

### CONTACTS ###            # [body, nodes]
slave   = [blk , blk_bottom]
master = [ptt,ptt_highernodes]

contact1 = Contact(slave, master, kn=5, C1Edges = False, maxGN = 0.001,f0=0.1)       # (slave, master) inputs can be surfaces as well

### MODEL ###
model = FEModel([blk, ptt], [contact1], BCs)           # [bodies, contacts, BCs, opts*]

ndofs = 3*(len(X_blk)+len(ptt.X))
ptt.surf.ComputeGrgPatches(np.zeros(ndofs),range(len(ptt.surf.nodes)))
# model.plotNow()       # Uncomment to see and verify geometry

#############
## Running ##
#############
model.Solve(TimeSteps=100, recover=False, ForcedShift=False,max_iter=10)
