import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pdb import set_trace
import meshio
from time import time


m  = meshio.read("VolumeTest/286t.msh")

X = m.points
C = m.cells_dict['triangle']


t0 = time()
vol = 0
for (i,j,k) in C:
    ap,p,bp = X[i], X[j], X[k]
    c = -np.cross(ap-p,bp-p)
    vol += np.dot(p,c)/6
print("\nVolume: ",vol," computed in ",time()-t0," seconds")

triangles = [((X[i],X[j],X[k])) for (i,j,k) in C]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection(Poly3DCollection(triangles,facecolor=(0,0,1,0.5),edgecolor="black"))
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_zlim([-4,4])


plt.show()