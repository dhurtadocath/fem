import pickle
from matplotlib import pyplot as plt
from matplotlib import cm
from math import log10
import numpy as np
from pdb import set_trace
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sx,sy,mz,cors = pickle.load(open("/home/diego/fem/2D Simple Optimization contact/MultiNodeRigidCircle/ObjectiveAt18xyz.dat","rb"))
X,Y = np.meshgrid(sx, sy)
Z = mz
n = X.shape[0]
cors = cors.reshape(n**2,4)
# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Flatten the X, Y, and Z arrays
x = X.flatten()
y = Y.flatten()
z = Z.flatten()

# set_trace()
# Create a Poly3DCollection with the X, Y, and Z coordinates and the cors
idxs = [i for i in range(n**2) if ((i+1)%n!=0 and i<n*(n-1))]
poly3d = [[(x[i], y[i], z[i]), (x[i+1], y[i+1], z[i+1]), (x[i+n+1], y[i+n+1], z[i+n+1]), (x[i+n], y[i+n], z[i+n])] for i in idxs]
colors = [ (cors[i]+cors[i+1]+cors[i+n+1]+cors[i+n])/4 for i in idxs]
collec = Poly3DCollection(poly3d, facecolors=colors, edgecolors=(0,0,0,0.2))
# collec = Poly3DCollection(poly3d, facecolors=colors)

# Add the Poly3DCollection to the axis
ax.add_collection(collec)

# Set the limits of the axis
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(z), np.max(z))

# Set the labels of the axis
ax.set_xlabel('S1')
ax.set_ylabel('S2')
ax.set_zlabel('mOBJ')

# Show the plot
plt.show()