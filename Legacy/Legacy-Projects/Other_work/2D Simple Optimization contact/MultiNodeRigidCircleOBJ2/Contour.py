import pickle
from matplotlib import pyplot as plt
from matplotlib import cm
from math import log10
import numpy as np
from pdb import set_trace

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)


sx,sy,mz,cors = pickle.load(open("/home/diego/fem/2D Simple Optimization contact/MultiNodeRigidCircleOBJ2/ObjectiveAt71SLSQP-con.dat","rb"))
X,Y = np.meshgrid(sx, sy)
Z = 10**mz - 1
ax.contour(X.T,Y.T,mz,50, alpha=0.25)


sx,sy,mz,cors = pickle.load(open("/home/diego/fem/2D Simple Optimization contact/MultiNodeRigidCircleOBJ2/ObjectiveAt71SLSQP-c2.dat","rb"))
X,Y = np.meshgrid(sx, sy)
Z = 10**mz - 1
ax.contour(X.T,Y.T,mz,50, alpha=0.25)

plt.show()


set_trace()


# 3D-Plot of the surface.
fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax1.plot_surface(X,Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
mymap = cm.ScalarMappable()
mymap.to_rgba(cors)
surf = ax2.plot_surface(X,Y, mz, cmap=mymap,
                       linewidth=0, antialiased=False)
# surf = ax2.plot_surface(X,Y, mz, cmap=cors,)
# fig1.colorbar(surf, shrink=0.5, aspect=5)
# fig2.colorbar(surf, shrink=0.5, aspect=5)
plt.show()