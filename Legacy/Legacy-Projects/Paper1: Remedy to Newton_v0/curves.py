import matplotlib.pyplot as plt
import numpy as np
import os
from pdb import set_trace
cwd = os.path.dirname(os.path.abspath(__file__)) + "/"

data = np.loadtxt(cwd+'fint.csv', delimiter=',')
tt = data[:,0]
fx = data[:,1]
fy = data[:,2]
fz = data[:,3]

fig, ax = plt.subplots(tight_layout=True, figsize=(12,8))

inters = [[0.41515625,0.4153125],
          [0.69359375,0.69375]]

ax.plot(tt,fx, linewidth=5,label=r"$\sum{F_x}$")
ax.plot(tt,fy, linewidth=5,label=r"$\sum{F_y}$")
ax.plot(tt,0.1*fz, linewidth=5,label=r"$0.1\sum{F_z}$")

ymin, ymax = ax.get_ylim()

for inter in inters:
    ax.plot([inter[0]]*2,[ymin,ymax], linewidth=2, color = "black")
    ax.plot([inter[1]]*2,[ymin,ymax], linewidth=2, color = "black")

plt.xlabel(r"time",fontsize=30)
plt.ylabel(r"force",fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=25)
plt.grid()
plt.savefig(cwd+'Forces.pdf')
plt.show()
