from PyClasses.FEAssembly import *
from PyClasses.Contacts import *
from PyClasses.FEModel import *

import meshio, sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
from math import pi
from scipy.stats import johnsonsu, lognorm, beta, gamma, norm


#####################
### Setting model ###
#####################
os.chdir(sys.path[0])

# POTATO
[ptt] = pickle.load(open("PotatoAssembly.dat","rb"))
ptt.isRigid = True     # faster solving when True
ndofs = 3*len(ptt.X)

# MODEL
model = FEModel([ptt], [],[])           # [bodies, contacts, BCs, opts*]
ptt.Translate([-6.0, 0.0, 0.0])
model.X = ptt.X.ravel()
ptt.surf.ComputeGrgPatches(np.zeros(ndofs),range(len(ptt.surf.nodes)))
# model.plotNow()       # Uncomment to see and verify geometry


##################################
# Not doing anything with this yet
##################################
# TODO: capture distribution (type and corresponding parameters to be used in next section)
data = np.loadtxt('ContactData', delimiter=',')
GNs = data[:,3]
GNs_pos = GNs[GNs > 0]
GNs_neg = GNs[GNs < 0]
logGNs = np.log10(np.abs(GNs))
gnmin=min(GNs)
gnmax=max(GNs)

params = johnsonsu.fit(GNs)


#########################
### Points generation ###
#########################
n_points = int(1e5)

# gns = johnsonsu.rvs(params[0], params[1], params[2], params[3], size=n_points)
gns = norm.rvs(0,1e-3,size=n_points)
# set_trace()
gns[::2] = johnsonsu.rvs(params[0], params[1], params[2], params[3], size=int(n_points/2))
# gns = johnsonsu.rvs(params[0], params[1], size=int(n_points/2))


with open("Points.csv", 'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    points = np.zeros((n_points,3))
    for i in range(n_points):
        # choose patch
        # ipatch = int(np.random.rand()*len(ptt.surf.patches))
        ipatch = 50
        patch = ptt.surf.patches[ipatch]
        # choose surface coordinate within patch
        xi = np.array([np.random.uniform(0.,1.),np.random.uniform(0,1.)])
        # choose a normal gap for the point on the surface
        # gn = np.random.normal(loc=0,scale=1e-3)/2.
        gn = gns[i]

        results = patch.get_Dgn_for_rigids(gn,xi)
        # compute the randomly produced point
        if i%1000==0:
            print(i)
        points[i]=results[:3]
        csvwriter.writerow(results)

# fig = plt.figure(tight_layout=True)
# ax = fig.add_subplot(111)
# ax.hist(logGNs,bins=1000,color='b')
# # ax.hist(gns,range=(-0.005,0.005),bins=200,color='r')
# plt.show()


########################################
### Plot of results (Might take forever)
########################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
model.plot(ax)
ax.scatter(points[:,0],points[:,1],points[:,2],s=0.1,c='k')
plt.show()