from PyClasses.FEAssembly import *
from PyClasses.Contacts import *
from PyClasses.FEModel import *

import argparse
import sys, os, pickle
import numpy as np
from pdb import set_trace
from time import time


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


# Argument parsing
parser = argparse.ArgumentParser(description='Process a specific subspace of points.')
parser.add_argument('--points_per_side', type=int, required=True, help='Total number of points per side')
parser.add_argument('--chunks_per_side', type=int, required=True, help='Number of partitions per side')
parser.add_argument('--i', type=int, required=True, help='Partition index along x')
parser.add_argument('--j', type=int, required=True, help='Partition index along y')
parser.add_argument('--k', type=int, required=True, help='Partition index along z')

args = parser.parse_args()

# Calculate subspace bounds
points_per_side = args.points_per_side
chunks_per_side = args.chunks_per_side
i, j, k = args.i, args.j, args.k
chunksize = int(points_per_side/chunks_per_side)+1

if i == chunks_per_side - 1:
    chunksizeX = points_per_side - i*chunksize
else:
    chunksizeX = chunksize
if j == chunks_per_side - 1:
    chunksizeY = points_per_side - j*chunksize
else:
    chunksizeY = chunksize
if k == chunks_per_side - 1:
    chunksizeZ = points_per_side - k*chunksize
else:
    chunksizeZ = chunksize


offset = 0.1
# n_per_side = 100 


model_X = model.X.reshape(-1,3)
xa,xb = min(model_X[:,0]), max(model_X[:,0])
ya,yb = min(model_X[:,1]), max(model_X[:,1])
za,zb = min(model_X[:,2]), max(model_X[:,2])

dx, dy, dz = offset*np.array([xb-xa,yb-ya,zb-za])
xmin, xmax = xa-dx, xb+dx
ymin, ymax = ya-dy, yb+dy
zmin, zmax = za-dz, zb+dz

xs = np.zeros((chunksizeX*chunksizeY*chunksizeZ , 3))

# Generate set of points in space
locator_index = 0
for x in np.linspace(xmin,xmax, points_per_side)[chunksize*i:chunksize*i+chunksizeX]:
    for y in np.linspace(ymin,ymax, points_per_side)[chunksize*j:chunksize*j+chunksizeY]:
        for z in np.linspace(zmin,zmax, points_per_side)[chunksize*k:chunksize*k+chunksizeZ]:
            xs[locator_index] = np.array([x,y,z])
            locator_index += 1

# import cProfile
# import pstats
# import io

# pr = cProfile.Profile()
# pr.enable()


# xs = np.array([[-2.7660309231561615,-1.0611164444611427,-0.8755869161071423],
#                [-2.7660309231561615,-1.0611164444611427,-0.8230838371203275],
#                [-2.7660309231561615,-1.0611164444611427,-0.7705807581335125],
#                [-2.7660309231561615,-1.0611164444611427,-0.7180776791466976]])


# xsi = xs[7705]
# patch = model.bodies[0].surf.patches[75]
# t1t2 = patch.findProjection(xsi,recursive=3)


t0 = time()

with open("Points_"+str(i)+'_'+str(j)+'_'+str(k)+".csv", 'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
# with open("Points_chacking.csv", 'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)


    # Iterate over points generated to get closest patch and surface parameters
    patches = model.bodies[0].surf.patches
    npatches = len(patches)
    raw_distances = np.zeros(npatches)


    xm_matrix = np.zeros((npatches,3))
    for ip,patch in enumerate(patches):
        xm_matrix[ip] = patch.BS.x
    distance_matrix = np.linalg.norm(xm_matrix[:, np.newaxis, :] - xs, axis=2)

    # set_trace()

    for i,xsi in enumerate(xs):
        for ip,patch in enumerate(patches):
            raw_distances[ip] = distance_matrix[ip,i]


        rec_lvl = 2 # seeding for recursive MinDist search (u_i = np.linspace(x0,x1,seeding+1))
        # get candidates of closest patch
        mindist = min(raw_distances)
        maxdist = max(raw_distances)
        patch_size = 2.0        # approximate average patch size (observed from BS radius ~1)
        size_factor = 0.03      # offset percentage for search radius where search_radius = mindist + size_factor*(maxdist-mindist)
        candidates = [patch for patch, dist in zip(patches, raw_distances) if dist < mindist+size_factor*(maxdist-mindist)]

        if i%100==0:
            print("pt_nr:",i,"\txsi=",xsi, "\ttotal_time:",time()-t0)  # to visualize evolution of computation

        t1t2 = np.zeros((len(candidates),2))
        gns = np.zeros(len(candidates))         # global normal vectors
        for ip,patch in enumerate(candidates):
            t1t2[ip] = patch.findProjection(xsi,recursive=rec_lvl)
            xc = patch.Grg0(t1t2[ip])
            nor = patch.D3Grg(t1t2[ip])
            gns[ip] = (xsi-xc)@nor

        cand_idx =  np.argmin(abs(gns))                   # index of the candidate with smallest angle between x
        patch = candidates[cand_idx]
        pid = patch.iquad                  # index of closest patch
        t1,t2 = t1t2[cand_idx]

        # Unprojected Case!
        trials = 0
        while not((0<=t1<=1) and (0<=t2<=1)):               # loop until a valid projection is found
            trials += 1
            size_factor *= 1.2      # offset percentage for search radius where search_radius = mindist + size_factor*(maxdist-mindist)
            candidates = [patch for patch, dist in zip(patches, raw_distances) if dist < mindist+size_factor*(maxdist-mindist)]
            t1t2 = np.zeros((len(candidates),2))
            gns = np.zeros(len(candidates))         # global normal vectors
            rec_lvl += 1
            for ip,patch in enumerate(candidates):
                t1t2[ip] = patch.findProjection(xsi,recursive=rec_lvl)
                xc = patch.Grg0(t1t2[ip])
                nor = patch.D3Grg(t1t2[ip])
                inside = ((0<=t1t2[ip][0]<=1) and (0<=t1t2[ip][1]<=1))
                gns[ip] = (xsi-xc)@nor if inside else 1000

            cand_idx =  np.argmin(abs(gns))                   # index of the candidate with smallest angle between x
            patch = candidates[cand_idx]
            pid = patch.iquad                  # index of closest patch
            t1,t2 = t1t2[cand_idx]

            if trials>10:
                print("more than 10 trials while looking for projections for point",xsi,".")
                print("last candidates: \n",[cand.iquad for cand in candidates])
                pass

        csvwriter.writerow([xsi[0],xsi[1],xsi[2],pid,t1,t2,gns[cand_idx]])

# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
# ps.print_stats()

# print(s.getvalue())



"""


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

t
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
plt.show()"""