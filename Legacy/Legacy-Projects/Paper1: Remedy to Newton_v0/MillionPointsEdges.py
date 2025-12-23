from PyClasses.FEAssembly import *
from PyClasses.Contacts import *
from PyClasses.FEModel import *

import argparse
import sys, os, pickle
import numpy as np
from pdb import set_trace
import time
import random


#####################
### Setting model ###
#####################
os.chdir(sys.path[0])

cwd = os.path.dirname(os.path.abspath(__file__))


#######################
# LOADING FE_ASSEMBLY #
#######################
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


################
# LOADING DATA #
################
# Assuming 'cwd' is correctly defined somewhere above this snippet
folderPath = os.path.join(cwd, 'csv_files')

# Step 1: Determine the total number of rows across all CSV files
total_rows = 0
for filename in os.listdir(folderPath):
    if filename.endswith('.csv'):  # Ensuring we only process CSV files
        with open(os.path.join(folderPath, filename), 'r') as file:
            total_rows += sum(1 for _ in file)  # Adjust if your CSVs have headers

# Step 2: Initialize a numpy array with shape (total_rows, 7)
data = np.zeros((total_rows, 7))

# Step 3: Read each CSV file and append its contents to the numpy array
current_index = 0
for filename in os.listdir(folderPath):
    if filename.endswith('.csv'):  # Ensuring we only process CSV files
        file_path = os.path.join(folderPath, filename)
        # Assuming the CSV files do not have headers; if they do, use skiprows=1
        file_data = np.loadtxt(file_path, delimiter=',')
        n_rows_i = file_data.shape[0]
        data[current_index:current_index+n_rows_i, :] = file_data
        current_index += n_rows_i

# filter wrong rows (t1 and/or t2 or out the [0,1] range)
data = data[(data[:,4]>=0)&(data[:,4]<=1)&(data[:,5]>=0)&(data[:,5]<=1)]




# # Argument parsing
# parser = argparse.ArgumentParser(description='User provides a number between 0 and 7. Each group contains 12 patches.')
# parser.add_argument('--patch_group', type=int, required=True, help='Group number')
# args = parser.parse_args()
# igp = args.patch_group

igp = 2


patches = model.bodies[0].surf.patches
npatches = len(patches)

xm_matrix = np.zeros((npatches,3))
for ip,patch in enumerate(patches):
    xm_matrix[ip] = patch.BS.x


def verify(xsi,patch_verify):
    rec_lvl = 2 # seeding for recursive MinDist search (u_i = np.linspace(x0,x1,seeding+1))
    # get candidates of closest patch
    raw_distances = norm(xm_matrix-xsi,axis=1)

    mindist = min(raw_distances)
    maxdist = max(raw_distances)
    size_factor = 0.03      # offset percentage for search radius where search_radius = mindist + size_factor*(maxdist-mindist)
    candidates = [patch for patch, dist in zip(patches, raw_distances) if dist < mindist+size_factor*(maxdist-mindist)]

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
            set_trace()
            pass

    return patch_verify==pid
 


patches_here = patches[12*igp:12*(igp+1)]
for patch in patches_here:


    # with open("Points_Edges.csv", 'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
    with open("Points_Edges_"+str(igp)+".csv", 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
    # with open("Points_chacking.csv", 'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)



        t0 = time.time()
        # here each edge is numerated ccw starting from the edge where t2==0
        # points (rows) in current patch
        pp = data[data[:,3]==patch.iquad]
        pp_neg = pp[pp[:,-1]< 0]
        pp_pos = pp[pp[:,-1]>=0]

        border = 5e-2
        pp_in =[pp_neg[pp_neg[:,5]<border],
                pp_neg[pp_neg[:,4]>1-border],
                pp_neg[pp_neg[:,5]>1-border],
                pp_neg[pp_neg[:,4]<border]]
        pp_out=[pp_pos[pp_pos[:,5]<border],
                pp_pos[pp_pos[:,4]>1-border],
                pp_pos[pp_pos[:,5]>1-border],
                pp_pos[pp_pos[:,4]<border]]




        # array to store groups of min/max gn's per edge
        n_intervals_per_edge = int(1/border)
        gm = np.zeros((n_intervals_per_edge,2))   # saves min/max for each interval

        for iedge in range(4):
            # limits inside of the body
            ppie = pp_in[ iedge] [ : , [4+(iedge%2),-1 ] ]  # shape (n_points_edge,2) cols: t_i,gn
            ppoe = pp_out[iedge] [ : , [4+(iedge%2),-1 ] ]

            # divide each edge into intervals between 0 and 1, each of width 'border'
            for i in range(n_intervals_per_edge):
                try:
                    gm[i,0] = min(ppie[(ppie[:,0]>i*border)&(ppie[:,0]<(i+1)*border),-1])
                except:
                    gm[i,0] = -border
                    # print("error at i:", i," (patch ", patch.iquad,", edge",iedge,") For points IN")
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(pp_in[ iedge][:,0],pp_in[ iedge][:,1],pp_in[ iedge][:,2],color=(0,1,0,1),s=0.2)
                    # patch.plot(ax,color=(0,0,1,1))
                    # fig2 = plt.figure()
                    # ax2 = fig2.add_subplot(111)
                    # ax2.scatter(ppie[:,0],ppie[:,1],color=(0,0,1,1))
                    # ax2.scatter([i*border, (i+1)*border],[0,0],color=(0,0,0,1))
                    # plt.show()
                
                try:
                    gm[i,1] = max(ppoe[(ppoe[:,0]>i*border)&(ppoe[:,0]<(i+1)*border),-1])
                except:
                    gm[i,1] = border


            # fig = plt.figure()
            # ax = fig.add_subplot(121, projection='3d')
            # ax.scatter(pp_out[ iedge][:,0],pp_out[ iedge][:,1],pp_out[ iedge][:,2],color=(1,0,0,1),s=0.2)
            # ax.scatter(pp_in[ iedge][:,0],pp_in[ iedge][:,1],pp_in[ iedge][:,2],color=(0,1,0,1),s=0.2)
            # patch.plot(ax,color=(0,0,1,1))
            # ax2 = fig.add_subplot(122)
            # ax2.scatter(ppoe[:,0],ppoe[:,1],color=(1,0,0,1),s=0.3)
            # ax2.scatter(ppie[:,0],ppie[:,1],color=(0,1,0,1),s=0.3)
            # plt.show()





            # generate 2500 random edge values between  0 and 1
            # ts_long = np.random.rand(2500)        # ts_long an be computed in the try loop "while i<2500:"
            ts_short= abs(np.random.normal(0,5e-2,2500))
            if iedge in [1,2]:
                ts_short = 1-ts_short

            # for i in range(2500):
            i = 0
            wrongProjs = 0
            while i <2500:
                ts_long = random.random()       # we compute it here to avoid insisting on unreachable parts of the edge
                interval = int(ts_long/border)
                gn = np.random.uniform(gm[interval,0],gm[interval,1])
                t = [ts_long,ts_short[i]] if iedge%2==0 else [ts_short[i],ts_long]
                xsi = patch.Grg0(t) + gn*patch.D3Grg(t)
                # verify if point actually as this patch as current closest projecting object
                is_patch_correct = verify(xsi,patch.iquad)            
                if is_patch_correct:
                    row = [xsi[0],xsi[1],xsi[2],patch.iquad,t[0],t[1],gn]
                    csvwriter.writerow(row)
                    i+=1
                else:
                    wrongProjs += 1
            print("incorrect projections for patch",patch.iquad,",edge",iedge,":",wrongProjs,".")


        # ax.scatter(gen_data_patch[:,0],gen_data_patch[:,1],gen_data_patch[:,2],color=(0,1,0,1),s=0.2)
        # ax.scatter(gen_data_patch[:,0],gen_data_patch[:,1],gen_data_patch[:,2],color=(1,0,0,1),s=0.2)


        # plt.show()



        print("time taken by patch",patch.iquad,":", time.time()-t0,"[s]")

