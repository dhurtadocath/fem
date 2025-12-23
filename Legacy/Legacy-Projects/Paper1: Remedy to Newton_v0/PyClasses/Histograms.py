from pdb import set_trace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# import csv
# with open("ContactData", 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
#     csvreader = csv.reader(csvfile,delimiter=',')

data = np.loadtxt('ContactData_+X.csv', delimiter=',')
datb = np.loadtxt('ContactData_Ind+X.csv', delimiter=',')
GNs = np.concatenate((data[:,3],datb[:,3]))
GNs_pos = GNs[GNs > 0]
GNs_neg = GNs[GNs < 0]
logGNs = np.log10(np.abs(GNs))
gnmin=min(GNs)
gnmax=max(GNs)


fig = plt.figure(tight_layout=True)

# ax = fig.add_subplot(111)
# ax.hist(GNs,range=(-0.005,0.005),bins=1000)

gs = gridspec.GridSpec(3, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1:3, :])
ax1.hist(GNs_neg,range=(-0.005,0.0),bins=200)
ax2.hist(GNs_pos,range=(0.0,0.005),bins=200)
ax3.hist(logGNs,log=False,bins=1000)
ax1.set_xlabel("gn (<0)")
ax2.set_xlabel("gn (>0)")
ax3.set_xlabel("log10(|gn|)")

plt.show()

set_trace()

