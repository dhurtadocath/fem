from pdb import set_trace
import numpy as np
import sys, os

os.chdir(sys.path[0])

import csv
csv.field_size_limit(sys.maxsize)


data = []
with open("./results.csv", newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        data.append(row)
iters_str = data [0][1:]
U_str = data[1]
iters = []
U = []

for itstr in iters_str:
    iters.append(float(itstr))

for ustr in U_str:
    U.append(float(ustr))

U = np.array(U)

Ux,Uy,Uz = U.reshape(-1,3).T

X = range(len(iters))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

set_trace()


ax.plot(X,iters)
ax.set_yscale('log')
plt.show()

