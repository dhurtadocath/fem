from pdb import set_trace
from pandas import read_csv
import os, sys
os.chdir(sys.path[0])
import matplotlib.pyplot as plt
import re


CASES = ["d_","f_","df"]
FOLS = [ f.path for f in os.scandir('.') if f.is_dir() ]
# METHODS = ["SLSQP","BFGS","CG","L-BFGS-B","TNC"]
METHODS = []
IGNORE = ['CG','TNC']
for fol in FOLS:
    if '_' in fol:
        newmeth = fol.split('_')[1]
        if newmeth not in METHODS:
            METHODS.append(newmeth)

# plt.figure(figsize=(2000,1000))
for case in CASES:
    # new style method 1; unpack the axes
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(16,9)
    NFEVS= []
    MOBJ = []
    DMDU = []
    EINT = []
    for meth in METHODS:
        for fol in FOLS:
            if case==fol[2:4] and ("_"+meth) in fol and meth not in IGNORE:
                data = read_csv(fol+"/OUT_verbose.csv")
                ti = list(data['time'])+[1.0]
                nf = [0]+list(data['nfev'])
                mo = [0]+list(data['m'])
                dm = [0]+list(data['|dmdu|'])
                ei = [0]+list(data['E_int'])
                ax1.plot(ti,nf,label=meth)
                # ax1.set_yscale('log')
                ax2.plot(ti,mo,label=meth)
                ax3.plot(ti,dm,label=meth)
                ax3.set_yscale('log')
                ax4.plot(ti,ei,label=meth)
                if case=="d_":
                    ax2.set_yscale('log')
                    ax4.set_yscale('log')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax1.set_title("nfev")
    ax2.set_title("m_obj")
    ax3.set_title("dmdu")
    ax4.set_title("E_int")
    if case=="d_":
        fig.suptitle("ONLY Displacement",size=40)
    elif case=="f_":
        fig.suptitle("ONLY Force",size=40)
    else:
        fig.suptitle("Force + Displacement",size=40)
    fig.savefig(case+"_curves.png")
    plt.close(fig)
