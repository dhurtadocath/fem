from pdb import set_trace
# from re import X
import numpy as np
from numpy.linalg import norm
from math import sqrt, sin, cos, atan,atan2, acos, pi, log10, exp
from scipy.optimize import minimize,LinearConstraint, newton, Bounds, HessianUpdateStrategy,approx_fprime
import os, sys, pickle, csv
from functools import lru_cache
from time import strftime
import argparse

os.chdir(sys.path[0])

os.chdir(sys.path[0])

def Truss2D(nx,ny,dx,dy,diags = True):
    C = []
    X = np.empty(((nx+1)*(ny+1),2),dtype=np.double)

    X[(nx+1)*(ny+1)-1]= np.array([nx*dx,ny*dy])
    for i in range(ny):
        for j in range(nx):
            a = (nx+1)*i+j
            b = (nx+1)*i+j+1
            c = (nx+1)*(i+1)+j
            d = (nx+1)*(i+1)+j+1

            if i==0:
                C.append([a,b])
            if j==0:
                C.append([a,c])
            if diags:
                C.extend([[a,d],[b,c],[b,d],[c,d]])
            else:
                C.extend([[b,d],[c,d]])
            X[a] = np.array([(j*dx,i*dy)])
            if j==nx-1:
                X[b] = np.array([((j+1)*dx,i*dy)])
            if i==ny-1:
                X[c] = np.array([(j*dx,(i+1)*dy)])

    return X, C

def Truss(nx,ny,nz,dx,dy,dz):
    C = []
    X = np.empty(((nx+1)*(ny+1)*(nz+1),3),dtype=np.double)

    bar_types = []
    X[0]= np.array([0.0,0.0,0.0])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                a = (nx+1)*(ny+1)*(k)   + (nx+1)*(j)   + (i)
                b = (nx+1)*(ny+1)*(k)   + (nx+1)*(j)   + (i) + 1
                c = (nx+1)*(ny+1)*(k)   + (nx+1)*(j+1) + (i) + 1
                d = (nx+1)*(ny+1)*(k)   + (nx+1)*(j+1) + (i)
                e = (nx+1)*(ny+1)*(k+1) + (nx+1)*(j)   + (i)
                f = (nx+1)*(ny+1)*(k+1) + (nx+1)*(j)   + (i) + 1
                g = (nx+1)*(ny+1)*(k+1) + (nx+1)*(j+1) + (i) + 1
                h = (nx+1)*(ny+1)*(k+1) + (nx+1)*(j+1) + (i)

                tidx = (i+j+k)%2    # diag type index   0: a,c,f,h      1: b,d,e,g
                if i==0:
                    C.extend([[e,h],[d,h]])
                    C.append([a,h] if tidx== 0 else [d,e])
                    bar_types.extend([0,0,1])
                    X[h] = np.array([0.0,(j+1)*dy,(k+1)*dz])
                if j==0:
                    C.extend([[e,f],[b,f]])
                    C.append([a,f] if tidx== 0 else [b,e])
                    bar_types.extend([0,0,1])
                    X[f] = np.array([(i+1)*dx,0.0,(k+1)*dz])
                if k==0:
                    C.extend([[b,c],[c,d]])
                    C.append([a,c] if tidx== 0 else [b,d])
                    bar_types.extend([0,0,1])
                    X[c] = np.array([(i+1)*dx,(j+1)*dy,0.0])

                if i+j==0:
                    C.append([a,e])
                    bar_types.append(0)
                    X[e] = np.array([0.0,0.0,(k+1)*dz])
                if i+k==0:
                    C.append([a,d])
                    bar_types.append(0)
                    X[d] = np.array([0.0,(j+1)*dy,0.0])
                if j+k==0:
                    C.append([a,b])
                    bar_types.append(0)
                    X[b] = np.array([(i+1)*dx,0.0,0.0])

                diags = [[[c,f],[c,h],[f,h]],[[b,g],[d,g],[e,g]]][tidx]
                C.extend([[g,h],[f,g],[c,g]])
                C.extend(diags)
                bar_types.extend([0,0,0,1,1,1])
                
                X[g] = np.array([((i+1)*dx,(j+1)*dy,(k+1)*dz)])

    return X, C, bar_types

def plotTruss(X,C, show_id=False,bar_types=None,showtypes=-1, ax = None):
    show = False
    if ax is None:        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        show=True

    if bar_types is None:
        bar_types = np.zeros(len(C))

    for i,(a,b) in enumerate(C):
        if bar_types[i]!=showtypes and showtypes!=-1: continue

        XX = X[np.ix_([a,b])]
        ax.plot(XX[:,0],XX[:,1],XX[:,2], c="black" if compr[i] else "red",linewidth=0.5)
        if show_id: 
            alp=  0.3
            avg = XX[0]*(1-alp) + XX[1]*alp
            ax.text(avg[0],avg[1],avg[2],i,c="blue")
    # for i, x in enumerate(X):
    #     color = "red"
    #     ax.scatter(x[0],x[1],x[2],c=color,s=5)
    #     if show_id: ax.text(x[0],x[1],x[2],i)


    if show:  plt.show()

def plotMS(u,X,C, ax=None, undef=False, show=False, save=None,sline = None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(221,projection='3d')
        ax2= fig.add_subplot(222,projection='3d')
        ax3= fig.add_subplot(223,projection='3d')
        ax4= fig.add_subplot(224)

    plotTruss(X+u,C,ax=ax,bar_types=btypes, show_id=False,showtypes=0)
    plotTruss(X+u,C,ax=ax2,bar_types=btypes,show_id=False,showtypes=0)
    plotTruss(X+u,C,ax=ax3,bar_types=btypes,show_id=False,showtypes=0)

    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_zlim([-1.0,1.0])
    ax2.set_xlim([-1.5,1.5])
    ax2.set_ylim([-1.5,1.5])
    ax2.set_zlim([-1.0,1.0])
    ax2.set_proj_type('ortho')
    ax2.view_init(elev=90, azim=-90)
    ax2.zaxis.set_ticklabels([])      #no numbers on axis
    ax3.set_xlim([-1.5,1.5])
    ax3.set_ylim([-1.5,1.5])
    ax3.set_zlim([-1.0,1.0])
    ax3.set_proj_type('ortho')
    ax3.view_init(elev=0, azim=-90)
    ax3.yaxis.set_ticklabels([])      #no numbers on axis

    ax4.plot(TT,Eint,c='red',label='E_int')
    ax4.plot(TT,M,c='orange',label='m_obj')
    ax4.legend()
    # ax4.scatter(TT[-1],jacs[-1])
    ax4.set_xlim([0,1])
    # ax4.set_ylim([-0.001,0.00015])

    if show: plt.show()

    if save is not None:
        if not os.path.exists(folder+"/plots"):
            os.mkdir(folder+"/plots")
        fig.tight_layout()
        plt.savefig(folder+"/plots//img"+str(tt)+".png",dpi=200,bbox_inches='tight')
        plt.close(fig)


def Translate(X,disp):
    disp = np.array(disp)
    for i in range(len(X)):
        X[i] += disp
    return X

def SelectNodesByBox(X,xyz_a,xyz_b):
    nodes = []
    xa, ya, za = xyz_a
    xb, yb, zb = xyz_b
    for node_id, [x,y,z] in enumerate(X):
        if xa<=x<=xb:
            if ya<=y<=yb:
                if za<=z<=zb:
                    nodes.append(node_id)
    return nodes

def SelectFlatSide(X,side, tol = 1e-6):
    minX = min(X[:,0])
    maxX = max(X[:,0])
    minY = min(X[:,1])
    maxY = max(X[:,1])
    minZ = min(X[:,2])
    maxZ = max(X[:,2])
    eps = tol

    if "x" in side:
        if "-" in side:
            return SelectNodesByBox(X,[minX-eps,minY-eps,minZ-eps],[minX+eps,maxY+eps,maxZ+eps])
        else:
            return SelectNodesByBox(X,[maxX-eps,minY-eps,minZ-eps],[maxX+eps,maxY+eps,maxZ+eps])
    elif "y" in side:
        if "-" in side:
            return SelectNodesByBox(X,[minX-eps,minY-eps,minZ-eps],[maxX+eps,minY+eps,maxZ+eps])
        else:
            return SelectNodesByBox(X,[minX-eps,maxY-eps,minZ-eps],[maxX+eps,maxY+eps,maxZ+eps])
    else:
        if "-" in side:
            return SelectNodesByBox(X,[minX-eps,minY-eps,minZ-eps],[maxX+eps,maxY+eps,minZ+eps])
        else:
            return SelectNodesByBox(X,[minX-eps,minY-eps,maxZ-eps],[maxX+eps,maxY+eps,maxZ+eps])


def Ei(u,X,C,EA,dofs):
    
    x=X+u.reshape(-1,3)
    ms=0
    for id_bar,(i,j) in enumerate(C):
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        dl = L-L0
        compr[id_bar] = dl/L0<=-0.01
        dL = max(dl,0.0)
        ms+=(EA*dL**2)/(2*L0)
    return ms

def fi(u,X,C,EA,dofs):
    x=X+u.reshape(-1,3)
    dms=np.zeros(ndofs)
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        dL = max(L-L0,0.0)
        dms[np.ix_(dofij)]+=EA*dL/L0*dLdu
    dms[di] = 0.0
    return dms

def Ki(u,X,C,EA,dofs):
    x=X+u.reshape(-1,3)
    ddms=np.zeros((ndofs,ndofs))
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dL = max(L-L0,0.0)
        dLdu = a/L
        dadu = np.array([[1.0,0.0,0.0,-1.0,0.0,0.0],[0.0,1.0,0.0,0.0,-1.0,0.0],[0.0,0.0,1.0,0.0,0.0,-1.0],[-1.0,0.0,0.0,1.0,0.0,0.0],[0.0,-1.0,0.0,0.0,1.0,0.0],[0.0,0.0,-1.0,0.0,0.0,1.0]])
        d2Ldu2 = dadu/L - np.outer(a,a)/(L**3)
        ddms[np.ix_(dofij,dofij)] += (np.outer(dLdu,dLdu)+dL*d2Ldu2)/L0
    return EA*ddms


def m(u,X,C,EA,dofs):
    return  Ei(u,X,C,EA,dofs) - fext@u

def dm(u,X,C,EA,dofs):
    dm = np.zeros_like(fext)
    dm[fr] = (fi(u,X,C,EA,dofs) - fext)[fr]
    return dm

def ddm(u,X,C,EA,dofs):
    return Ki(u,X,C,EA,dofs)


def df2ds(u,X,C,EA,dofs):
    x=X+u.reshape(-1,3)
    dms=np.zeros(ndofs)
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        dL = max(L-L0,0.0)
        dms[np.ix_(dofij)]+=EA*dL/L0*dLdu
    dms[di] = 0.0
    # print("dms:",dms)
    # return kin*ms(u,X,C,EA,dofs)*dms
    return dms@dms

def d2f2d2s(u,X,C,EA,dofs):
    x=X+u.reshape(-1,3)
    dms=np.zeros(ndofs)
    ddms=np.zeros((ndofs,ndofs))
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        dadu = np.array([[1.0,0.0,0.0,-1.0,0.0,0.0],[0.0,1.0,0.0,0.0,-1.0,0.0],[0.0,0.0,1.0,0.0,0.0,-1.0],[-1.0,0.0,0.0,1.0,0.0,0.0],[0.0,-1.0,0.0,0.0,1.0,0.0],[0.0,0.0,-1.0,0.0,0.0,1.0]])
        d2Ldu2 = dadu/L - np.outer(a,a)/(L**3)
        # dL = max(L-L0,0.0)
        dL = L-L0
        dms[np.ix_(dofij)]+=EA*dL/L0*dLdu
        ddms[np.ix_(dofij,dofij)] += EA*(np.outer(dLdu,dLdu)+(L-L0)*d2Ldu2)/L0
    # print("ddms:",EA*dms)
    return 2*dms@ddms


def ApplyBCs(tpre,tnow, BCs,ndofs):
    """BCs = [[ dof_list , "dir/neu", val, t0(opt),tf(opt) ],[...],...]"""
    diri = []
    du = np.zeros(ndofs)
    df = np.zeros(ndofs)
    
    for BC in BCs:
        if len(BC)==3: BC.extend([0,1])
        dof_list, typ, val, t0,tf = BC
        if t0 <= tnow <= tf or t0 <= tpre <= tf:
            # load factor
            if tpre < t0:          # just entered in time interval
                LF = (tnow - t0) / (tf - t0)
            elif tnow > tf:            # just exited time interval
                LF = (tf - tpre) / (tf - t0)
            else:                   # going through time interval (normal case)
                LF = (tnow - tpre ) / (tf - t0)

            if typ == "dir":          # Dirichlet
                diri.extend(dof_list)
                if LF!=0:
                    for dof in dof_list:
                        du[dof] += val*LF
            elif LF!=0:                           # Neumann
                for dof in dof_list:
                    df[dof] += val*LF

    return du,df,diri

def savedata(time, dofs = "all", name = "OUT", Sum=False, things = None):

    if things is None:
        things = ['u','fint']

    for thing in things:
        filename = folder+"/"+name+"_"+thing+".csv"
        # writing to csv file
        with open(filename, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
            csvwriter = csv.writer(csvfile)
            if type(dofs)==list or type(dofs)==np.ndarray:
                if Sum:
                    csvwriter.writerow([time,sum(eval(thing)[dofs])])
                else:
                    csvwriter.writerow([time]+(eval(thing)[dofs]).tolist())
            elif type(dofs)==str:
                if dofs == "all": csvwriter.writerow([time]+eval(thing).tolist())
                else: print("Unrecognized datatype to be stored from string. Not saving this data"), set_trace()
            else:
                print("Unrecognized datatype to be stored. Not saving this data"), set_trace()

    filename2 = folder+"/"+name+"_verbose.csv"

    if not os.path.exists(filename2):
        csvwriter = csv.writer(open(filename2, 'a'))
        csvwriter.writerow(['time','nfev','m','|dmdu|','E_int'])

    with open(filename2, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
        csvwriter = csv.writer(csvfile)
        row = [time,MinPro.nfev,MinPro.fun,jacs[-1],Eint[-1]]
        csvwriter.writerow(row)


X = np.zeros((247,3))
X[:,:2], C = Truss2D(12,18,0.25,0.25,diags=False)
# btypes = np.zeros_like(C)
btypes = None
dim = len(X[0])
dofs = np.array(range(dim*len(X))).reshape(-1,dim)
ndofs = len(dofs.ravel())
X = Translate(X,[-3.0,-2.25, 0.0])
EA = 1.0

kin = 2
compr = [0.0]*len(C)

base = [0,12,234,246]
dofs_base_x = dofs[base][:,0]
dofs_base_y = dofs[base][:,1]
dofs_base_z = dofs[base][:,2]
edge_x_plus = dofs[SelectFlatSide(X,"+x")][:,0]
edge_z_plus = dofs[SelectFlatSide(X,"+x")][:,2]
mu = 0.0

u = np.zeros(ndofs)
idx_act = []

# INPUTS
parser = argparse.ArgumentParser(description="the method that will be used in the minimization")
parser.add_argument('method',type=str,help="the method that will be used in the minimization")
parser.add_argument('f_or_d',type=str,help="f:force,d:displacement")
args = parser.parse_args()
meth = args.method
bc = args.f_or_d
# meth= "asd"
# bc = 'd'

#Step1  (Right)
if 'd' in bc: 
    bc1 = [dofs_base_x,"dir", 1.0, 0.0,1.0]
else:
    bc1 = [dofs_base_x,"dir", 0.0, 0.0,1.0]
bc2 = [dofs_base_y,"dir", 0.0, 0.0,1.0]
bc3 = [dofs_base_z,"dir", 0.0, 0.0,1.0]
if 'f' in bc:
    bc4 = [edge_x_plus,"neu",-0.03, 0.0,1.0]
    bc5 = [edge_z_plus,"neu",-0.03, 0.0,1.0]
    BCs = [bc1,bc2,bc3,bc4,bc5]
else:
    BCs = [bc1,bc2,bc3]

kn = kt = 1e3
tol = 1e-8

TT = []

folder = bc+"_"+meth+"_"+strftime("%y%m%d-%H%M")
if not os.path.exists(folder):
    os.mkdir(folder)
idx_act = []
u = np.zeros(ndofs)
fint = np.zeros(ndofs)
fext = np.zeros(ndofs)

t,tt=0.0, 0
dt=0.02

nevals = []
jacs = []
M = []
Eint= []
trials = 0
trace, InRepeat=False, False

while t+dt<=1.0+1e-4:
    print("\n----\ntime:", round(t,4), " to ",round(t+dt,4),"\n----")
    # Apply BCs
    du,dfext,di=ApplyBCs(t,t+dt, BCs,ndofs)
    fr = np.delete(np.arange(ndofs), di)       # free DOFs 'b'
    fext+=dfext
    u += du

    args = (X,C,EA,dofs)
    cons = ()

    print('method being used: ',meth)
    if meth=='BFGS':
        MinPro = minimize(m,u,args=args,method=meth,jac=dm,hess=ddm,constraints=cons,tol=1e-5,options={'disp':True,'maxiter':10000})
    else:
        MinPro = minimize(m,u,args=args,method=meth,jac=dm,hess=ddm,constraints=cons,tol=1e-10,options={'disp':True,'maxiter':10000})
    u = MinPro.x
    nevals.append(MinPro.nfev)
    jacs.append(norm(MinPro.jac))
    M.append(MinPro.fun)
    Eint.append(Ei(u,X,C,EA,dofs))

    savedata(t)

    t += dt
    tt+=1
    InRepeat=False
    trials = 0      # if here means that it converged into the right contact config
    TT.append(t)

    plotMS(u.reshape(-1,3),X,C,undef=True,save=folder)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(TT,nevals,label=meth)
ax.set_xlabel("time")
ax.set_ylabel("# of evals")

plt.savefig(folder+"/results.png")

print('# of f_evals: (counting redone increments):')
print(meth,": ",sum(nevals))


print("--FINISHED--")






