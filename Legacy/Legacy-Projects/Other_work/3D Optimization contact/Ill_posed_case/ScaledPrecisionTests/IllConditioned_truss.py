from pdb import set_trace
# from re import X
import numpy as np
from numpy.linalg import norm
from math import sqrt, sin, cos, atan,atan2, acos, pi, log10, exp
from scipy.optimize import minimize,LinearConstraint, newton, Bounds, HessianUpdateStrategy,approx_fprime
# from _optimize import _minimize_bfgs as bfgs
import os, sys, pickle, csv
from functools import lru_cache
from time import strftime
import argparse
from decimal import Decimal as D
from decimal import getcontext

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

def prec_Truss2D(nx,ny,dx,dy,diags = True):
    C = []
    X = np.empty(((nx+1)*(ny+1),2),dtype=D)

    # X[(nx+1)*(ny+1)-1]= np.array([nx*dx,ny*dy])
    X[(nx+1)*(ny+1)-1]= np.array([D(nx*dx),D(ny*dy)])
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
            # X[a] = np.array([(j*dx,i*dy)])
            X[a] = np.array([D(j*dx),D(i*dy)])
            if j==nx-1:
                X[b] = np.array([D((j+1)*dx),D(i*dy)])
            if i==ny-1:
                X[c] = np.array([D(j*dx),D((i+1)*dy)])
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
        ax.plot(XX[:,0],XX[:,1], c="black" if compr[i] else "red",linewidth=0.5)
        if show_id: 
            alp=  0.3
            avg = XX[0]*(1-alp) + XX[1]*alp
            ax.text(avg[0],avg[1],avg[2],i,c="blue")

    if show:  plt.show()

def plotMS(u,X,C, ax=None, undef=False, show=False, save=None,sline = None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(221,projection='3d')
        ax2= fig.add_subplot(222,projection='3d')
        ax3= fig.add_subplot(223,projection='3d')
        ax4= fig.add_subplot(224)

    plotTruss(dec_to_float(X+u),C,ax=ax,bar_types=btypes, show_id=False,showtypes=0)
    plotTruss(dec_to_float(X+u),C,ax=ax2,bar_types=btypes,show_id=False,showtypes=0)
    plotTruss(dec_to_float(X+u),C,ax=ax3,bar_types=btypes,show_id=False,showtypes=0)

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

    # ax4.plot(TT,Eint,c='red',label='E_int')
    # ax4.plot(TT,M,c='orange',label='m_obj')
    # ax4.legend()
    # # ax4.scatter(TT[-1],jacs[-1])
    # ax4.set_xlim([0,1])
    # # ax4.set_ylim([-0.001,0.00015])

    if show: plt.show()

    if save is not None:
        if not os.path.exists(folder+"/plots"):
            os.mkdir(folder+"/plots")
        fig.tight_layout()
        plt.savefig(folder+"/plots//img"+str(tt)+".png",dpi=200,bbox_inches='tight')
        plt.close(fig)

def dec_to_float(numb):
    arr_dim = len(numb.shape)
    farray = np.zeros_like(numb)
    for ir, row in enumerate(numb):
        if arr_dim==1:
            farray[ir]= float(row)
        else:
            farray[ir]=dec_to_float(row)
    return farray


def Translate(X,disp):
    disp = np.array(disp)
    for i in range(len(X)):
        X[i] += disp
    return X

def Resize(X,factor, center = None, dir = None):
    xc = np.array([0,0,0]) if center is None else center
    newX = np.zeros_like(X)

    if dir is None:
        for idx, x in enumerate(X):
            newX[idx] = xc + factor*(x-xc)
    else:
        if dir in "Xx":
            dir = 0
        elif dir in "Yy":
            dir = 1
        elif dir in "Zz":
            dir = 2
        newX = X
        for idx, x in enumerate(X):
            newX[idx][dir] = xc[dir] + factor*(x[dir]-xc[dir])


    return newX



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
    # if type(u[0])!=D:
    #     set_trace()
    x=X+u.reshape(-1,3)
    ms=D(0.0)
    # ms=0.0
    for id_bar,(i,j) in enumerate(C):
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        dl = L-L0
        compr[id_bar] = dl/L0<=-0.01
        # dL = max(dl,0.0)
        # ms+=(EA*dL**2)/(2*L0)
        dL = max(dl,D(0.0))
        ms+=(D(EA)*dL**2)/(2*L0)
    return ms

def fi(u,X,C,EA,dofs):
    x=X+u.reshape(-1,3)
    # dms=np.zeros(ndofs,dtype=np.float128)
    dms = np.array([D(0) for _ in range(ndofs)])
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        dL = max(L-L0,D(0.0))
        dms[np.ix_(dofij)]+=D(EA)*dL/L0*dLdu
        # dL = max(L-L0,0.0)
        # dms[np.ix_(dofij)]+=EA*dL/L0*dLdu
    dms[di] = D(0.0)
    # dms[di] = 0.0
    return dms


def Ki(u,X,C,EA,dofs):
    x=X+u.reshape(-1,3)
    ddms=np.zeros((ndofs,ndofs),dtype=Decimal)
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

def line_search(f,grad,x,p,nabla,X,C,EA,dofs):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = D(1)
    c1 = 1e-4 
    c2 = 0.9 
    fx = f(x,X,C,EA,dofs)
    x_new = x + a * p 
    nabla_new = grad(x_new,X,C,EA,dofs)
    while f(x_new,X,C,EA,dofs) >= fx + (D(c1)*D(a)*nabla.T@p) or nabla_new.T@p <= D(c2)*nabla.T@p : 
        a *= D(0.5)
        x_new = x + a * p 
        nabla_new = grad(x_new,X,C,EA,dofs)
    return a


def BFGS(f,grad,x0,X,C,EA,dofs,max_it,talk=False):
    if talk: print("entered BFGS")
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method, implemented as described in Nocedal:
    Numerical Optimisation.


    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    plot:   if the problem is 2 dimensional, returns 
            a trajectory plot of the optimisation scheme.

    OUTPUTS: 
    x:      the optimal solution of the function f 

    '''

    def identidad(n):
        II=np.array([[D(1 if i==j else 0) for i in range(n)] for j in range(n)])
        return II


    d = len(x0) # dimension of problem 
    nabla = grad(x0,X,C,EA,dofs) # initial gradient 
    # H = np.eye(d) # initial hessian
    H = identidad(d) # initial hessian
    x = x0[:]
    it = 2

    while np.linalg.norm(nabla) > 1e-8: # while gradient is positive
        if talk: print("in the while loop: it=",it)
        if it > max_it:
            print('Maximum iterations reached!')
            break
        it += 1
        p = -H@nabla # search direction (Newton Method)
        a = line_search(f,grad,x,p,nabla,X,C,EA,dofs) # line search 
        s = a * p 
        x_new = x + a * p 
        nabla_new = grad(x_new,X,C,EA,dofs)
        y = nabla_new - nabla 
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1))
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        # li = (np.eye(d)-(r*((s@(y.T)))))
        # ri = (np.eye(d)-(r*((y@(s.T)))))
        li = (identidad(d)-(r*((s@(y.T)))))
        ri = (identidad(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        nabla = nabla_new[:] 
        if talk:print("jac:",norm(nabla))
        x = x_new[:]
    return x


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
    du=np.array([D(0) for _ in range(ndofs)])
    df=np.array([D(0) for _ in range(ndofs)])
    # du=np.zeros(ndofs)
    # df=np.zeros(ndofs)
    
    for BC in BCs:
        if len(BC)==3: BC.extend([0,1])
        dof_list, typ, val, t0,tf = BC
        val = D(val)
        if t0 <= tnow <= tf or t0 <= tpre <= tf:
            # load factor
            if tpre < t0:          # just entered in time interval
                LF = D((tnow - t0) / (tf - t0))
            elif tnow > tf:            # just exited time interval
                LF = D((tf - tpre) / (tf - t0))
            else:                   # going through time interval (normal case)
                LF = D((tnow - tpre ) / (tf - t0))

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




D = float
# gtol = 1e-14
gtol = 1e-10
getcontext().prec = 50

X = np.zeros((70,3))
X = np.array([[D(0.0),D(0.0),D(0.0)] for _ in range(70)])
X[:,:2], C = prec_Truss2D(6,9,0.25,0.25,diags=False)
# btypes = np.zeros_like(C)
btypes = None
dim = len(X[0])
dofs = np.array(range(dim*len(X))).reshape(-1,dim)
ndofs = len(dofs.ravel())
X = Translate(X,[D(-1.5),D(-2.25/2), D(0.0)])

X = Resize(X,1e-4)

EA = 1.0

XF = dec_to_float(X)
kin = 2

base = [0,6,63,69]
dofs_base_x = dofs[base][:,0]
dofs_base_y = dofs[base][:,1]
dofs_base_z = dofs[base][:,2]

# Rigid Circle
Cxy = [1.0, 0.0, 3.0]
Cr  = 3.1
Circ = [Cxy,Cr]
compr = [0.0]*len(C)
idx_act = []


#Step1  (Right)
bc1 = [dofs_base_x,"dir", 0.01, 0.0,1.0]
bc2 = [dofs_base_y,"dir", 0.0, 0.0,1.0]
bc3 = [dofs_base_z,"dir", 0.0, 0.0,1.0]
BCs = [bc1,bc2,bc3]

kn = kt = 1e3

TT = []
folder = "tol"+str(gtol)+"truss_decprec"+str(getcontext().prec)+"_"+strftime("%y%m%d-%H%M")
if not os.path.exists(folder):
    os.mkdir(folder)

u=np.array([D(0) for _ in range(ndofs)])
fint=np.array([D(0) for _ in range(ndofs)])
fext=np.array([D(0) for _ in range(ndofs)])
# u   =np.zeros(ndofs,dtype=np.float128)
# fint=np.zeros(ndofs,dtype=np.float128)
# fext=np.zeros(ndofs,dtype=np.float128)


t,tt=0.0, 0
dt=1.0

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
    # cons = ({'type':'eq', 'fun':Cgn, 'jac':dCgndu, 'args':args})
    cons = ()

    # u = BFGS(m,dm,u,X,C,EA,dofs,2000,talk=True)
    if D==float:
        MinPro = minimize(Ei,dec_to_float(u),args=args,method='BFGS',jac=fi,hess=Ki,constraints=cons,tol=gtol,options={'disp':True,'maxiter':10000})
    else:
        MinPro = bfgs(Ei,u,args=args,jac=fi,gtol=gtol,ret_gnorm=True,fol=folder,maxiter=10000,disp=True)
    u = MinPro.x

    Eint.append(Ei(u,X,C,EA,dofs))
    # niters.append(MinPro.nit)


    with open(folder+"/results", 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
        csvwriter = csv.writer(csvfile)
        # csvwriter.writerow(['gnorms']+MinPro.gnorms)
        csvwriter.writerow(u)
        csvwriter.writerow(['fun',MinPro.fun])
        csvwriter.writerow(['nfev',MinPro.nfev])
        csvwriter.writerow(['nit',MinPro.nit])

    # plotMS(u.reshape(-1,3),X,C,undef=True,save=folder)
    plotMS(u.reshape(-1,3),X,C,undef=True,show=True)


    t += dt
    tt+=1
    InRepeat=False
    trials = 0      # if here means that it converged into the right contact config
    TT.append(t)


print("--FINISHED--")
