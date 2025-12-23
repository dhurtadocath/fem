from pdb import set_trace
# from re import X
import numpy as np
from numpy.linalg import norm
from math import sqrt, sin, cos, atan,atan2, pi, log10
from scipy.optimize import minimize,LinearConstraint, newton, Bounds
import os, sys, pickle
from functools import lru_cache
os.chdir(sys.path[0])

def Truss(nx,ny,dx,dy):
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
            C.extend([[a,d],[b,c],[b,d],[c,d]])

            X[a] = np.array([(j*dx,i*dy)])
            if j==nx-1:
                X[b] = np.array([((j+1)*dx,i*dy)])
            if i==ny-1:
                X[c] = np.array([(j*dx,(i+1)*dy)])

    return X, C

def plotTruss(X,C, show_nodes=False,show_id=False):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i,(a,b) in enumerate(C):
        XX = X[np.ix_([a,b])]
        ax.plot(XX[:,0],XX[:,1], c="blue")
        if show_id: 
            alp=  0.3
            avg = XX[0]*(1-alp) + XX[1]*alp
            ax.text(avg[0],avg[1],i,c="blue")
    if show_nodes:
        for i, x in enumerate(X):
            ax.scatter(x[0],x[1],c='red')
            if show_id: ax.text(x[0],x[1],i)

    plt.show()

def Translate(X,disp):
    disp = np.array(disp)
    for i in range(len(X)):
        X[i] += disp
    return X

def SelectNodesByBox(X,xy_a,xy_b):
    nodes = []
    xa, ya = xy_a
    xb, yb = xy_b
    for node_id, [x,y] in enumerate(X):
        if xa<=x<=xb:
            if ya<=y<=yb:
                nodes.append(node_id)
    return nodes

def SelectFlatSide(X, side, tol = 1e-6, OnSurface=False):
    minX = min(X[:,0])
    maxX = max(X[:,0])
    minY = min(X[:,1])
    maxY = max(X[:,1])
    eps = tol
    if "x" in side:
        if "-" in side:
            return SelectNodesByBox(X,[minX-eps,minY-eps],[minX+eps,maxY+eps])
        else:
            return SelectNodesByBox(X,[maxX-eps,minY-eps],[maxX+eps,maxY+eps])
    elif "y" in side:
        if "-" in side:
            return SelectNodesByBox(X,[minX-eps,minY-eps],[maxX+eps,minY+eps])
        else:
            return SelectNodesByBox(X,[minX-eps,maxY-eps],[maxX+eps,maxY+eps])


def ms(u,X,C,EA):
    
    x=X+u.reshape(-1,2)
    ms=0
    for (i,j) in C:
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        ms+=(EA*(L-L0)**2)/(2*L0)
    return ms

def dms(u,X,C,EA,dofs):
    x=X+u.reshape(-1,2)
    dms=np.zeros_like(x.ravel())
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        # L0= norm(X[i]-X[j])
        # L = norm(x[i]-x[j])
        # dLdu = np.array([ x[i,0]-x[j,0] , x[i,1]-x[j,1] , -x[i,0]+x[j,0], -x[i,1]+x[j,1] ])/L
        # dms[np.ix_(dofij)]+=EA*((L-L0)/L0)*dLdu
        uax, uay, ubx, uby = u[np.ix_(dofij)]
        Xax, Xay, Xbx, Xby = X.ravel()[np.ix_(dofij)]
        fintij = np. array([(EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),-((EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))),-((EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))])
        dms[np.ix_(dofij)]+=fintij
    return dms

def ddms(u,X,C,EA,dofs):
    x=X+u.reshape(-1,2)
    ddms=np.zeros((len(u),len(u)))
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        uax, uay, ubx, uby = u[np.ix_(dofij)]
        Xax, Xay, Xbx, Xby = X.ravel()[np.ix_(dofij)]


        Kij = np.array([[-((EA*(uax-ubx+Xax-Xbx)**2*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2)))+(EA*(uax-ubx+Xax-Xbx)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))+(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),-((EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2)))+(EA*(uax-ubx+Xax-Xbx)*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),(EA*(uax-ubx+Xax-Xbx)**2*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))-(EA*(uax-ubx+Xax-Xbx)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))-(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),(EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))-(EA*(uax-ubx+Xax-Xbx)*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))],
                        [-((EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2)))+(EA*(uax-ubx+Xax-Xbx)*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))-(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))+(EA*(uay-uby+Xay-Xby)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),(EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))-(EA*(uax-ubx+Xax-Xbx)*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),-((EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))+(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))-(EA*(uay-uby+Xay-Xby)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))],
                        [(EA*(uax-ubx+Xax-Xbx)**2*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))-(EA*(uax-ubx+Xax-Xbx)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))-(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),(EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))-(EA*(uax-ubx+Xax-Xbx)*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),-((EA*(uax-ubx+Xax-Xbx)**2*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2)))+(EA*(uax-ubx+Xax-Xbx)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))+(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),-((EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2)))+(EA*(uax-ubx+Xax-Xbx)*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))],
                        [(EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))-(EA*(uax-ubx+Xax-Xbx)*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),-((EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))+(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))-(EA*(uay-uby+Xay-Xby)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),-((EA*(uax-ubx+Xax-Xbx)*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2)))+(EA*(uax-ubx+Xax-Xbx)*(uay-uby+Xay-Xby))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)),(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)))/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))-(EA*(-sqrt((Xax-Xbx)**2+(Xay-Xby)**2)+sqrt((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))*(uay-uby+Xay-Xby)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2)**(3/2))+(EA*(uay-uby+Xay-Xby)**2)/(sqrt((Xax-Xbx)**2+(Xay-Xby)**2)*((uax-ubx+Xax-Xbx)**2+(uay-uby+Xay-Xby)**2))]])
        ddms[np.ix_(dofij,dofij)]+=Kij
    return ddms

def ddms_fd(u,X,C,EA,dofs, du=1e-7):
    ddms = np.zeros((len(u),len(u)))
    for idx in range(len(u)):
        u1 = np.array(u)
        u1[idx]+=du
        ddms[idx,:] = (dms(u1,X,C,EA,dofs) - dms(u,X,C,EA,dofs))/du
    return ddms


def f_rstr_fd(u,Circ,s,idx_act,du=1e-8):
    f = np.zeros_like(u)
    m0 = m_rstr(u,Circ,s,idx_act)
    for idx in range(len(u)):
        if idx not in fr: continue
        u1 = np.array(u)
        u1[idx]+=du
        f[idx] = (m_rstr(u1,Circ,s,idx_act) - m0)/du
    return f

def k_rstr_fd(u,Circ,s,idx_act,du=1e-8):
    f0 = f_rstr_fd(u,Circ,s,idx_act,du=10*du)
    k = np.zeros((len(u),len(u)))
    for idx in range(len(u)):
        if idx not in fr: continue
        u1 = np.array(u)
        u1[idx]+=du
        k[idx,:] = (f_rstr_fd(u1,Circ,s,idx_act,du=10*du) - f0)/du
    return k


def plotMS(u,X,C, ax=None, undef=False, show=False, save=None,sline = None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)

    x = X+u
    for (i,j) in C:
        xa, xb = x[i],x[j]
        ax.plot([xa[0],xb[0]],[xa[1],xb[1]], c="blue")
        if undef:
            Xa, Xb = X[i],X[j]
            ax.plot([Xa[0],Xb[0]],[Xa[1],Xb[1]],lw=0.5, c="grey")

    ax.add_patch(plt.Circle(Circ[0],Circ[1],color=(0.5,0.5,0.5,0.5)))
    for shi,si in zip(sh,s):
        if shi is None: continue
        s_on_circ = np.array(Cxy)-Cr*np.array([cos( si*pi),sin( si*pi)])
        sh_on_circ = np.array(Cxy)-Cr*np.array([cos( shi*pi),sin( shi*pi)])
        arr_size = 0.1
        nor = -np.array([ cos(si*pi), sin(si*pi)])
        tau =  np.array([ sin(si*pi),-cos(si*pi)])

        ax.plot([Cxy[0], s_on_circ[0]],[Cxy[1], s_on_circ[1]],"k:",linewidth=0.3)
        ax.plot([Cxy[0],sh_on_circ[0]],[Cxy[1],sh_on_circ[1]],"k:",linewidth=0.3)
        # ax.quiver(s_on_circ[0],s_on_circ[1],arr_size*nor[0],arr_size*nor[1])
        # ax.quiver(s_on_circ[0],s_on_circ[1],arr_size*tau[0],arr_size*tau[1])
        # ax.scatter(s_on_circ[0],s_on_circ[1],marker='o')
        ax.scatter(sh_on_circ[0],sh_on_circ[1],marker='x')
    if sline is not None:    
        for sl in sline:
            nor = -np.array([ cos(sl*pi), sin(sl*pi)])
            xc = np.array(Cxy)+Cr*nor
            # set_trace()
            ax.plot([Cxy[0],xc[0]],[Cxy[1],xc[1]],"r:",linewidth=0.5)


    ax.set_xlim([-1.5,3])
    ax.set_ylim([-1.5,1.5])

    if show: plt.show()
    if save is not None:
        if not os.path.exists(save):
            os.mkdir(save)

        plt.savefig(save+"mu"+str(mu)+"img"+str(tt)+".png")
        plt.close(fig)

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


def mNM(s,u0,idx_act,sh,Circ,tracePoints=False, k =1e3,):

    if tracePoints: PairsUsed.append(s)
    u = minimize(m_rstr,u0,args=(Circ,s,idx_act),method='Newton-CG',jac=f_rstr,options={'disp':False}).x
    # u = minimize(m_rstr,u0,args=(Circ,s,idx_act),method='Newton-CG',jac=f_rstr,hess=k_rstr,options={'disp':False}).x
    # u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)

    Cxy, Cr = Circ
    mOBJ,cnt_act = 0.0 ,0
    for idx in idx_act:
        si = s[cnt_act]
        cnt_act+=1
        # if si<0 or si>1: set_trace()
        nor = -np.array([ cos(si*pi), sin(si*pi)])
        tau =  np.array([ sin(si*pi),-cos(si*pi)])
        xc = np.array(Cxy)+Cr*nor
        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Fn, Ft = (F@nor)*nor, (F@tau)*tau

        xch= Cxy-Cr*np.array([cos(sh[idx]*pi),sin(sh[idx]*pi)])
        gt = norm(xc-xch)

        Tau = (xch-xc)/norm(xc-xch)     # vector pointing TOWARDS the hook
        mOBJ +=  gt**2 + 1/2*k*(Ft@Tau-norm(Ft))**2 + 1/2*k*((mu*norm(Fn)-norm(Ft))**2)*(norm(Ft)>mu*norm(Fn))

    print("u(",s,") = \t", "\tm =",mOBJ)
    # print("Fn =",Fn,"\tFt =",Ft)
    return log10(mOBJ+1)
    # return mOBJ

def NR(u0,fint,K,args=(),tol=1e-7):
    res, cnt = 10*5, 0
    u = u0
    while res>tol:
        f = fint(u,*args)
        df = K(u,*args)

        dua = u[di] - u_pre[di]      # difference due to dirichlet BCs applied
        Kba=df[np.ix_(fr,di)]
        Kbb=df[np.ix_(fr,fr)]
        finb=f[fr]
                
        dub =np.linalg.solve(Kbb,-finb-Kba.dot(dua))
        u[fr] += dub

        cnt += 1
        res = norm(f)
    return u


@lru_cache(maxsize=20)
def solve_for_a_given_s(u0,Circ,s,idx_act,cache_key):
    u_iter = minimize(m_rstr,u0,args=(Circ,s,idx_act),method='Newton-CG',jac=f_rstr,hess=k_rstr,options={'disp':False}).x
    return u_iter

def solve_for_a_given_s_wrapper(u0,Circ,s,idx_act):
    cache_key = (tuple(u0),((Circ[0][0],Circ[0][1]),Circ[1]),tuple(idx_act))
    return solve_for_a_given_s(tuple(u0),((Circ[0][0],Circ[0][1]),Circ[1]),tuple(s),tuple(idx_act),cache_key)


def mgt2(s,u0,idx_act,sh,Circ,tracePoints=True):
    # print("in OBJ")
    if tracePoints: PairsUsed.append(s)

    Cxy, Cr = Circ
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)

    mOBJ,cnt_act = 0.0 ,0
    for idx in idx_act:
        si = s[cnt_act]
        cnt_act+=1
        if si<0 or si>1: set_trace()
        nor = -np.array([ cos(si*pi), sin(si*pi)])
        xc = np.array(Cxy)+Cr*nor
        xch= Cxy-Cr*np.array([cos(sh[idx]*pi),sin(sh[idx]*pi)])
        gt = norm(xc-xch)
        mOBJ +=  gt**2

    args = (s,u0,idx_act,sh,Circ,u)
    print("s : ",s,"\tm : ",mOBJ,"\t(c1,c2)=",end="\t")
    cc1 = c1(*args)
    cc2 = c2(*args)
    if not all([ccc1>-1e-8 for ccc1 in cc1]) and not all([ccc2>-1e-8 for ccc2 in cc2]):
        print("c1 and c2 violated")
    elif not all([ccc1>-1e-8 for ccc1 in cc1]):
        print("c1 violated")
    elif not all([ccc2>-1e-8 for ccc2 in cc2]):
        print("c2 violated")
    else:
        print("constraints fulfilled! ")
    # print("s : ",s,"\tm : ",mOBJ)
    return mOBJ

def mgt2c1(s,u0,idx_act,sh,Circ):
    # print("in OBJ")
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    k=1e3

    Cxy, Cr = Circ
    mOBJ,cnt_act = 0.0 ,0
    for idx in idx_act:
        si = s[cnt_act]
        cnt_act += 1        # counter must continue even if later happens that gn>0
        xsi = xs[idx]
        gn = min( norm(xsi-Cxy)-Cr , 0.0 )
        # if gn==0: continue

        if si<0 or si>1: set_trace()
        nor = -np.array([ cos(si*pi), sin(si*pi)])
        tau =  np.array([ sin(si*pi),-cos(si*pi)])
        xc = np.array(Cxy)+Cr*nor
        xch= Cxy-Cr*np.array([cos(sh[idx]*pi),sin(sh[idx]*pi)])
        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Ft = (F@tau)*tau
        Tau = (xch-xc)/norm(xc-xch)     # vector pointing TOWARDS the hook
          
        gt = norm(xc-xch)
        mOBJ +=  gt**2 + 1/2*k*(norm(Ft)-Ft@Tau)**2

    print("s : ",s,"\tm : ",mOBJ )
    return mOBJ


# equality  
def c1(s,u0,idx_act,sh,Circ, eval = None):
    # print("in c1")
    # print("s : ",s)   
    if eval is None:
        u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    else: u = eval
    Cxy, Cr = Circ
    c1 = np.zeros(len(idx_act))
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in idx_act:
        si = s[cnt_act]
        cnt_act += 1        # counter must continue even if later happens that gn>0
        xsi = xs[idx]
        gn = min( norm(xsi-Cxy)-Cr , 0.0 )
        # if gn==0: continue
        if si<0 or si>1: set_trace()
        nor = -np.array([ cos(si*pi), sin(si*pi)])
        tau =  np.array([ sin(si*pi),-cos(si*pi)])
        xc = np.array(Cxy)+Cr*nor
        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Ft = (F@tau)*tau

        xch= Cxy-Cr*np.array([cos(sh[idx]*pi),sin(sh[idx]*pi)])

        Tau = (xch-xc)/norm(xc-xch)     # vector pointing TOWARDS the hook
        c1[cnt_act-1] =  norm(Ft)-Ft@Tau
        # c1[cnt_act-1] =  norm(Tau-Ft/norm(Ft))  # diesnt work. positive gradient in linesearch

    # print("c1 = ",c1)
    # return c1     #stric 'eq' (equality) constraint. Complicates the computation
    return (1e-8)-(c1**2)

# inequality    Ft <= mu*Fn
def c2(s,u0,idx_act,sh,Circ,eval= None):
    # print("in c2")
    # print("s : ",s)
    if eval is None:
        u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    else: u = eval
    c2 = np.zeros(len(idx_act))
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in idx_act:
        si = s[cnt_act]
        cnt_act += 1        # counter must continue even if later happens that gn>0
        xsi = xs[idx]
        gn = min( norm(xsi-Cxy)-Cr , 0.0 )
        # if gn==0: continue
        if si<0 or si>1: set_trace()
        nor = -np.array([ cos(si*pi), sin(si*pi)])
        tau =  np.array([ sin(si*pi),-cos(si*pi)])
        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Fn, Ft = (F@nor)*nor, (F@tau)*tau

        c2[cnt_act-1] = mu*norm(Fn) - norm(Ft) # must be positive (?)
    # print("c2 = ",c2)
    return c2


def m_rstr(u,Circ,s,idx_act, show=False):
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    mN, mT = 0.0 , 0.0
    # for idx,si in enumerate(s):
    cnt_act = 0
    for idx in range(ns):
        xsi = xs[idx]
        gn = min( norm(xsi-Cxy)-Cr , 0.0 )
        mN += 1/2*kn*gn**2
        if idx in idx_act:
            si = s[cnt_act]
            cnt_act+=1
            nor = -np.array([ cos(si*pi), sin(si*pi)])
            tau =  np.array([ sin(si*pi),-cos(si*pi)])
            xc = np.array(Cxy)+Cr*nor
            mT += 1/2*kt*((xsi-xc)@tau)**2
    m = ms(u,X,C,EA) + mN + mT
    if show: print("here in m (r) \t m =",m)

    return m

def f_rstr(u,Circ,s,idx_act, show=False):
    Cxy, Cr = Circ
    f = np.zeros_like(u)
    f[fr] = dms(u,X,C,EA,dofs)[fr]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in range(ns):
        xsi = xs[idx]
        gn = min( norm(xsi-Cxy)-Cr , 0.0 )
        # if gn==0: continue
        dgndu = (xsi-Cxy)/norm(xsi-Cxy)
        f[dofs[slaves[idx]]] += kn*gn*dgndu                  # += fN
        if idx in idx_act:
            si = s[cnt_act]
            cnt_act+=1
            nor = -np.array([ cos(si*pi), sin(si*pi)])
            tau =  np.array([ sin(si*pi),-cos(si*pi)])
            xc = np.array(Cxy)+Cr*nor
            f[dofs[slaves[idx]]] += kt*((xsi-xc)@tau)*tau  # += fT

    if show: print("here in Fi (r) \tRES:",norm(f))
        
    return f

def k_rstr(u,Circ,s,idx_act, show=False):
    Cxy, Cr = Circ
    kij = np.zeros((len(u),len(u)))
    kij[np.ix_(fr,fr)] = ddms(u,X,C,EA,dofs)[np.ix_(fr,fr)]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in range(ns):
        xsi = xs[idx]
        dsc = norm(xsi-Cxy)
        gn = min( dsc -Cr , 0.0 )
        if gn==0: continue
        dgndu = (xsi-Cxy)/dsc
        # dgndu = np.zeros(2) if gn==0.0 else (xsi-Cxy)/dsc
        d2gndu2 = (np.eye(2) - np.outer(dgndu,dgndu))/dsc
        kij[np.ix_(dofs[slaves[idx]],dofs[slaves[idx]])] += kn*(np.outer(dgndu,dgndu)+gn*d2gndu2)                   # += fN
        if idx in idx_act:
            si = s[cnt_act]
            cnt_act+=1
            tau =  np.array([ sin(si*pi),-cos(si*pi)])
            kij[np.ix_(dofs[slaves[idx]],dofs[slaves[idx]])] += kt*np.outer(tau,tau)  # += kij
    
    if show: print("here in Kij (r)")
    
    return kij


def GoodShape(X,u):
    x = X+u.reshape(-1,2)
    a = np.cross(x[1]-x[0],x[2]-x[0])
    b = np.cross(x[2]-x[3],x[1]-x[3])
    return (a*b>0)      #boolean

def Hook(u,sh,Circ,s):
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    shnew = [None]*ns

    for idx,(shi,xsi) in enumerate(zip(sh,xs)):
        gn = norm(xsi-Cxy)-Cr
        if gn >=0: 
            # return None
            shnew[idx] = None; continue
        
        dxy = xsi-Cxy
        si = atan(dxy[1]/dxy[0])/pi
        si = si if si>=0 else 1+si
        nor = dxy/norm(dxy)
        tau =  nor@np.array([[0,1],[-1,0]])

        
        if shi is None:
            print("node",slaves[idx],"just entered")
            # return s
            
            shnew[idx] = si; continue
        
        xch= np.array(Cxy)-Cr*np.array([cos(shi*pi),sin(shi*pi)])
        dL = (xsi-xch)@tau
        if abs(kt*dL) <= abs(mu*gn*kn):    # elastic case
            # return shi
            shnew[idx] = shi
        else:
            shnew[idx] = si
            # shnew[idx] = s[idx]

            # ge = -mu*gn*(kn/kt)     # elastic length (radius)
            # if ge == 0 : return s
            # xc = np.array(Cxy)-Cr*np.array([cos(s*pi),sin(s*pi)])
            # R = norm(xc-Cxy)
            # xa = 0.5*(Cxy+xc) + (Cr**2-ge**2)/(2*R**2)*(xc-Cxy) + 0.5*sqrt(2*(Cr**2+ge**2)/(R**2)-((Cr**2-ge**2)**2)/(R**4)-1)*(Cxy-xc)@np.array([[0,1],[-1,0]])
            # xb = 0.5*(Cxy+xc) + (Cr**2-ge**2)/(2*R**2)*(xc-Cxy) - 0.5*sqrt(2*(Cr**2+ge**2)/(R**2)-((Cr**2-ge**2)**2)/(R**4)-1)*(Cxy-xc)@np.array([[0,1],[-1,0]])
            # xn = xa if norm(xa-xch)<norm(xb-xch) else xb
            # dxyn = xn-Cxy
            # sn = atan(dxyn[1]/dxyn[0])/pi
            # return sn if sn>=0 else 1+sn
    return shnew

def getProjs(xs,Circ, idx = None):
    Cxy,Cr = Circ
    s = np.zeros(ns)
    if idx is None:
        for idx,xsi in enumerate(xs):
            dxyi = xsi-Cxy
            si=atan(dxyi[1]/dxyi[0])/pi
            si = si if si>=0 else 1+si
            s[idx] = si
        return s
    else:
        dxyi = xs[idx]-Cxy
        si=atan(dxyi[1]/dxyi[0])/pi
        return si if si>=0 else 1+si

def get_simplex(x0,delta=1e-2):
    n = len(x0)
    simplex = np.zeros((n+1, n))
    simplex[0] = x0-delta
    if type(delta) in [int,float]:
        for i in range(1, n+1):
            simplex[i] = x0.copy()
            simplex[i][i-1] += delta
    elif len(delta)==len(x0):
        # if n>1: set_trace()
        # simplex[1:]=x0+np.max(delta,0.001*np.ones_like(x0))
        simplex[1:]=x0+np.diag([delti if abs(delti)>0.001 else 0.001 for delti in delta])
    else:
        raise ValueError("delta must be int, float or array_like(x0")
    return simplex

def get_simplex2(x0,delta=1e-2,alpha=0.75):
    n = len(x0)
    simplex = np.zeros((n+1, n))
    simplex[0] = x0-delta
    for i in range(1, n+1):
        simplex[i] = x0.copy()
        # if type(delta) not in [int,float]: set_trace()
        deli = delta if type(delta) in [int,float] else delta[i-1]
        deli = 0.001 if abs(deli)<0.001 else deli   # helps to prevent delta=0
        simplex[i] += [(alpha/(1+mu))**(abs(i-j))*deli for j in range(1, n+1)]
    return simplex



X, C = Truss(6,3,0.25,0.25)
dofs = np.array(range(2*len(X))).reshape(-1,2)
X = Translate(X,[-1.5,-0.75])

slaves = SelectFlatSide(X,"+y")
base = SelectFlatSide(X,"-y")
dofs_base_x = dofs[base][:,0]
dofs_base_y = dofs[base][:,1]
ns = len(slaves)

# plotTruss(X,C,show_id=True,show_nodes=True)

EA = 1.0

# Rigid Circle
Cxy = [1.0 , 3.0]
Cr  = 3.1
Circ = [Cxy,Cr]

sh = [None]*ns   # Initial hook

ndofs = len(dofs.ravel())
u = np.zeros(ndofs)

global u_iter, s_iter, t_iter
u_iter = np.zeros(ndofs)
s_iter = np.empty(ns)
t_iter = 0

fint = np.zeros(ndofs)
# set_trace()

#Step1  (Right)
bc1 = [dofs_base_x,"dir",4.0, 0.0,1.0]
bc2 = [dofs_base_y,"dir",0.0, 0.0,1.0]
BCs = [bc1,bc2]

kn = kt = 1e3


TT = []
dX, FN, FFN, FT, RES,SS1,SS2= [],[],[],[],[],[],[]
NEVALS = []

MUs = [0.00, 0.50]



# meth = ['Nelder-Mead','SLSQP']
meth = ['L-BFGS-B','SLSQP-con','mix']
# meth = ['Nelder-Mead']

# meth = ["COBYLA"]
# opts = {'rhobeg':0.01}

# meth = ["trust-constr"]
# opts = {'initial_tr_radius':0.01}

# meth = ['BOBYQA']
# opts = {}
# opts = {'initial_trust_radius':0.01}
# opts = {'initial_trust_radius':0.01,'max_trust_radius':0.01}

folder = "results/"

NSIM = len(meth)
# for idx, mu in enumerate(MUs):
for idx in range(NSIM):

    u = np.zeros(ndofs)
    mu = 0.5
    tol = 1e-8
    t,tt=0.0, 0
    dt=0.01
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    s = getProjs(xs,Circ)
    ds = np.zeros_like(s)
    ds_pre = np.zeros_like(s)
    print("first dummy 's':",s)
    
    # plotMS(u.reshape(-1,2),X,C,undef=True,show=True)

    FNi, FFNi, FTi, RESi, S1, S2= [],[],[],[],[],[]
    nevals = []
    trace, InRepeat=False, False

    while t+dt<=1.0+1e-4:
        print("\n----\ntime:", round(t,4), " to ",round(t+dt,4),"\n----")
        print("nowSH: ",sh)
        # Apply BCs
        du,df,di=ApplyBCs(t,t+dt, BCs,ndofs)
        fr = np.delete(np.arange(ndofs), di)       # free DOFs 'b'
        u_pre = np.array(u)
        u += du
        fint += df
        s_pre = s.copy()
        for shidx, shi in enumerate(sh):
            if shi is None:
                s[shidx] = getProjs(xs,Circ,idx=shidx)
        idx_act = [i for i in range(len(sh)) if sh[i] is not None]
        print("active Nodes: ",[slaves[i] for i in range(ns) if i in idx_act])

        # s0 = (s_pre+(2*ds-ds_pre))[idx_act]
        s0 = (s_pre+(ds))[idx_act]

        print("mine NM")

        PairsUsed = []

        args = (u,idx_act,sh,Circ)
        cons = ({'type':'ineq', 'fun':c1, 'args':args},
                {'type':'ineq', 'fun':c2, 'args':args})
        # cons = ({'type':'ineq', 'fun':c2, 'args':args})
        bds = Bounds(lb=s_pre[idx_act],ub=s0+ds[idx_act])
        
        if meth[idx]=="Nelder-Mead":
            simplex = get_simplex2(s0,0.001,0.6)
            NM = minimize(mNM,s0,method='Nelder-Mead',args=(u,idx_act,sh,Circ,True),options={'bounds':bds,'disp':False,'initial_simplex':simplex,'adaptive':len(idx_act)>0},tol=1e-5)
        if meth[idx]=="SLSQP-tot":
            if len(idx_act)>0:
                NM = minimize(mNM,s0,method='SLSQP',args=args,bounds= bds,tol=1e-5) 
        elif meth[idx]=='SLSQP-con':
            if len(idx_act)>0:
                NM = minimize(mgt2,s0,method='SLSQP',args=args, constraints=cons,bounds= bds,tol=1e-5)
        elif meth[idx]=='L-BFGS-B':
            if len(idx_act)>0:
                NM = minimize(mNM,s0,method='BFGS',args=args, bounds= bds,tol=1e-5)
        elif meth[idx]=='mix':
            f=2         # simplex range factor
            mixopt, nfev = 0, 0
            while(1):
                if not mixopt:
                    if len(idx_act)>0:
                        NM = minimize(mgt2,s0,method='SLSQP',args=args, constraints=cons,bounds= bds,tol=1e-5,options={'maxfev':20,'maxiter':20})
                        nfev += NM.nfev
                else:
                    if len(idx_act)>0:
                        NM = minimize(mNM,s0,method='L-BFGS-B',args=args, bounds= bds,tol=1e-5,options={'maxfev':25,'maxiter':25})
                        nfev += NM.nfev

                    # f=f/2
                    # simplex = get_simplex2(NM.x,0.001*f,0.6)
                    # NM = minimize(mNM,s0,method='Nelder-Mead',args=(u,idx_act,sh,Circ,True),options={'maxfev':100,'bounds':bds,'disp':False,'initial_simplex':simplex,'adaptive':len(idx_act)>0},tol=1e-5)
                    # nfev += NM.nfev
                if len(idx_act)>0:
                    if not NM.success:
                        mixopt = not mixopt     # changes 0 <-> 1
                        s0 = NM.x
                    else:
                        NM.nfev = nfev
                        break
                else: break


        if len(idx_act)>0:
            s[idx_act] = NM.x
            print("s after",s)
            print("# of evals:",NM.nfev)
            if not InRepeat:
                nevals.append(NM.nfev)
            else:
                nevals[-1] += NM.nfev
        else:
            print("# of evals:",1)
            if not InRepeat:
                nevals.append(1)
            else:
                nevals[-1] += 1


        """
        if abs(t-0.18)<1e-5:
            # dels=5e-3
            # n = 100
            # sx = np.linspace(s[idx_act][0]-dels,s[idx_act][0]+dels,n)
            # sy = np.linspace(s[idx_act][1]-dels,s[idx_act][1]+dels,n)
            # mz =np.zeros((n,n))
            # colors  = np.zeros((n,n,4))
            # for idx,s1 in enumerate(sx):
            #     for jdx,s2 in enumerate(sy):
            #         # mz[idx,jdx] = log10(mNM([s1,s2],u,idx_act,sh,Circ)+1)
            #         mz[idx,jdx] = mgt2([s1,s2],u,idx_act,sh,Circ)
            #         C1 = c1([s1,s2],u,idx_act,sh,Circ)
            #         C2 = c2([s1,s2],u,idx_act,sh,Circ)
            #         # if out of both constraints
            #         if not all([cc1>0 for cc1 in C1]) and not all([cc2>0 for cc2 in C2])<0:
            #             colors[idx,jdx,:] = (1,1,0,1)   # yellow
            #         elif not all([cc1>0 for cc1 in C1]):
            #             colors[idx,jdx,:] = (1,0,0,1)   # red
            #         elif not all([cc2>0 for cc2 in C2]):
            #             colors[idx,jdx,:] = (1,0.5,0,1) # orange
            #         else:
            #             colors[idx,jdx,:] = (0,0,1,1)   # blue
            # pickle.dump([sx,sy,mz,colors],open("ObjectiveAt18xyz.dat","wb"))

            sx,sy,mz,Colors = pickle.load(open("/home/diego/fem/2D Simple Optimization contact/MultiNodeRigidCircle/ObjectiveAt18xyz.dat","rb"))

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(20,12))
            ax = fig.add_subplot(111)
            sx,sy = np.meshgrid(sx,sy)
            ax.contour(sx,sy,mz,50,alpha=0.25)
            Colors[:,:,3]=0.2
            ax.pcolormesh(sx,sy,Colors,shading='nearest')
            if meth[idx]=="Nelder-Mead":  ax.add_patch(plt.Polygon(simplex,fill=False))
            ax.scatter(s0[0],s0[1],marker="o",s=50)
            ax.scatter(s[idx_act][0],s[idx_act][1],marker="x",s=50)
            PU = np.array(PairsUsed)
            ax.scatter(PU[:,0],PU[:,1],color=(0.5,0.5,0.5,0.6),s=10)

            plt.savefig("x0At18"+meth[idx]+".svg",format='svg',dpi=1200)
            plt.show()
            plt.close(fig)
        """


        u = solve_for_a_given_s_wrapper(u,Circ,s[idx_act],idx_act)

        FF = f_rstr(u,Circ,s[idx_act],idx_act)
        print("residual out:",norm(FF))

        xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]

        sh = Hook(u,sh,Circ,s)
        print("newSH: ",sh)
        idx_hooked = [i for i in range(len(sh)) if sh[i] is not None]

        exited = list(set(idx_act)-set(idx_hooked))     # nodes no-longer penetrating
        if len(exited):
            print("nodes ",exited," exited\nREDOING INCREMENT...")
            idx_act = idx_hooked
            u = np.array(u_pre)
            xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
            s, ds = s_pre, ds_pre
            InRepeat = True
            continue
        else:
            InRepeat=False

        ds_pre = ds.copy()
        ds = s-s_pre

        for xsi,Si in zip(xs,[S1,S2]):
            dxy = xsi-Cxy
            si = atan(dxy[1]/dxy[0])/pi
            si = si if si>=0 else 1+si
            Si.append(si)

            nor = dxy/norm(dxy)
            tau =  nor@np.array([[0,1],[-1,0]])


        F = dms(u,X,C,EA,dofs)[dofs[base]].reshape(-1,2).sum(axis=0)
        
        FNi.append(norm(F[0]))
        FTi.append(norm(F[1]))
        RESi.append(log10(norm(FF)))


        # plotMS(u.reshape(-1,2),X,C,undef=True,show=True)
        plotMS(u.reshape(-1,2),X,C,undef=True,save="meth_"+meth[idx]+"/")

        t += dt
        tt+=1

        if idx==0: TT.append(t)


    FN.append(FNi)
    SS1.append(S1)
    SS2.append(S2)
    FT.append(FTi)
    RES.append(RESi)
    NEVALS.append(nevals)

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# for idx, mu in enumerate(MUs):
for idx in range(NSIM):
    ax1.plot(TT,FN[idx],label=meth[idx])
    ax2.plot(TT,FT[idx],label=meth[idx])
    ax3.plot(TT,NEVALS[idx],label=meth[idx])
    ax4.plot(TT,RES[idx],label=meth[idx])
ax1.legend(),ax2.legend(),ax3.legend(),ax4.legend()
ax1.set_xlabel("time")
ax1.set_ylabel("Fx")
ax2.set_xlabel("time")
ax2.set_ylabel("Fy")
ax3.set_xlabel("time")
ax3.set_ylabel("# of evals")
ax4.set_xlabel("time")
ax4.set_ylabel("log10(res)")
plt.show()

print('# of f_evals: (counting redone increments):')
for idx in range(NSIM):
    print(meth[idx],": ",sum(NEVALS[idx]))


print("--FINISHED--")















"""
# def m_free(u,Circ):
#     Cxy, Cr = Circ
#     mN = 0 
#     xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
#     for xsi in xs:
#         gn = min( norm(xsi-Cxy)-Cr , 0.0 )
#         mN += 1/2*kn*gn**2

#     return ms(u,X,C,EA) + mN

# def f_free(u,Circ):
#     Cxy, Cr = Circ
#     f     = np.zeros_like(u)
#     f[fr] = dms(u,X,C,EA,dofs)[fr]
#     xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
#     for idx,xsi in enumerate(xs):
#         gn = min( norm(xsi-Cxy)-Cr , 0.0 )
#         dgndu = (xsi-Cxy)/norm(xsi-Cxy)
#         f[dofs[slaves[idx]]] += kn*gn*dgndu
#     return f


# def res_rstr(u,Circ,s,idx_act):
#     return norm(f_rstr(u,Circ,s,idx_act, show=False))

# def dres_rstr(u,Circ,s,idx_act):
#     f = f_rstr(u,Circ,s,idx_act, show=False)
#     return (f/norm(f))@k_rstr(u,Circ,s,idx_act, show=False)

# def minimize_u(u0,Circ,s,fr):
#     alpha_init = 1
#     c, r = 0.5, 0.5
#     nreset = 10

#     Cxy, Cr = Circ
#     nor = -np.array([ cos(s*pi), sin(s*pi)])
#     tau =  np.array([ sin(s*pi),-cos(s*pi)])

#     xc = np.array(Cxy)+Cr*nor
#     xs = u0[2:4]

#     u = np.array(u0)
#     gn = min( norm(xs-Cxy)-Cr , 0.0 )
#     mN = 1/2*kn*gn**2
#     mT = 1/2*kt*((xs-xc)@tau)**2
#     mnew = ms(u,X,C,EA) + mN + mT
#     alpha = alpha_init
#     mold = 10**100
#     fnew,hnew, cnt = np.zeros(6), np.zeros(6), 0

#     while abs(mnew-mold)>tol:
#         mold=mnew
#         fold = fnew
#         hold = hnew

#         fnew[fr] = dms(u,X,C,EA,dofs)[fr]
#         xs = u[2:4]
#         gn = min( norm(xs-Cxy)-Cr , 0.0 )
#         fN =  kn*gn*nor
#         fT = -kt*((xc-xs)@tau)*tau
#         fnew[2:4] += fN + fT

#         if cnt%nreset==0:
#             hnew=-fnew
#         else:
#             # beta = (fnew@fnew)/(fold@fold)
#             beta=0
#             # beta = (fnew@(fnew-fold))/(fold@fold)
#             # beta = (fnew@(fnew-fold))/(hold@(fnew-fold))
#             # beta = (fnew@fnew)/(hold@(fnew-fold))
#             hnew = -fnew+max(0,beta)*hold
        
#         alpha = (1/r)*alpha_init
#         mx = 10**100
#         xniter= 0 
#         ux = np.array(u)
#         while mx>mnew+c*alpha*(hnew@fnew):
#             alpha = r*alpha
#             ux= u + (alpha*hnew)

#             # if s<0: set_trace()

#             xs = ux[2:4]
#             gn = min( norm(xs-Cxy)-Cr , 0.0 )
#             mN = 1/2*kn*gn**2
#             mT = 1/2*kt*((xs-xc)@tau)**2
#             mx = ms(ux,X,C,EA) + mN + mT
#             xniter +=1

#         mnew = mx
#         u = ux
#         cnt += 1

#     # fprint = dms(u,X,C,EA,dofs,fr)
#     # fprint[3] += fN
#     # fprint[2] -= fT*Tau
#     # print("\t\tdone optimizing u for fT*τ =",fT*Tau, "\tniter:",cnt,"\txniter:",xniter )

#     return u, fnew[2:4]

# def minimize_u_free(u0,Circ,fr,trace=False):
#     alpha_init = 1
#     c, r = 0.5, 0.5
#     nreset = 10

#     Cxy, Cr = Circ

#     xs = u0[2:4]
#     dxy = xs-Cxy
#     nor = dxy/norm(dxy)
#     u = np.array(u0)
#     gn = min( norm(xs-Cxy)-Cr , 0.0 )
#     mN = 1/2*kn*gn**2
#     mnew = ms(u,X,C,EA) + mN
#     alpha = alpha_init
#     mold = 10**100
#     fnew,hnew, cnt = np.zeros(6), np.zeros(6), 0

#     while abs(mnew-mold)>tol:
#         mold=mnew
#         fold = fnew
#         hold = hnew

#         fnew[fr] = dms(u,X,C,EA,dofs)[fr]

#         xs = u[2:4]
#         dxy = xs-Cxy
#         nor = dxy/norm(dxy)


#         gn = min( norm(xs-Cxy)-Cr , 0.0 )

#         fN =  kn*gn*nor
#         fnew[2:4] += fN

#         if cnt%nreset==0:
#             hnew=-fnew
#         else:
#             # beta = (fnew@fnew)/(fold@fold)
#             beta=0
#             # beta = (fnew@(fnew-fold))/(fold@fold)
#             # beta = (fnew@(fnew-fold))/(hold@(fnew-fold))
#             # beta = (fnew@fnew)/(hold@(fnew-fold))
#             hnew = -fnew+max(0,beta)*hold
        
#         alpha = (1/r)*alpha_init
#         mx = 10**100
#         xniter= 0 
#         ux = np.array(u)
#         while mx>mnew+c*alpha*(hnew@fnew):
#             alpha = r*alpha
#             ux= u + (alpha*hnew)
#             xs = ux[2:4]
#             gn = min( norm(xs-Cxy)-Cr , 0.0 )
#             mN = 1/2*kn*gn**2
#             mx = ms(ux,X,C,EA) + mN
#             xniter +=1

#         mnew = mx
#         u = ux
#         cnt += 1

#     return u,fnew[2:4]







"""






"""





def m1(u, g, a, kn, kt, mu, xp, yc, cub=None, showGN=False):
    a1, a2 = a
    g1, g2 = g

    # set_trace()


    mint = a1*(sqrt((1+u[0])**2+(1+u[1])**2)-sqrt(2))**2 + a2*(sqrt((1-u[0])**2+(1+u[1])**2)-sqrt(2))**2 - g1*u[0] - g2*u[1]
    gn = min(yc-u[1],0)
    
    if showGN: print(gn)

    if cub is None:
        mN = 0.5*kn*gn**2
    else:
        mN = kn/2*(gn**2-gn*cub+(cub**2)/3) if gn<cub else kn/(6*cub)*gn**3
    gt = abs(xp-u[0]) if gn<0 else 0
    gt_c = mu*(kn/kt)*abs(gn)
    elastic = kt*gt<=mu*abs(kn*gn)
    # if elastic: set_trace()
    mT = 0.5*kt*gt**2 if elastic else 0.5*kt*gt_c**2+mu*kn*abs(gn)*(gt-gt_c)
    # return mT, elastic 
    return mint + mN + mT, elastic 

def dm1(u, g, a, kn, kt, mu, xp, yc, cub=None, showGN=False):
    fint = np.array([2*a[0]*(u[0] + 1)*(sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - sqrt(2))/sqrt((u[0] + 1)**2 + (u[1] + 1)**2) + 2*a[1]*(u[0] - 1)*(sqrt((1 - u[0])**2 + (u[1] + 1)**2) - sqrt(2))/sqrt((1 - u[0])**2 + (u[1] + 1)**2) - g[0],
                     2*a[0]*(u[1] + 1)*(sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - sqrt(2))/sqrt((u[0] + 1)**2 + (u[1] + 1)**2) + 2*a[1]*(u[1] + 1)*(sqrt((1 - u[0])**2 + (u[1] + 1)**2) - sqrt(2))/sqrt((1 - u[0])**2 + (u[1] + 1)**2) - g[1]])
    gn = min(yc-u[1],0)

    dgndu = np.array([0,1])
    if cub is None:
        fN = kn*gn*dgndu
    else:
        fN = kn/2*(2*gn-cub)*dgndu if gn<cub else kn/(2*cub)*gn**2*dgndu

    gt = abs(xp-u[0]) if gn<0 else 0
    gt_c = mu*(kn/kt)*abs(gn)
    elastic = kt*gt<=mu*abs(kn*gn)

    dgtdu = np.array([1,0])
    fT = kt*gt*dgtdu if elastic else mu*kn*gn*dgtdu

    return fint + fN + fT



    def d2m1(u,g,a):
        return [[2*(a[0]*(u[0] + 1)**2/((u[0] + 1)**2 + (u[1] + 1)**2) - a[0]*(u[0] + 1)**2*(sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - sqrt(2))/((u[0] + 1)**2 + (u[1] + 1)**2)**(3/2) + a[0]*(sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - sqrt(2))/sqrt((u[0] + 1)**2 + (u[1] + 1)**2) + a[1]*(1 - u[0])*(u[0] - 1)*(sqrt((1 - u[0])**2 + (u[1] + 1)**2) - sqrt(2))/((1 - u[0])**2 + (u[1] + 1)**2)**(3/2) + a[1]*(u[0] - 1)**2/((1 - u[0])**2 + (u[1] + 1)**2) + a[1]*(sqrt((1 - u[0])**2 + (u[1] + 1)**2) - sqrt(2))/sqrt((1 - u[0])**2 + (u[1] + 1)**2)), 2*(u[1] + 1)*(a[0]*(u[0] + 1)/((u[0] + 1)**2 + (u[1] + 1)**2) - a[0]*(u[0] + 1)*(sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - sqrt(2))/((u[0] + 1)**2 + (u[1] + 1)**2)**(3/2) + a[1]*(1 - u[0])*(sqrt((1 - u[0])**2 + (u[1] + 1)**2) - sqrt(2))/((1 - u[0])**2 + (u[1] + 1)**2)**(3/2) + a[1]*(u[0] - 1)/((1 - u[0])**2 + (u[1] + 1)**2))], [2*(u[1] + 1)*(a[0]*(u[0] + 1)/((u[0] + 1)**2 + (u[1] + 1)**2) - a[0]*(u[0] + 1)*(sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - sqrt(2))/((u[0] + 1)**2 + (u[1] + 1)**2)**(3/2) + a[1]*(u[0] - 1)/((1 - u[0])**2 + (u[1] + 1)**2) - a[1]*(u[0] - 1)*(sqrt((1 - u[0])**2 + (u[1] + 1)**2) - sqrt(2))/((1 - u[0])**2 + (u[1] + 1)**2)**(3/2)), 2*(a[0]*(u[1] + 1)**2/((u[0] + 1)**2 + (u[1] + 1)**2) - a[0]*(u[1] + 1)**2*(sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - sqrt(2))/((u[0] + 1)**2 + (u[1] + 1)**2)**(3/2) + a[0]*(sqrt((u[0] + 1)**2 + (u[1] + 1)**2) - sqrt(2))/sqrt((u[0] + 1)**2 + (u[1] + 1)**2) + a[1]*(u[1] + 1)**2/((1 - u[0])**2 + (u[1] + 1)**2) - a[1]*(u[1] + 1)**2*(sqrt((1 - u[0])**2 + (u[1] + 1)**2) - sqrt(2))/((1 - u[0])**2 + (u[1] + 1)**2)**(3/2) + a[1]*(sqrt((1 - u[0])**2 + (u[1] + 1)**2) - sqrt(2))/sqrt((1 - u[0])**2 + (u[1] + 1)**2))]]


    def mCN(u,kn,yc):
        gn = u[1]-yc
        return 0.5*kn*gn**2 if gn>0 else 0

    def fCN(u,kn,yc):
        gn = u[1]-yc
        dmdgn = -kn*gn if gn>0 else 0
        dgndu = np.array([0,1])
        return dmdgn*dgndu


    def mCT(u,kt,xp):
        gt = u[0]-xp
        return 0.5*kt*gt**2

    def fCT(u,kt,xp):
        gt = u[0]-xp
        dmdgt = -kt*gt
        dgtdu = np.array([1,0])
        return dmdgt*dgtdu




    a = np.ones(2)
    g = np.zeros(2)
    u0 = np.array([-0.2,-0.015])
    kn = 100
    kt = 100
    mu = 0.2
    cub = None
    xp= -1.0
    yc = -0.01

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # U1 = [-0.1,0.1]
    U1 = [-0.1,0.1]
    n1 = 20
    eps = 5.0e-10
    U2 = [-0.1,0.1]
    n2 = 5000

    x = np.zeros((n1,n2),dtype = float)
    y = np.zeros((n1,n2),dtype = float)
    z = np.zeros((n1,n2),dtype = float)
    c = np.zeros((n1,n2,4),dtype = float)


    minM = 10**100

    for i in range(n1):
        for j in range(n2):
            u1 = ((n1-i)*U1[0] + i*(U1[1]))/n1
            u2 = ((n2-j)*U2[0] + j*(U2[1]))/n2

            x[i,j] = u1
            y[i,j] = u2
            current_m,elas = m1(np.array([u1,u2]),g,a,kn,kt,mu,xp,yc,cub=cub,showGN=False)
            minM = min(minM,current_m)
            z[i,j]=current_m
            c[i,j] = (0,0,1,0.5) if elas else (1,0,0,0.5)
            
    ax.plot_surface(x,y,z,facecolors=c)
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel("m")
    # plt.show()





    # a = np.ones(2)
    # g = np.ones(2)

    alpha_init = 0.8
    c = 0.2
    r = 0.7
    nreset = 200

    u = np.array(u0)
    mnew = m1(u, g, a, kn, kt, mu, xp, yc )[0]
    alpha = alpha_init
    mold = 10**100
    fnew = 0
    hnew = 0
    cnt = 0

    while mnew<mold:
        mold=mnew
        fold = fnew
        hold = hnew

        fnew=dm1(u,g,a,kn,kt,mu,xp,yc,cub=cub,showGN=False)
        if cnt%nreset==0:
            hnew=-fnew
        else:
            beta = (fnew@fnew)/(fold@fold)
            # beta = (fnew@(fnew-fold))/(fold@fold)
            # beta = (fnew@(fnew-fold))/(hold@(fnew-fold))
            # beta = (fnew@fnew)/(hold@(fnew-fold))
            hnew = -fnew+max(0,beta)*hold
        
        alpha = (1/r)*alpha_init
        mx = 10**100
        xniter= 0 
        while mx>mnew+c*alpha*(hnew@fnew):
            alpha = r*alpha
            ux = u + alpha*hnew
            mx = m1(ux,g,a,kn,kt,mu,xp,yc,cub=cub, )[0]
            xniter +=1

        mnew = mx
        u = ux
        cnt += 1

    m = mold
    u = u - alpha*hnew

    print("with u0=",u0,"\tcubic=",cub, ":")
    print("")
    print("u:", u)
    print("m ( CG ):", m)
    print("m (grid):", minM)
    print("gn:", u[1]-yc)
    print("niter", cnt)
    print("xniter", xniter)

    ax.scatter(u[0],u[1],m,s=5,c='r')
    plt.show()
"""