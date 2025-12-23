from pdb import set_trace
# from re import X
import numpy as np
from numpy.linalg import norm
from math import sqrt, sin, cos, atan,atan2, pi
from scipy.optimize import minimize
import os, sys
os.chdir(sys.path[0])


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
    f0 = f_rstr_fd(u,Circ,s,idx_act,du=du)
    k = np.zeros((len(u),len(u)))
    for idx in range(len(u)):
        if idx not in fr: continue
        u1 = np.array(u)
        u1[idx]+=du
        k[idx,:] = (f_rstr_fd(u1,Circ,s,idx_act,du=du) - f0)/du
    return k



def fidiff(u0,X,C,EA, du = 1e-8):
    u0_flat = u0.ravel()
    m0 = ms(u0,X,C,EA)
    dm = np.zeros_like(u0_flat)
    for ii in range(len(dm)):
        u = np.array(u0_flat)
        u[ii] += du
        dm[ii] = (ms(u.reshape(-1,2),X,C,EA)-m0)/du
    return dm

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

def plot_objective(u,X,C,du):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    U = np.array(u)

    U1 = [U[2]-0.2,U[2]+0.2]
    n1 = 20
    U2 = [U[3]-0.2,U[3]+0.2]
    n2 = 50

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
            U[2] = u1
            U[3] = u2
            gn = min(yc-U[3],0.0)
            mN = 1/2*kn*gn**2

            gt = U[2]-xp
            FN = -gn*kn
            gte = -0.2*gn*(kn/kt)
            if gn<0:
                # set_trace()
                mT = 1/2*kt*gt**2 if (kt*gt<0.2*FN) else 1/2*kt*gte**2 + 0.2*FN*(gt-gte)
            else:
                mT = 0

            current_m= ms(U,X,C,EA) + mN +mT
            minM = min(minM,current_m)
            z[i,j]=current_m
            c[i,j] = (1,0.2,0.2,0.5)

    gn = min(yc-u[3],0.0)
    mN = 1/2*kn*gn**2
    m= ms(u,X,C,EA) + mN

    ax.scatter(u[2],u[3],m, c="blue",s=5)

    ax.plot_surface(x,y,z,facecolors=c)
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel("m")
    plt.show()

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


def mNM(s,u0,idx_act,sh,Circ):
    xs = X.ravel()[dofs[slaves]]+u0[dofs[slaves]]
    k = 1e3
    Cxy, Cr = Circ

    # u = minimize(m_rstr,u0,args=(Circ,s,idx_act),method='Newton-CG',jac=f_rstr,hess=k_rstr,options={'disp':False}).x
    u = minimize(m_rstr,u0,args=(Circ,s,idx_act),method='Newton-CG',jac=f_rstr,options={'disp':False}).x
    
    # plotMS(u.reshape(-1,2),X,C,undef=True,show=True,sline=s)
    # if t >0.04: set_trace()
    
    # if not GoodShape(X,u): return 10^10
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    mOBJ = 0
    # for idx,si in enumerate(s):
    cnt_act = 0
    # for idx in range(ns):
    for idx in idx_act:
        xsi = xs[idx]
        gn = min( norm(xsi-Cxy)-Cr , 0.0 )
        # if gn>=0: continue
        # if sh[idx] is None: continue
        # if idx not in idx_act: continue
        si = s[cnt_act]
        cnt_act+=1
        if si<0 or si>1: set_trace()
        nor = -np.array([ cos(si*pi), sin(si*pi)])
        tau =  np.array([ sin(si*pi),-cos(si*pi)])
        xc = np.array(Cxy)+Cr*nor
        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Fn, Ft = (F@nor)*nor, (F@tau)*tau

        # xch= None if sh[idx] is None else np.array(Cxy)-Cr*np.array([cos(sh[idx]*pi),sin(sh[idx]*pi)])
        xch= Cxy-Cr*np.array([cos(sh[idx]*pi),sin(sh[idx]*pi)])
        # gt = 0.0 if (sh[idx] is None or gn>=0) else norm(xc-xch)
        gt = norm(xc-xch)

        Tau = (xch-xc)/norm(xc-xch)     # vector pointing towards the hook
        mOBJ +=  gt**2 + 1/2*k*(Ft@Tau-norm(Ft))**2 + 1/2*k*((mu*norm(Fn)-norm(Ft))**2)*(norm(Ft)>mu*norm(Fn))

    # print("Fn =",Fn,"\tFt =",Ft)
    return mOBJ


def m_free(u,Circ):
    Cxy, Cr = Circ
    mN = 0 
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    for xsi in xs:
        gn = min( norm(xsi-Cxy)-Cr , 0.0 )
        mN += 1/2*kn*gn**2

    return ms(u,X,C,EA) + mN

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
        # if sh[idx] is not None:
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

def f_free(u,Circ):
    Cxy, Cr = Circ
    f     = np.zeros_like(u)
    f[fr] = dms(u,X,C,EA,dofs)[fr]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    for idx,xsi in enumerate(xs):
        gn = min( norm(xsi-Cxy)-Cr , 0.0 )
        dgndu = (xsi-Cxy)/norm(xsi-Cxy)
        f[dofs[slaves[idx]]] += kn*gn*dgndu
    return f

def f_rstr(u,Circ,s,idx_act, show=False):
    Cxy, Cr = Circ
    f = np.zeros_like(u)
    f[fr] = dms(u,X,C,EA,dofs)[fr]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    # for idx,si in enumerate(s):
    cnt_act = 0
    for idx in range(ns):
        xsi = xs[idx]
        gn = min( norm(xsi-Cxy)-Cr , 0.0 )
        dgndu = (xsi-Cxy)/norm(xsi-Cxy)
        f[dofs[slaves[idx]]] += kn*gn*dgndu                  # += fN
        # if sh[idx] is not None:
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
    # set_trace()
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    kij = np.zeros((len(u),len(u)))
    kij[np.ix_(fr,fr)] = ddms(u,X,C,EA,dofs)[np.ix_(fr,fr)]
    # for idx,si in enumerate(s):
    cnt_act = 0
    for idx in range(ns):
        xsi = xs[idx]
        dsc = norm(xsi-Cxy)
        gn = min( dsc -Cr , 0.0 )
        dgndu = (xsi-Cxy)/dsc
        d2gndu2 = np.eye(2)/dsc - np.outer(xsi-Cxy,xsi-Cxy)/dsc**3
        kij[np.ix_(dofs[slaves[idx]],dofs[slaves[idx]])] += kn*(np.outer(dgndu,dgndu)+gn*d2gndu2)                   # += fN
        # if sh[idx] is not None:
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

def minimize_u(u0,Circ,s,fr):
    alpha_init = 1
    c, r = 0.5, 0.5
    nreset = 10

    Cxy, Cr = Circ
    nor = -np.array([ cos(s*pi), sin(s*pi)])
    tau =  np.array([ sin(s*pi),-cos(s*pi)])

    xc = np.array(Cxy)+Cr*nor
    xs = u0[2:4]

    u = np.array(u0)
    gn = min( norm(xs-Cxy)-Cr , 0.0 )
    mN = 1/2*kn*gn**2
    mT = 1/2*kt*((xs-xc)@tau)**2
    mnew = ms(u,X,C,EA) + mN + mT
    alpha = alpha_init
    mold = 10**100
    fnew,hnew, cnt = np.zeros(6), np.zeros(6), 0

    while abs(mnew-mold)>tol:
        mold=mnew
        fold = fnew
        hold = hnew

        fnew[fr] = dms(u,X,C,EA,dofs)[fr]
        xs = u[2:4]
        gn = min( norm(xs-Cxy)-Cr , 0.0 )
        fN =  kn*gn*nor
        fT = -kt*((xc-xs)@tau)*tau
        fnew[2:4] += fN + fT

        if cnt%nreset==0:
            hnew=-fnew
        else:
            # beta = (fnew@fnew)/(fold@fold)
            beta=0
            # beta = (fnew@(fnew-fold))/(fold@fold)
            # beta = (fnew@(fnew-fold))/(hold@(fnew-fold))
            # beta = (fnew@fnew)/(hold@(fnew-fold))
            hnew = -fnew+max(0,beta)*hold
        
        alpha = (1/r)*alpha_init
        mx = 10**100
        xniter= 0 
        ux = np.array(u)
        while mx>mnew+c*alpha*(hnew@fnew):
            alpha = r*alpha
            ux= u + (alpha*hnew)

            # if s<0: set_trace()

            xs = ux[2:4]
            gn = min( norm(xs-Cxy)-Cr , 0.0 )
            mN = 1/2*kn*gn**2
            mT = 1/2*kt*((xs-xc)@tau)**2
            mx = ms(ux,X,C,EA) + mN + mT
            xniter +=1

        mnew = mx
        u = ux
        cnt += 1

    # fprint = dms(u,X,C,EA,dofs,fr)
    # fprint[3] += fN
    # fprint[2] -= fT*Tau
    # print("\t\tdone optimizing u for fT*τ =",fT*Tau, "\tniter:",cnt,"\txniter:",xniter )

    return u, fnew[2:4]

def minimize_u_free(u0,Circ,fr,trace=False):
    alpha_init = 1
    c, r = 0.5, 0.5
    nreset = 10

    Cxy, Cr = Circ

    xs = u0[2:4]
    dxy = xs-Cxy
    nor = dxy/norm(dxy)
    u = np.array(u0)
    gn = min( norm(xs-Cxy)-Cr , 0.0 )
    mN = 1/2*kn*gn**2
    mnew = ms(u,X,C,EA) + mN
    alpha = alpha_init
    mold = 10**100
    fnew,hnew, cnt = np.zeros(6), np.zeros(6), 0

    while abs(mnew-mold)>tol:
        mold=mnew
        fold = fnew
        hold = hnew

        fnew[fr] = dms(u,X,C,EA,dofs)[fr]

        xs = u[2:4]
        dxy = xs-Cxy
        nor = dxy/norm(dxy)


        gn = min( norm(xs-Cxy)-Cr , 0.0 )

        fN =  kn*gn*nor
        fnew[2:4] += fN

        if cnt%nreset==0:
            hnew=-fnew
        else:
            # beta = (fnew@fnew)/(fold@fold)
            beta=0
            # beta = (fnew@(fnew-fold))/(fold@fold)
            # beta = (fnew@(fnew-fold))/(hold@(fnew-fold))
            # beta = (fnew@fnew)/(hold@(fnew-fold))
            hnew = -fnew+max(0,beta)*hold
        
        alpha = (1/r)*alpha_init
        mx = 10**100
        xniter= 0 
        ux = np.array(u)
        while mx>mnew+c*alpha*(hnew@fnew):
            alpha = r*alpha
            ux= u + (alpha*hnew)
            xs = ux[2:4]
            gn = min( norm(xs-Cxy)-Cr , 0.0 )
            mN = 1/2*kn*gn**2
            mx = ms(ux,X,C,EA) + mN
            xniter +=1

        mnew = mx
        u = ux
        cnt += 1

    return u,fnew[2:4]

def Hook(u,sh,Circ):
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
            # return s
            shnew[idx] = si

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

# X = np.array([[-1.0,-1.0],[0.0,0.0],[1.0,-1.0]])
# dofs = np.array([[0,1],[2,3],[4,5]])
# C = [[0,1],[1,2]]

X, C = Truss(1,1,1,1)
dofs = np.array(range(2*len(X))).reshape(-1,2)
X = Translate(X,[-1,-1])
slaves = [2,3]
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
fint = np.zeros(ndofs)
# set_trace()

#Step1  (Right)
bc1 = [[0,2],"dir",4.0, 0.0,1.0]
bc2 = [[1,3],"dir",0.0, 0.0,1.0]
BCs = [bc1,bc2]

kn = kt = 1e3


TT = []
dX, FN, FFN, FT, RES,SS1,SS2= [],[],[],[],[],[],[]

# MUs = [0.0]
# MUs = [0.00, 0.25, 0.50, 0.75, 1.00]
MUs = [0.00, 0.50]

folder = "results/"

for idx, mu in enumerate(MUs):

    u = np.zeros(ndofs)

    tol = 1e-8
    t,tt=0.0, 0
    dt=0.01
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    s = getProjs(xs,Circ)
    print("first dummy 's':",s)
    
    
    # plotMS(u.reshape(-1,2),X,C,undef=True,show=True)

    FNi, FFNi, FTi, RESi, S1, S2= [],[],[],[],[],[]
    trace=False

    while t+dt<=1.0+1e-4:
        print("\n----\ntime:", round(t,4), " to ",round(t+dt,4),"\n----")
        print("nowSH: ",sh)
        # Apply BCs
        du,df,di=ApplyBCs(t,t+dt, BCs,ndofs)
        fr = np.delete(np.arange(ndofs), di)       # free DOFs 'b'
        u_pre = np.array(u)
        u += du
        fint += df

        if all(i is None for i in sh):
            # u, FF = minimize_u_free(u,Circ,fr,trace=trace)
            u = minimize(m_free,u,args=(Circ),method='CG',jac=f_free,options={'disp':False}).x
            FF = f_free(u,Circ)
            xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
            gn = [min( norm(xsi-Cxy)-Cr , 0.0 ) for xsi in xs]
        else:
            for shidx, shi in enumerate(sh):
                if shi is None:
                    s[shidx] = getProjs(xs,Circ,idx=shidx)
            idx_act = [i for i in range(len(sh)) if sh[i] != None]
            print("active Nodes: ",[slaves[i] for i in range(ns) if i in idx_act])
            s[idx_act] = minimize(mNM,s[idx_act],method='Nelder-Mead',args=(u,idx_act,sh,Circ),options={'disp':False},tol=1e-5).x
            print("s from N-M: ",s)
            # u, FF = minimize_u(u,Circ,s,fr)
            
            # u = minimize(m_rstr,u,args=(Circ,s[idx_act],idx_act),method='Newton-CG',jac=f_rstr,hess=k_rstr,options={'disp':False}).x
            u = minimize(m_rstr,u,args=(Circ,s[idx_act],idx_act),method='Newton-CG',jac=f_rstr,options={'disp':False}).x

            FF = f_rstr(u,Circ,s[idx_act],idx_act)
            print("residual out:",norm(FF))

            # # verif
            # xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
            # gn = min( norm(xs-Cxy)-Cr , 0.0 )
            # if gn>=0:
            #     # set_trace()
            #     # trace=True
            #     sh = None
            #     u = np.array(u_pre)
            #     continue

        xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]

        print("u: ",u)
        sh = Hook(u,sh,Circ)
        print("newSH: ",sh)

        for xsi,Si in zip(xs,[S1,S2]):
            dxy = xsi-Cxy
            si = atan(dxy[1]/dxy[0])/pi
            si = si if si>=0 else 1+si
            Si.append(si)

            nor = dxy/norm(dxy)
            tau =  nor@np.array([[0,1],[-1,0]])


        F = dms(u,X,C,EA,dofs)[0:4].reshape(-1,2).sum(axis=0)
        # Fn, Ft = (F@nor)*nor, (F@tau)*tau
        
        # if t>0.06:set_trace()

        from math import log10
        FNi.append(norm(F[0]))
        # FFNi.append(-kn*gn)
        FTi.append(norm(F[1]))
        RESi.append(log10(norm(FF)))

        # if abs(t-0.85)<1e-4:
        #     set_trace()
        #     SI, NMNM = np.linspace(0.574,0.575,200),[]
        #     for si in SI:
        #         NMNM.append(mNM(si,u,fr,sh,Circ,Ft,Tau))
        #     import matplotlib.pyplot as plt
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.plot(SI,NMNM)
        #     plt.show()

        #     plotMS(u.reshape(-1,2),X,C,undef=True,show=True)


        # if t>0.06: plotMS(u.reshape(-1,2),X,C,undef=True,show=True)
        plotMS(u.reshape(-1,2),X,C,undef=True,save="mu"+str(mu)+"/")

        t += dt
        tt+=1

        if idx==0: TT.append(t)

    FN.append(FNi)
    SS1.append(S1)
    SS2.append(S2)
    FT.append(FTi)
    RES.append(RESi)

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
for idx, mu in enumerate(MUs):
    ax1.plot(TT,FN[idx],label=str(mu))
    ax2.plot(TT,FT[idx],label=str(mu))
    ax3.plot(TT,SS1[idx],label=str(mu)+"n"+str(slaves[0]))
    ax3.plot(TT,SS2[idx],label=str(mu)+"n"+str(slaves[1]))
    ax4.plot(TT,RES[idx],label=str(mu))
ax1.legend(),ax2.legend(),ax3.legend(),ax4.legend()
ax1.set_xlabel("time")
ax1.set_ylabel("Fx")
ax2.set_xlabel("time")
ax2.set_ylabel("Fy")
ax3.set_xlabel("time")
ax3.set_ylabel("proj 's'")
ax4.set_xlabel("time")
ax4.set_ylabel("log10(res)")
plt.show()








print("--FINISHED--")



























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