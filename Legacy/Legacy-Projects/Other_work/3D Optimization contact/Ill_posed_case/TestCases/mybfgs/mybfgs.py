import numpy as np 
import matplotlib.pyplot as plt 
from numpy.linalg import norm
import os
from pdb import set_trace
from decimal import Decimal as D
from decimal import getcontext
getcontext().prec = 28
# set_trace()

def identidad(n):
    II=np.array([[D(1 if i==j else 0) for i in range(n)] for j in range(n)])
    return II

def Ei(u,X,C,EA,dofs):
    
    x=X+u.reshape(-1,2)
    ms=D(0.0)
    for id_bar,(i,j) in enumerate(C):
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        dl = L-L0
        compr[id_bar] = dl/L0<=-0.01
        dL = max(dl,D(0.0))
        ms+=(D(EA)*dL**2)/(2*L0)
    return ms

def fi(u,X,C,EA,dofs):
    x=X+u.reshape(-1,2)
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
    dms[di] = D(0.0)
    return dms


def f(u,X,C,EA,dofs):
    return  Ei(u,X,C,EA,dofs) - fext@u

def grad(u,X,C,EA,dofs):
    # dm = np.zeros_like(fext,dtype=np.float128)
    dm = np.array([D(0) for _ in range(ndofs)])
    dm[fr] = (fi(u,X,C,EA,dofs) - fext)[fr]
    return dm

def ApplyBCs(tpre,tnow, BCs,ndofs):
    """BCs = [[ dof_list , "dir/neu", val, t0(opt),tf(opt) ],[...],...]"""
    diri = []
    # du = np.zeros(ndofs,dtype=np.float128)
    # df = np.zeros(ndofs,dtype=np.float128)
    du=np.array([D(0) for _ in range(ndofs)])
    df=np.array([D(0) for _ in range(ndofs)])
    
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


def BFGS(f,grad,x0,X,C,EA,dofs,max_it,plot=False):
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
    if plot == True: 
        if d == 2: 
            x_store =  np.zeros((1,2)) # storing x values 
            x_store[0,:] = x 
        else: 
            print('Too many dimensions to produce trajectory plot!')
            plot = False

    while np.linalg.norm(nabla) > 1e-20: # while gradient is positive
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
        x = x_new[:]
        if plot == True:
            x_store = np.append(x_store,[x],axis=0) # storing x
    if plot == True:
        x1 = np.linspace(min(x_store[:,0]-0.5),max(x_store[:,0]+0.5),30)
        x2 = np.linspace(min(x_store[:,1]-0.5),max(x_store[:,1]+0.5),30)
        X1,X2 = np.meshgrid(x1,x2)
        Z = f([X1,X2])
        plt.figure()
        plt.title('OPTIMAL AT: '+str(x_store[-1,:])+'\n IN '+str(len(x_store))+' ITERATIONS')
        plt.contourf(X1,X2,Z,30,cmap='jet')
        plt.colorbar()
        plt.plot(x_store[:,0],x_store[:,1],c='w')
        plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
        plt.show()
    return x

n_elems = 10
X = np.array([[D(0.0),D(0.0)] for _ in range(n_elems+1)])
X[:,0] = np.array([D(i) for i in range(n_elems+1)])
C = [[i,i+1] for i in range(n_elems)]

btypes = None
dim = len(X[0])
dofs = np.array(range(dim*len(X))).reshape(-1,dim)
ndofs = len(dofs.ravel())
EA = 1.0
kin = 2

base = [0,n_elems]
dofs_base_x = dofs[base][:,0]
dofs_base_y = dofs[base][:,1]

# Rigid Circle
Cxy = [1.0, 0.0, 3.0]
Cr  = 3.1
Circ = [Cxy,Cr]
compr = [0.0]*len(C)
idx_act = []

#Step1  (Right)
bc1 = [dofs_base_x,"dir", 0.0, 0.0,1.0]
bc2 = [dofs_base_y,"dir", 0.1, 0.0,1.0]
BCs = [bc1,bc2]

kn = kt = 1e3
tol = 1e-8

TT = []

# u = np.zeros(ndofs,dtype=np.float128)
# fint = np.zeros(ndofs,dtype=np.float128)
# fext = np.zeros(ndofs,dtype=np.float128)
u=np.array([D(0) for _ in range(ndofs)])
fint=np.array([D(0) for _ in range(ndofs)])
fext=np.array([D(0) for _ in range(ndofs)])

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

    u = BFGS(f,grad,u,X,C,EA,dofs,2000,plot=False)

    t += dt
    tt+=1
    InRepeat=False
    trials = 0      # if here means that it converged into the right contact config
    TT.append(t)

    # plotMS(u.reshape(-1,2),X,C,undef=True,show=True)
    # plotMS(u.reshape(-1,3),X,C,undef=True,save=folder)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
x = X+u.reshape(-1,2)
ax.plot(x[:,0],x[:,1])
plt.show()


