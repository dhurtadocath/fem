from pdb import set_trace
# from re import X
import numpy as np
from numpy.linalg import norm
from math import sqrt
from scipy.optimize import minimize


def ms(u,X,C,EA):
    
    x=X+u.reshape(-1,2)
    ms=0
    for (i,j) in C:
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        ms+=1/2*EA*L0*((L-L0)/L0)**2
    return ms

def dms(u,X,C,EA,dofs):
    x=X+u.reshape(-1,2)
    dms=np.zeros_like(x.ravel())
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        dLdu = np.array([ x[i,0]-x[j,0] , x[i,1]-x[j,1] , -x[i,0]+x[j,0], -x[i,1]+x[j,1] ])/L
        dms[np.ix_(dofij)]+=EA*((L-L0)/L0)*dLdu

    return dms

def fidiff(u0,X,C,EA, du = 1e-8):
    u0_flat = u0.ravel()
    m0 = ms(u0,X,C,EA)
    dm = np.zeros_like(u0_flat)
    for ii in range(len(dm)):
        u = np.array(u0_flat)
        u[ii] += du
        dm[ii] = (ms(u.reshape(-1,2),X,C,EA)-m0)/du
    return dm

def plotMS(u,X,C, ax= None, undef = False,show=False):
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

    ax.plot([-2,3],[0.1,0.1],c="red")
    ax.set_xlim([-2,3])
    ax.set_ylim([-1.5,0.5])

    if show: plt.show()

def plot_objective(u,X,C,du):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # U1 = [-0.1,0.1]
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

    # ax.quiver(u[2],u[3],m,du[0],du[1],0)
            
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


def mNM(s,u,fr,xp,yc,Ft,Tau, trace=False):
    k = 1e6
    u1 = np.array(u)
    u1[2] = s
    u1 = minimize_u(u1,s)
    gn = min(yc - u1[3], 0.0)
    Ft, Fn = dms(u1,X,C,EA,dofs)[2:4]
    Fn = -kn*gn
    gt = 0.0 if xp is None else u1[2]-xp
    Tau = 0.0 if (xp is None or gt==0) else -gt/abs(gt)
    mOBJ =  gt**2 + 1/2*k*(Ft*Tau-abs(Ft))**2 + 1/2*k*((mu*Fn-abs(Ft))**2)*(abs(Ft)>mu*Fn)

    return mOBJ


def minimize_u(u0,s):
    alpha_init = 1
    c, r = 0.5, 0.5
    nreset = 10

    u = np.array(u0)
    u[2]= s
    gn = min(yc-u[3],0.0)
    mN = 1/2*kn*gn**2
    fN = kn*gn
    mnew = ms(u,X,C,EA) + mN
    alpha = alpha_init
    mold = 10**100
    fnew,hnew, cnt = np.zeros(6), np.zeros(6), 0

    while abs(mnew-mold)>tol:
        mold=mnew
        fold = fnew
        hold = hnew

        fnew[3] = dms(u,X,C,EA,dofs)[3]
        gn = min(yc-u[3],0.0)
        fN = kn*gn*-1
        fnew[3] += fN

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
        ux = np.array(u)
        while mx>mnew+c*alpha*(hnew@fnew):
            alpha = r*alpha
            ux= u + (alpha*hnew)
            gn = min(yc-ux[3],0.0)
            mN = 1/2*kn*gn**2
            mx = ms(ux,X,C,EA) + mN
            xniter +=1

        mnew = mx
        u = ux
        cnt += 1

    fprint = dms(u,X,C,EA,dofs)
    fprint[3] += fN
    # print("\t\tdone optimizing u for fT*τ =",fT*Tau, "\tniter:",cnt,"\txniter:",xniter,"\tFPRINT =", norm(fprint) )

    return u

def Hook(u,xp,yc):
    gn = yc-u[3]
    if gn >=0: 
        return None
    elif xp is None:
        return u[2]
    elif abs(kt*(u[2]-xp)) <= abs(mu*gn*kn):    # elastic case
        return xp
    else:
        # ge = -mu*gn*(kn/kt)
        # return u[2] - ge
        return u[2]


X = np.array([[-1.0,-1.0],[0.0,0.0],[1.0,-1.0]])
dofs = np.array([[0,1],[2,3],[4,5]])
C = [[0,1],[1,2]]
EA = 1.0

xp, yc = None, 0.1        # hook(x)  and wall(y) position

ndofs = len(dofs.ravel())
u = np.zeros(ndofs)
fint = np.zeros(ndofs)

#Step1  (Up)
bc1 = [[0,4],"dir",0.0, 0.0,0.1]
bc2 = [[1,5],"dir",0.5, 0.0,0.1]
#Step2  (Right)
bc3 = [[0,4],"dir",1.0, 0.1,0.9]
bc4 = [[1,5],"dir",0.0, 0.1,0.9]
#Step3 (Down)
bc5 = [[0,4],"dir", 0.0, 0.9,1.0]
bc6 = [[1,5],"dir",-0.5, 0.9,1.0]
BCs = [bc1,bc2,bc3,bc4,bc5,bc6]

mu, FNapp, TAUapp = 0.5, 0, 0
kn = kt = 100

tol = 1e-8
# tol = 0.0
t=0
dt=0.005
s=0
Ft=0
Tau=0

FN,FT,TT= [],[],[]
DX, GN = [],[]

while t+dt<=1.0+1e-4:
    print("\n----\ntime:", round(t,4), " to ",round(t+dt,4),"\n----")
    print("nowXP: ",xp)
    # Apply BCs
    du,df,di=ApplyBCs(t,t+dt, BCs,ndofs)
    fr = np.delete(np.arange(ndofs), di)       # free DOFs 'b'

    u += du
    fint += df

    s = round(minimize(mNM,s,method='Nelder-Mead',args=(u,fr,xp,yc,Ft,Tau),options={'disp':False},tol=1e-12).x[0],10)
    print("s: ",s)
    u[2] = s
    u = minimize_u(u,s)
    print("u: ",u)
    xp = Hook(u,xp,yc)
    print("newXP: ",xp)
    print("Dx =",u[0]-u[2])


    Ft, Fn = -dms(u,X,C,EA,dofs)[2:4]
    gn = min(yc-u[3],0.0)
    # Fn = -kn*gn

    FN.append(Fn)
    FT.append(Ft)
    TT.append(t+dt)
    DX.append(u[0]-u[2])
    GN.append(gn)

    # if t>0.89:
    #     plotMS(u.reshape(-1,2),X,C,undef=True,show=True)

    t += dt

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(TT,FN,label='Fn')
ax.plot(TT,FT,label='Ft')
ax.plot(TT,DX,label='dX')
ax.plot(TT,GN,label='gn')
plt.legend()
plt.show()



set_trace()




























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