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

def dms(u,X,C,EA,dofs,fr):
    x=X+u.reshape(-1,2)
    dms=np.zeros_like(x.ravel())
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        dLdu = np.array([ x[i,0]-x[j,0] , x[i,1]-x[j,1] , -x[i,0]+x[j,0], -x[i,1]+x[j,1] ])/L
        dms[np.ix_(dofij)]+=EA*((L-L0)/L0)*dLdu

    di = np.delete(np.arange(ndofs), fr)       # free DOFs 'b'

    # set_trace()
    dms[np.ix_(di)]=0.0


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

def plotMS(u,X,C, ax= None, undef = False):
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


    if ax is None: plt.show()

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


def minimize_muOLD(u0,fr,xp,mu0,mur,FNapp0,TAUapp0,tracing=False):
    if xp is None:
        u, FNapp, TAUapp = minimize_FT(u0,fr,xp,mu0,FNapp0,TAUapp0)
        mu = mu0
    else: 
        print("NOW there's a hook!")
        gt = u0[2]-xp
        # gtpre = -1
        gtpre = 10**100
        u=u0
        # fT = min(kt*gt,mu*FNapp0)*TAUapp0

        FNapp, TAUapp, mu = FNapp0, TAUapp0, mu0
        gn=-1
        niterMU = 0
        while gt<gtpre and gn<0:

            # m = ms(u,X,C,EA) - fT*u
            dmu=1e-3
            u, FNapp, TAUapp = minimize_FT(u,fr,xp,mu,FNapp,TAUapp)
            u1,FNapp1,TAUapp1= minimize_FT(u,fr,xp,mu+dmu/2,FNapp,TAUapp)
            u2,FNapp2,TAUapp2= minimize_FT(u,fr,xp,mu+dmu,FNapp,TAUapp)
            dgtdmu = ((u2[2]-xp)**2 - (u[2]-xp)**2)/dmu
            if tracing: set_trace()

            gtpre = gt
            gt = u[2]-xp
            gn = min(yc-u[3], 0.0)
            
            mu = min(mu+(mur-mu0)/20,mur)
            print("currentMU: ",mu)
            print("niterMU: ",niterMU)
            niterMU+=1
            # set_trace()
    return u, mu, FNapp, TAUapp

def minimize_mu(u,fr,xp,mu0,mur,FNapp,TAUapp,tracing=False):
    if xp is None:
        # Frictionless case
        u, FNapp, TAUapp = minimize_FT(u,fr,xp,mu0,FNapp,TAUapp)
        return u, mu0, FNapp, TAUapp

    print("NOW there's a hook!")

    # Computing frictionless case as reference
    u0, FNapp0, TAU0 = minimize_FT(u, fr, xp, 0.0,  0.0, 0.0 )
    print("just did u0 =",u0)
    gt0 = u0[2]-xp
    
    gt_tol = 1e-4
    if abs(gt0)<gt_tol:
        return u0, 0.0, FNapp0, 0.0

    # else frictional case:  0 < mu <= mur
    gtpre = 10**10
    gn=-1
    fnew,hnew, cnt = 0.0, 0.0, 0

    c, r = 0.5, 0.5
    nreset = 10
    gt, mu = gt0, 0.0
    niterMU = 0
    while abs(gt**2-gtpre**2)>1e-7 and gn<0:

        gtpre = gt
        fold = fnew
        hold = hnew

        dmu=1e-5
        u, FNapp, TAUapp = minimize_FT(u,fr,xp,mu,FNapp,TAUapp)
        u2, _   , _      = minimize_FT(u,fr,xp,mu+dmu,FNapp,TAUapp)
        gt, gt2 = u[2]-xp, u2[2]-xp
        fnew = (gt2**2 - gt**2)/dmu



        if cnt%nreset==0:
            hnew=-fnew
        else:
            beta = (fnew*fnew)/(fold*fold)
            # beta = (fnew*(fnew-fold))/(fold*fold)
            # beta = (fnew*(fnew-fold))/(hold*(fnew-fold))
            # beta = (fnew*fnew)/(hold*(fnew-fold))
            hnew = -fnew+max(0,beta)*hold

        xniter= 0 

        alpha_init = (mur-mu)/hnew
        alpha = (1/r)*alpha_init
        gtx = 10**100
        mux = np.array(mu)
        ux = np.array(u)
        FNappx = np.array( FNapp)
        TAUappx= np.array(TAUapp)

        """
        # if t>=0.099:
        #     import matplotlib.pyplot as plt
        #     fig = plt.figure()
        #     ax = fig. add_subplot(121)
        #     ax1= fig.add_subplot(122)
        #     first = True
        #     for toli in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:

        #         mus =[]
        #         gts2 = []
        #         for alpha in np.linspace(0,alpha_init,50):
        #             mux= mu + (alpha*hnew)
        #             ux, FNappx, TAUappx = minimize_FT(ux,fr,xp,mux,FNapp,TAUapp)
        #             # mark = "x"  if FNappx==-1 else "*"
        #             gtx = ux[2]-xp
        #             mus.append(mux)
        #             gts2.append(gtx**2)

        #         ax.plot(mus,gts2, label="Fn_tol="+f'{toli:.0e}')
        #         if first: ax.plot([mu,mux],[gt**2,gt**2+c*alpha*(hnew*fnew)], color="black")
        #         first = False



        #         dmui =1e-10
        #         dms = []
        #         fs = []
        #         while dmui<1:
        #             u0, FNapp, TAUapp = minimize_FT(u,fr,xp,mu,FNapp,TAUapp,tolerance = toli)
        #             u1, _   , _      = minimize_FT(u,fr,xp,mu+dmui,FNapp,TAUapp,tolerance = toli)
        #             gt0, gt1 = u0[2]-xp, u1[2]-xp
        #             f = (gt0**2 - gt1**2)/dmui
        #             dms.append(dmui)
        #             fs.append(f+1e-4)
        #             dmui *=10
        #         ax1.plot(dms,fs, label="Fn_tol="+f'{toli:.0e}')

        #     ax1.set_xscale('log')
        #     ax1.set_yscale('log')
        #     ax. legend()
        #     ax1.legend()
        #     ax. set_xlabel("mux")
        #     ax. set_ylabel("gtx^2")
        #     ax1.set_xlabel("dmu")
        #     ax1.set_ylabel("d(gt^2)/dmu")
        #     plt.show()
        """


        while gtx**2>gt**2+c*alpha*(hnew*fnew) and 0<=mux<=mur and alpha>1e-20:
            alpha = r*alpha
            mux= mu + (alpha*hnew)
            print("MUX =",mux)
            ux, FNappx, TAUappx = minimize_FT(ux,fr,xp,mux,FNappx,TAUappx)
            if FNappx==-1: 
                FNappx = np.array( FNapp)   # restores FNapp after using it as an alert to decrease alpha
                continue
            gtx = ux[2]-xp
            print("Hnew =", hnew)
            print("GTx =",gtx, "\tGT=",gt)
            print("in MU: \txniter =",xniter)
            print("ALPHA =",alpha)
            xniter += 1
            # if t>=0.119: set_trace()

        if alpha<1e-20 or gtx**2>gt**2:
            pass
        else:
            mu = np.array(mux)
            u = np.array(ux)
            FNapp = np.array( FNappx)
            TAUapp= np.array(TAUappx)
            gt = gtx
            gn = min(yc-u[3], 0.0)

        # ux, FNappx = u, FNapp
        # mu1 = mu
        # mu2 = mur
        # mux = mu2
        # while xniter<10 and mu1<mu2:
        #     ux, FNappx, TAUappx = minimize_FT(ux,fr,xp,mu1,FNappx,TAU0)
        #     # if TAUappx*TAUapp==-1:
        #     set_trace()
        #     if FNappx==-1:
        #         mu2 = mux
        #     else:
        #         mu1 = mux
        #     mux = 0.5*(mu1+mu2)
        #     gtx = u[2]-xp

        # # if alpha<1e-20 or gtx**2>gt**2:
        # #     pass
        # # else:
        # mu = np.array(mux)
        # u = np.array(ux)
        # FNapp = np.array( FNappx)
        # TAUapp= np.array(TAUappx)
        # gt = gtx
        # gn = min(yc-u[3], 0.0)






        cnt += 1

        
        # mu = min(mu+(mur-mu0)/20,mur)
        print("currentMU: ",mu)
        print("niterMU: ",niterMU)
        niterMU+=1
        # set_trace()
    return u, mu, FNapp, TAUapp


def minimize_FT(u0,fr,xp,mu0,FNapp,TAU0, lockTAU=False, tracing=False, tolerance = 1e-5):
    
    u = minimize_u(u0,fr,mu0*FNapp,TAU0)

    TAUapp=TAU0
    niter, resFN = 0, 1.0
    FNapp_pre, TAUapp_pre = 10**100,10**100
    while resFN>tolerance or TAUapp!=TAUapp_pre:
        FNapp_pre, TAUapp_pre = FNapp, TAUapp
        
        if xp is not None: gt = u[2]-xp
        gn = min(yc-u[3], 0.0)
        if gn >=0: return minimize_u(u,fr,0,0), FNapp, TAUapp

        FNapp = -kn*gn
        if xp is not None and abs(gt)>1e-4:
            TAUapp = -int(gt/abs(gt)) if not lockTAU else TAU0
            fT = mu0*FNapp
        else:
            TAUapp, fT = 0.0, 0.0

        if xp is not None and not lockTAU:
            if (TAUapp*TAU0==-1) or (TAU0==0 and niter<2 and (TAUapp_pre*gt>0)):
                return u0, -1, TAUapp_pre    # return previous values of u, and Tau plus "-1 alert" to decrease alpha in outer loop

        u = minimize_u(u,fr,fT,TAUapp)


        # if xp is not None: print("resFN:",resFN,"\tTAUapp:",TAUapp,"\tgt:",gt)
        # print("\titerFT",niter)
    
        niter+=1

        resFN = abs(FNapp-FNapp_pre)/FNapp_pre
    print("\tdone optimizing fT for FNapp: ",FNapp," and TAUapp: ",TAUapp, "\tin ",niter,"iterations")
    if xp is not None and abs(gt)>1e-4: TAUapp = -int(gt/abs(gt))
    return u, FNapp, TAUapp


def minimize_u(u0,fr,fT,Tau):

    alpha_init = 1
    c, r = 0.5, 0.5
    nreset = 10

    u = np.array(u0)
    gn = min(yc-u[3],0.0)
    mN = 1/2*kn*gn**2
    fN = kn*gn
    mnew = ms(u,X,C,EA) + mN - (fT*Tau)*u[2]
    alpha = alpha_init
    mold = 10**100
    fnew,hnew, cnt = 0.0, 0.0, 0

    while abs(mnew-mold)>tol:
        mold=mnew
        fold = fnew
        hold = hnew

        fnew = dms(u,X,C,EA,dofs,fr)
        gn = min(yc-u[3],0.0)
        fN = kn*gn*-1
        fnew[3] += fN
        fnew[2] -= fT*Tau

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
            mx = ms(ux,X,C,EA) + mN - fT*Tau*ux[2]
            xniter +=1

        mnew = mx
        u = ux
        cnt += 1

    fprint = dms(u,X,C,EA,dofs,fr)
    fprint[3] += fN
    fprint[2] -= fT*Tau
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
        ge = -mu*gn*(kn/kt)
        return u[2] - ge


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
bc2 = [[1,5],"dir",0.8, 0.0,0.1]
#Step2  (Right)
bc3 = [[0,4],"dir",1.0, 0.1,0.9]
bc4 = [[1,5],"dir",0.0, 0.1,0.9]
#Step3 (Down)
bc5 = [[0,4],"dir", 0.0, 0.9,1.0]
bc6 = [[1,5],"dir",-0.8, 0.9,1.0]
BCs = [bc1,bc2,bc3,bc4,bc5,bc6]

mur, FNapp, TAUapp = 0.5, 0, 0
kn = kt = 100
mu=0.0

tol = 1e-8
t=0
dt=0.02
while t+dt<=1.0+1e-4:
    print("\n----\ntime:", round(t,4), " to ",round(t+dt,4),"\n----")
    print("nowXP: ",xp)
    # Apply BCs
    du,df,di=ApplyBCs(t,t+dt, BCs,ndofs)
    fr = np.delete(np.arange(ndofs), di)       # free DOFs 'b'

    u += du
    fint += df

    u, mu, FNapp, TAUapp = minimize_mu(u,fr,xp,mu,mur,FNapp,TAUapp)

    print("u: ",u)
    xp = Hook(u,xp,yc)
    print("newXP: ",xp)
    print("Dx =",u[0]-u[2])

    m  = ms(u.reshape(-1,2),X,C,EA)

    # if t>0.89:
    plotMS(u.reshape(-1,2),X,C,undef=True)
    import matplotlib.pyplot as plt
    plt.show()

    t += dt




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