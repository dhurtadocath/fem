from pdb import set_trace
# from re import X
import numpy as np
from numpy.linalg import norm
from math import sqrt, sin, cos, atan,atan2, pi, log10, log
from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint, newton, Bounds, approx_fprime
import os, sys, pickle, csv
from functools import lru_cache
from time import strftime
import matplotlib.pyplot as plt

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

def UglyTruss(nx,ny,dx,dy):
    """This functions creates a truss with a node in the middle of each cell"""
    C = []
    X = np.empty(((nx+1)*(ny+1)+nx*ny,2),dtype=np.double)

    X[(nx+1)*(ny+1)-1]= np.array([nx*dx,ny*dy])
    for i in range(ny):
        for j in range(nx):
            a = (nx+1)*i+j
            b = (nx+1)*i+j+1
            c = (nx+1)*(i+1)+j
            d = (nx+1)*(i+1)+j+1
            e = (nx+1)*(ny+1)+nx*i+j

            if i==0:
                C.append([a,b])
            if j==0:
                C.append([a,c])
            C.extend([[a,e],[b,e],[c,e],[d,e],[b,d],[c,d]])

            X[a] = np.array([(j*dx,i*dy)])
            if j==nx-1:
                X[b] = np.array([((j+1)*dx,i*dy)])
            if i==ny-1:
                X[c] = np.array([(j*dx,(i+1)*dy)])

            X[e] = np.array([((j+0.5)*dx,(i+0.5)*dy)])

    return X, C

def plotTruss(X,C, show_nodes=False,show_id=False, ax = None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i,(a,b) in enumerate(C):
        XX = X[np.ix_([a,b])]
        ax.plot(XX[:,0],XX[:,1], c="blue",linewidth=0.5)
        if show_id: 
            alp=  0.3
            avg = XX[0]*(1-alp) + XX[1]*alp
            ax.text(avg[0],avg[1],i,c="blue")
    if show_nodes:
        actives = [slaves[slid] for slid in idx_act]
        for i, x in enumerate(X):
            if i not in slaves: continue
            color = "red" if  i in actives else "blue" 
            ax.scatter(x[0],x[1],c=color,s=5)
            if show_id: ax.text(x[0],x[1],i)

    if ax is None:  plt.show()

def plotCircle(Circ,ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.add_patch(plt.Circle(Circ[0],Circ[1],color=(0.5,0.5,0.5,0.5)))
    if ax is None:  plt.show()

def plotMS(u,X,C, ax=None, undef=False, show=False, save=None,sline = None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)

    plotTruss(X+u.reshape(-1,2),C, show_nodes=False,show_id=False,ax=ax)
    plotCircle(Circ,ax=ax)

    ax.set_xlim([-1.5,3])
    ax.set_ylim([-1.5,1.5])

    if save is not None:
        if not os.path.exists(folder+"/plots"):
            os.mkdir(folder+"/plots")

        plt.savefig(folder+"/plots//img"+str(incr+1)+".png")
        if not show: plt.close(fig)
        else: plt.show()
        return None

    if show: plt.show()


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


def ms(u,fr,X,C,EA):
    
    x=X+u.reshape(-1,2)
    ms=0
    for (i,j) in C:
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        # ms+=(EA*(L-L0)**2)/(2*L0) # spring
        # ms+=EA*(L0-L+L*log(L/L0))    # ABQ's truss
        ms+=EA*L0*(0.5*(log(L/L0))**2)    #lars

    return ms

def dms(u,fr,X,C,EA,dofs):
    x=X+u.reshape(-1,2)
    dms=np.zeros(ndofs)
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        # dms[np.ix_(dofij)]+=EA*(L-L0)/L0*dLdu   # spring
        # dms[np.ix_(dofij)]+=EA*log(L/L0)*dLdu    # ABQ's truss
        dms[np.ix_(dofij)]+=(EA*L0/L)*log(L/L0)*dLdu    #lars
    diri = np.delete(np.arange(ndofs), fr)
    dms[diri] = 0
    return dms

def ddms(u,fr,X,C,EA,dofs):
    # consider changing to chain-rule expressions as in 3D case
    x=X+u.reshape(-1,2)
    ddms=np.zeros((len(u),len(u)),dtype=np.float64)
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        dadu = np.array([[ 1.0, 0.0,-1.0, 0.0],
                         [ 0.0, 1.0, 0.0,-1.0],
                         [-1.0, 0.0, 1.0, 0.0],
                         [ 0.0,-1.0, 0.0, 1.0]])
        d2Ldu2 = dadu/L - np.outer(a,a)/(L**3)
        # ddms[np.ix_(dofij,dofij)] += (np.outer(dLdu,dLdu)+(L-L0)*d2Ldu2)/L0
        # ddms[np.ix_(dofij,dofij)] += EA*(1/L*np.outer(dLdu,dLdu)+log(L/L0)*d2Ldu2) # ABQ's truss
        ddms[np.ix_(dofij,dofij)] += EA*L0*((1-log(L/L0))/L**2*np.outer(dLdu,dLdu)+log(L/L0)/L*d2Ldu2) #lars

    return EA*ddms


def ddms_fd(u,X,C,EA,dofs, du=1e-8):
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


def mNM(s,u0,fr,idx_act,sh,Circ, k =1e3,):

    Cxy, Cr = Circ

    # idxs_hooked = [idx for idx in range(ns) if sh[idx]!=-1]
    # fr_u = fr.copy()
    # ih_count = 0
    # for ih in idxs_hooked:
    #     node = slaves[ih]
    #     dofi = dofs[node]
    #     si = s[ih_count]
    #     nor = get_nor(si)
    #     xc = Cxy+Cr*nor
    #     u_imposed = xc-X[node]
    #     u0[dofi] = u_imposed
    #     fr_u = list(set(fr_u) - set(dofi)) 
    #     ih_count += 1

    u = solve_for_a_given_s_wrapper(u0,fr,Circ,s,idx_act)
    # s=s.reshape(-1,2)
    mOBJ,cnt_act = 0.0 ,0
    for idx in idx_act:
        if sh[idx] == -1: continue
        si = s[cnt_act]
        cnt_act+=1
        nor = get_nor(si)
        nor_h = get_nor(sh[idx])
        xc = np.array(Cxy)+Cr*nor
        xch= Cxy+Cr*nor_h
        gt = norm(xc-xch)

        F = dms(u,fr,X,C,EA,dofs)[dofs[slaves[idx]]]
        Fn = (F@nor)*nor
        Ft = F-Fn
        
        # Tau = (xch-xc)/norm(xc-xch)     # vector pointing TOWARDS the hook
        dsh = xch-xc
        Taup = dsh - (dsh@nor)*nor
        Tau = Taup/norm(Taup)
        # print("node",idx,':',"Ft/Fn=",norm(Ft)/(F@nor),,end="\t")
        print(f"node {idx}: gt2={gt**2:.3}, Ft/Fn={norm(Ft)/(F@nor):.3f}",end="\t")
        mOBJ +=  gt**2 + 1/2*k*(Ft@Tau-norm(Ft))**2 + 1/2*k*((mu*norm(Fn)-norm(Ft))**2)*(norm(Ft)>mu*norm(Fn))
        # breakpoint()

    print("\n")
    # print("Fn =",Fn,"\tFt =",Ft)
    # return log10(mOBJ+1)
    return mOBJ, u

def NR(u0,fr,fK,args=(),tol=1e-10,verbose=0):
    res, cnt = 10*5, 0
    u = u0

    if type(u) is tuple:
        u = np.array(u)
    u_pre = u.copy()
    f,df = fK(u,*args)
    while res>tol:
        # df = K(u_pre,*args)

        dua = u[di] - u_pre[di]      # difference due to dirichlet BCs applied
        Kba=df[np.ix_(fr,di)]
        Kbb=df[np.ix_(fr,fr)]
        finb=f[fr]
                
        dub =np.linalg.solve(Kbb,-finb-Kba.dot(dua))
        u[fr] += dub
        # f = fint(u,*args)
        f,df = fK(u,*args)

        u_pre = u

        cnt += 1
        res = norm(f)

        if cnt>10:
            return False

    if verbose: print("NR performed",cnt,"iterations")
    return u

def CG_mf_constr(u,X,C,fr,actives,mu,ratio,k_pen,Circ,slaves,dofs,s_opt,sh,k_op,tol,fd_step):
    fd_method = 'central'
    
    s = s_opt.copy()
    c= 0.5
    r = 0.5
    nreset = 10

    mnew, unew = mNM(s,u,fr,actives,sh,Circ, k_op,)

    mold = 10^100
    fnew = 10**5*np.ones_like(s)
    hnew = np.zeros_like(s)
    cnt = 0
    tot_lnsrch = 0
    fold = 10^100

    reset = False
    steepest_descent_counter = 0  

    while (mold-mnew) > tol*abs(mold) or reset:
        mold = mnew
        fold = fnew
        hold = hnew

        print("\nCOMPUTING DERIVATIVE...")
        fnew = numericalDerivative(lambda sv: mNM(sv,u,fr,actives,sh,Circ, k_op)[0],s,fd_method,fd_step)

        if reset:
            hnew = -fnew
            reset = False
        else:
            # beta = (fnew@fnew)/(fold@fold)                   # (  I)
            beta = (fnew@(fnew-fold))/(fold@fold)            # ( II)
            # beta = (fnew@(fnew-fold))/(hold@(fnew-fold))     # (III)
            # beta = (fnew@fnew)/(hold@(fnew-fold))            # ( IV)

            hnew = -fnew + max(beta,0)*hold

        cnt_lnsrch = 0
        alpha_init = min([1.0, 0.1/max(np.abs(hnew)),max([-mnew/(c*(hnew@fnew)),1e-10])])


        alpha = 1/r*alpha_init
        mx = 10^100
        sx = s.copy()
        ux = unew.copy()

        print("\nSTARTING LINESEARCH..")
        while mx > mnew + c*alpha*(hnew@fnew) and alpha>1e-25:
            alpha *= r
            print("alpha =",alpha)
            sx = s+alpha*hnew
            mx, ux = mNM(sx,ux,fr,actives,sh,Circ, k_op,)

            cnt_lnsrch += 1

        s = sx.copy()
        unew = ux.copy()
        mnew = mx

        cnt += 1
        tot_lnsrch += cnt_lnsrch

        if ((mold-mnew)<=tol*abs(mold) or cnt%nreset==0) and steepest_descent_counter<3:
            if norm(fnew)>1:
                reset = True
                steepest_descent_counter += 1


    return s,cnt,unew




def numericalDerivative(func,x,method,h=1e-8):
    n = len(x)
    dF = np.zeros(n)

    if method in ['forward', 'backward']:
        f0 = func(x)

    for i in range(n):
        xTemp = x.copy()
        if method == 'central':
            xTemp[i] = x[i] + h
            f1 = func(xTemp)
            xTemp[i] = x[i] - h
            f2 = func(xTemp)
            dF[i] = (f1 - f2) / (2*h)
        elif method == 'forward':
            f1=f0
            xTemp[i] = x[i] + h
            f2 = func(xTemp)
            dF[i] = (f2 - f1) / h
        elif method == 'backward':
            xTemp[i] = x[i] - h
            f1= func(xTemp)
            f2 = f0
            dF[i] = (f2 - f1) / h
            
    return dF

@lru_cache(maxsize=20)
def solve_for_a_given_s(u0,fr,Circ,s,idx_act,cache_key):
    s = np.array(s)
    # u_iter = NR(u0,fr,dms,ddms,args=(Circ,s,idx_act))
    u_iter = NR(u0,list(fr),mfk_with_constr,args=(X,C,fr,actives,mu,ratio,k_pen,Circ,slaves,dofs,s,sh,1),verbose=0)
    # plotMS(u_iter,X,C, ax=None, undef=False, show=True, save=False,sline = None)

    if type(u_iter) != np.ndarray:
        print("NR failed, trying Newton-CG...")
        # u_iter = minimize(m_rstr,u0,args=(Circ,s,idx_act),method='Newton-CG',jac=f_rstr,hess=k_rstr,options={'disp':False}).x
        u_iter = minimize(mfk_with_constr,u0,args=(X,C,fr,actives,mu,ratio,k_pen,Circ,slaves,dofs,s,sh,0),method='Newton-CG',jac=True,hess=None,options={'disp':False}).x
    return u_iter

def solve_for_a_given_s_wrapper(u0,fr,Circ,s,idx_act,tol=1e-5):
    cache_key = (tuple(u0),tuple(fr),((Circ[0][0],Circ[0][1]),Circ[1]),tuple(idx_act))
    return solve_for_a_given_s(tuple(u0),tuple(fr),((Circ[0][0],Circ[0][1]),Circ[1]),tuple(s),tuple(idx_act),cache_key)

def u_of_s(s,Circ,u0,idx_act):
    return solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)


def mgt2(s,u0,idx_act,sh,Circ,tracePoints=True):
    if tracePoints: PairsUsed.append(s)

    Cxy, Cr = Circ
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)

    mOBJ,cnt_act = 0.0 ,0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act]
        cnt_act+=1
        nor = get_nor(si)
        nor_h = get_nor(sh[idx])
        xc = np.array(Cxy)+Cr*nor
        xch= Cxy+Cr*nor_h

        mOBJ +=  (xc-xch)@(xc-xch)

    print("s : ",s,"\tm : ",mOBJ)
    return mOBJ

def mgt2c1(s,u0,idx_act,sh,Circ,kc1,tracePoints=True,):
    if tracePoints: PairsUsed.append(s)

    Cxy, Cr = Circ
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)

    mgt, mc1, cnt_act = 0.0 ,0.0 ,0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act]
        cnt_act+=1
        nor = get_nor(si)
        nor_h = get_nor(sh[idx])
        xc = np.array(Cxy)+Cr*nor
        xch= Cxy+Cr*nor_h

        mgt += (xc-xch)@(xc-xch)

        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Ft = F - (F@nor)*nor

        if np.allclose(xch-xc,0.0):
            Tau = np.zeros(3)
        else:
            Taup = (xch-xc)-((xch-xc)@nor)*nor     # projection vector pointing TOWARDS the hook
            Tau = Taup/norm(Taup)     # normalized vector on the 'nor'-plane pointing towards the hook
        
        mc1 += kc1*(norm(Ft)-Ft@Tau)**2

    return mgt+mc1

# equality  
def c1(s,u0,idx_act,sh,Circ,kc1,eval=None):
    if eval is None:
        u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    else: u = eval
    Cxy, Cr = Circ
    c1 = np.zeros(len(s))
    cnt_act = 0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act]
        nor = get_nor(si)
        nor_h = get_nor(sh[idx])
        xc = np.array(Cxy)+Cr*nor
        xch= np.array(Cxy)+Cr*nor_h
   
        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Ft = F - (F@nor)*nor

        if np.allclose(xch-xc,0.0):
            Tau = np.zeros(2)
        else:
            Taup = (xch-xc)-((xch-xc)@nor)*nor     # projection vector pointing TOWARDS the hook
            Tau = Taup/norm(Taup)     # normalized vector on the 'nor'-plane pointing towards the hook

        c1[cnt_act] =  norm(Ft)-Ft@Tau if not np.allclose(xch-xc,0.0) else 0.0
        # c1[cnt_act] =  Ft@Tau if not np.allclose(xch-xc,0.0) else 0.0
        cnt_act += 1        # counter must continue even if later happens that gn>0

    print("c1:",c1)

    return c1     #stric 'eq' (equality) constraint. Complicates the computation

# inequality    Ft <= mu*Fn
def c2(s,u0,idx_act,sh,Circ,kc1,eval=None):
    
    if eval is None:
        u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    else: u = eval
    c2 = np.zeros(len(s),dtype=np.float128)
    cnt_act = 0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act].copy()
        nor = get_nor(si)

        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Fn = (F@nor)*nor
        Ft = F - Fn

        c2[cnt_act] = mu*norm(Fn) - norm(Ft) # must be positive
        cnt_act += 1        # counter must continue even if later happens that gn>0
    print("c2:",c2)
    return c2


def m_rstr(u,Circ,s,idx_act, show=False):
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    mN, mT = 0.0 , 0.0
    cnt_act = 0
    for idx in idx_act:
        xsi = xs[idx].copy()
        gn = norm(xsi-Cxy)-Cr
        mN += 1/2*kn*gn**2
        if sh[idx] == -1: continue
        si = s[cnt_act].copy()
        nor = get_nor(si)
        xc = np.array(Cxy)+Cr*nor
        dxsi = xsi-xc
        dxs_proj = dxsi - (dxsi@nor)*nor
        mT += 1/2*kt*norm(dxs_proj)**2
        cnt_act+=1
    m = ms(u,fr,X,C,EA) + mN + mT
    if show: print("here in m (r) \t m =",m)

    return m

def f_rstr(u,Circ,s,idx_act, show=False):

    Cxy, Cr = Circ
    f = np.zeros_like(u,dtype=np.float64)
    f[fr] = dms(u,fr,X,C,EA,dofs)[fr]
    # f = dms(u,X,C,EA,dofs)
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in idx_act:
        xsi = xs[idx].copy()
        gn = norm(xsi-Cxy)-Cr
        dgndu = (xsi-Cxy)/norm(xsi-Cxy)
        f[dofs[slaves[idx]]] += kn*gn*dgndu                  # += fN
        if sh[idx] == -1: continue
        si = s[cnt_act].copy()
        nor = get_nor(si)
        xc = np.array(Cxy)+Cr*nor
        dxsi = xsi-xc
        dxs_proj = dxsi - (dxsi@nor)*nor
        
        f[dofs[slaves[idx]]] += kt*dxs_proj@(np.eye(2)-np.outer(nor,nor))  # += fT
        cnt_act+=1

    if show: print("here in Fi (r) \tRES:",norm(f))
        
    return f

def k_rstr(u,Circ,s,idx_act, show=False):

    Cxy, Cr = Circ
    kij = np.zeros((len(u),len(u)),dtype=np.float32)
    # kij[np.ix_(fr,fr)] = ddms(u,X,C,EA,dofs)[np.ix_(fr,fr)]
    kij = ddms(u,fr,X,C,EA,dofs)
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in idx_act:
        dofi = dofs[slaves[idx]]
        xsi = xs[idx].copy()
        dsc = norm(xsi-Cxy)
        gn = dsc -Cr
        dgndu = (xsi-Cxy)/dsc
        d2gndu2 = (np.eye(2) - np.outer(dgndu,dgndu))/dsc
        kij[np.ix_(dofi,dofi)] += kn*(np.outer(dgndu,dgndu)+gn*d2gndu2)                   # += fN
        if sh[idx] == -1 : continue
        si = s[cnt_act].copy()
        nor = get_nor(si)
        m1 = np.eye(2)-np.outer(nor,nor)

        kij[np.ix_(dofi,dofi)] += kt*(m1@m1)  # += kij
        cnt_act+=1
    
    if show: print("here in Kij (r)")
    
    return kij

def mfk_with_constr(u,X,C,fr,actives,mu,ratio,k_pen,Circ,slaves,dofs,s,sh,rtn=0):
    """
    rtn =   0: returns m,f
            1: returns f,k
            2: returns m,f,k
    """
    x=X+u.reshape(-1,2)
    m=0
    f=np.zeros(ndofs)
    if rtn>0:
        k=np.zeros((len(u),len(u)),dtype=np.float64)
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])

        if L>=L0:
            EAi = EA
        else:
            EAi = ratio*EA

        m+=EAi*L0*(0.5*(log(L/L0))**2)
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        f[np.ix_(dofij)]+=(EAi*L0/L)*log(L/L0)*dLdu
        if rtn>0:
            dadu = np.array([[ 1.0, 0.0,-1.0, 0.0],
                            [ 0.0, 1.0, 0.0,-1.0],
                            [-1.0, 0.0, 1.0, 0.0],
                            [ 0.0,-1.0, 0.0, 1.0]])
            d2Ldu2 = dadu/L - np.outer(a,a)/(L**3)
            k[np.ix_(dofij,dofij)] += EAi*L0*((1-log(L/L0))/L**2*np.outer(dLdu,dLdu)+log(L/L0)/L*d2Ldu2)

    diri = np.delete(np.arange(ndofs), fr)
    f[diri] = 0

    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    idxs_fless = [idx for idx in range(ns) if (idx in actives and sh[idx]==-1)]
    for idx in idxs_fless:
        xsi = xs[idx]
        dofi = dofs[slaves[idx]]

        g = norm(xsi-Cxy)-Cr
        dgdu = (xsi-Cxy)/(g+Cr)
        if rtn>0:
            out_dgdu = np.outer(dgdu,dgdu)
            d2gdu2=(np.eye(2) - out_dgdu)/(g+Cr)
            k[np.ix_(dofi,dofi)] += k_pen*(out_dgdu+g*d2gdu2)
        
        m += 0.5*k_pen*g**2
        f[np.ix_(dofi)] += k_pen*g*dgdu


    return [(m,f),(f,k),(m,f,k)][rtn]



def fiT(u,Circ,s,idx_act):
    Cxy, Cr = Circ
    f = np.zeros_like(u,dtype=np.float64)
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in idx_act:
        if sh[idx] is None: continue
        xsi = xs[idx]
        si = s[cnt_act]
        nor = get_nor(si)
        xc = np.array(Cxy)+Cr*nor
        dxsi = xsi-xc
        dxs_proj = dxsi - (dxsi@nor)*nor
        
        f[dofs[slaves[idx]]] += kt*dxs_proj@(np.eye(2)-np.outer(nor,nor))  # += fT
        cnt_act+=1
        
    return f

def kijT(u,Circ,s,idx_act):
    kij = np.zeros((len(u),len(u)),dtype=np.float64)
    cnt_act = 0
    for idx in idx_act:
        dofi = dofs[slaves[idx]]
        if sh[idx] is None: continue
        si = s[cnt_act]
        nor = get_nor(si)
        m1 = np.eye(2)-np.outer(nor,nor)

        kij[np.ix_(dofi,dofi)] += kt*(m1@m1)  # += kij
        cnt_act+=1
    
    return kij

def fiN(u,Circ,s,idx_act):
    Cxy, Cr = Circ
    f = np.zeros_like(u,dtype=np.float64)
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    for idx in idx_act:
        xsi = xs[idx]
        gn = norm(xsi-Cxy)-Cr
        dgndu = (xsi-Cxy)/norm(xsi-Cxy)
        f[dofs[slaves[idx]]] += kn*gn*dgndu                  # += fN
        
    return f

def kijN(u,Circ,s,idx_act):
    Cxy, Cr = Circ
    kij = np.zeros((len(u),len(u)),dtype=np.float64)
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    for idx in idx_act:
        dofi = dofs[slaves[idx]]
        xsi = xs[idx]
        dsc = norm(xsi-Cxy)
        gn = dsc -Cr
        dgndu = (xsi-Cxy)/dsc
        d2gndu2 = (np.eye(2) - np.outer(dgndu,dgndu))/dsc
        kij[np.ix_(dofi,dofi)] += kn*(np.outer(dgndu,dgndu)+gn*d2gndu2)                   # += fN
    
    return kij


def get_nor(s):
    return -np.array([ cos(s*pi), sin(s*pi)],dtype=np.float64)

def get_dnords(s):
    return pi*np.array([ sin(s*pi), -cos(s*pi)],dtype=np.float64)

def get_dmds(s,u0,idx_act,sh,Circ,kc1):
    Cxy,Cr = Circ
    dmds,cnt_act = np.zeros_like(s) ,0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act]
        xc = np.array(Cxy)+Cr*get_nor(si)
        dxcds = Cr*get_dnords(si)
        nor_h = get_nor(sh[idx])
        xch= Cxy+Cr*nor_h
        dmds[cnt_act]=2*(xc-xch)@dxcds
        cnt_act+=1

    return dmds

def get_duds(s,u,idx_act,sh,Circ,kc1):
    Cxy,Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    KK = np.zeros((len(u),len(u)))
    KK[np.ix_(fr,fr)] = k_rstr(u.copy(),Circ,s.copy(),idx_act).copy()[np.ix_(fr,fr)]
    # KK= approx_fprime(u,f_rstr,1e-10,Circ,s,idx_act)     # this works => problem in 'k_rstr'
    dfds,cnt_act = np.zeros((KK.shape[0],len(s))) ,0
    duds = np.zeros((ndofs,len(s)))
    # dudsa= np.zeros((ndofs,len(s)))
    for idx in idx_act:
        if sh[idx] is None: continue
        xsi = xs[idx]
        si = s[cnt_act]
        nor = get_nor(si)
        dnds = get_dnords(si)
        xc = np.array(Cxy)+Cr*nor
        dxcds = Cr*dnds
        dsi = xsi - xc
        dsip = dsi - (dsi@nor)*nor
        ddsipddsi = np.eye(2)-np.outer(nor,nor)
        NI = np.multiply.outer(nor,np.eye(2))
        # dnndn = NI + NI.swapaxes(1,0)
        dnndn = NI + NI.swapaxes(0,2)
        ddsipdn = -np.tensordot(dsi,dnndn,axes=[0,0])
        # ddsipdn = -(dsi@nor)*np.eye(2) - np.outer(nor,dsi)
        # ddsipdn = -(dsi@nor)*np.eye(2) - np.outer(dsi,nor)
        ddsids = -dxcds
        ddsipds = ddsipddsi@ddsids + ddsipdn@dnds

        dofi = dofs[slaves[idx]]
        # dfds[dofi,cnt_act] += kt*(np.eye(2)-np.outer(nor,nor))@ddsipds - kt*dsip@(dnndn@dnds)
        dfds[dofi,cnt_act] += kt*(np.eye(2)-np.outer(nor,nor))@ddsipds - kt*np.tensordot(dsip,np.tensordot(dnndn,dnds,axes=[2,0]),axes=[0,1])
        cnt_act+=1



    duds[fr] = np.linalg.solve(KK[np.ix_(fr,fr)],-dfds[fr,:])
    # dudsa[fr]= np.linalg.solve(KKA[np.ix_(fr,fr)],-dfds[fr,:])
    
    # if t >=0.16: set_trace()

    
    # set_trace()
    return duds


def dc1ds(s,u0,idx_act,sh,Circ,kc1):
    Cxy,Cr = Circ
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    dc1ds,cnt_act = np.zeros((len(s),len(s))) ,0
    fint = dms(u,X,C,EA,dofs)
    kint = ddms(u,X,C,EA,dofs)
    duds = get_duds(s,u,idx_act,sh,Circ,kc1)
    for idx in idx_act:
        if sh[idx] is None: continue
        dofi = dofs[slaves[idx]]
        si = s[cnt_act]
        nor = get_nor(si)
        xc = np.array(Cxy)+Cr*nor
        nor_h = get_nor(sh[idx])
        xch= Cxy+Cr*nor_h
        
        fnode = fint[dofi]
        ft = fnode - (fnode@nor)*nor
        dsh = (xch-xc)
        taup = dsh - (dsh@nor)*nor
        tau = taup/norm(taup)

        dxcds = Cr*get_dnords(si)
        dnords = get_dnords(si)
        dfdu = kint[dofi]
        dftdf = np.eye(2)-np.outer(nor,nor)
        dftdn = -(fnode@nor)*np.eye(2)-np.outer(fnode,nor)
        dtaupdxc = -dftdf
        dtaupdn = -(dsh@nor)*np.eye(2)-np.outer(dsh,nor)

        dtaudtaup = np.eye(2)/norm(taup) - np.outer(taup,taup)/norm(taup)**3

        dc1dft = tau - ft/norm(ft)
        # dc1dft = tau    # for when c1 = Ft·tau > 0
        dc1dtau = ft

        dc1ds[cnt_act,cnt_act] += dc1dft@dftdn@dnords + dc1dtau@dtaudtaup@(dtaupdn@dnords+dtaupdxc@dxcds)
        dc1ds[cnt_act] += dc1dft@dftdf@dfdu@duds

        cnt_act+=1

    return dc1ds

def dc2ds_pp(s,u0,idx_act,sh,Circ,kc1):
    dc2ds = np.zeros((len(s),len(s)))
    for ic in range(len(s)):
        dc2ds[ic] = approx_fprime(s,c2i,1e-10,u0,idx_act,sh,Circ,ic)
    return dc2ds


def dc2ds(s,u0,idx_act,sh,Circ,kc1):
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    dc2ds,cnt_act = np.zeros((len(s),len(s))) ,0
    fint = dms(u,X,C,EA,dofs)
    kint = ddms(u,X,C,EA,dofs)
    duds = get_duds(s,u,idx_act,sh,Circ,kc1)
    # duds = approx_fprime(s,u_of_s,1e-10,Circ,u0,idx_act)      # thisa works => problem in 'get_duds'
    for idx in idx_act:
        if sh[idx] is None: continue
        dofi = dofs[slaves[idx]]
        # imposed 's'
        si = s[cnt_act]
        nor = get_nor(si)

        fnode = fint[dofi]
        fn = (fnode@nor)*nor
        ft = fnode - fn

        dnords = get_dnords(si)
        dfndn = (fnode@nor)*np.eye(2)+np.outer(fnode,nor)
        dfndf = np.outer(nor,nor)
        dftdf = np.eye(2)
        dftdfn = -np.eye(2)
        dc2dfn = mu*fn/norm(fn)
        dc2dft = -ft/norm(ft)

        dfdu = kint[dofi]

        dc2ds[cnt_act,cnt_act] += (dc2dfn + dc2dft@dftdfn)@dfndn@dnords
        dc2ds[cnt_act] += (dc2dfn@dfndf+dc2dft@(dftdf+dftdfn@dfndf))@dfdu@duds

        cnt_act+=1

    return dc2ds

def verdict_like3d(u,Circ,idx_act,sh,s):
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    act, shnew, Redo = [], [None]*ns, False
    fint = dms(u,X,C,EA,dofs)
    for idx, xsi in enumerate(xs):

        gn = norm(xsi-np.array(Cxy))-Cr
        
        if gn>=0:           # slave is out
            if idx in idx_act:  #... but was in
                print("node",slaves[idx],"exited")
                Redo = True

        else:
            si = s[idx]
            nor = get_nor(si)

            if idx in idx_act:          # was active
                if sh[idx] is None:     # first slide
                    print("node",slaves[idx],"first slide")
                    shnew[idx] = si
                else:
                    fnode = fint[dofs[slaves[idx]]]
                    f_proj = fnode - (fnode@nor)*nor
                    
                    if norm(f_proj) <= abs(mu*fnode@nor):    # elastic case
                        shnew[idx] = sh[idx]
                        print("node",slaves[idx],"continues elastic")
                    else:
                        # shnew[idx] = 0.5*(si+sh[idx])
                        shnew[idx] = si
                        print("node",slaves[idx],"continues plastic")
                    print("fnode",fnode)

            else:
                print("node",slaves[idx],"just entered")
                Redo = True
            
            act.append(idx)
    
    return act, shnew, Redo

def checkContact(u,Circ,actives,sh,s,s_pre, X,dofs,slaves,C,ns,mu,ratio):
    Cxy, Cr = Circ
    Fn_eps = 1e-4
    gn_eps = 1e-5

    xs = X[slaves]+u[dofs[slaves]]
    act = []
    exited = []
    shnew = -1*np.ones(ns)
    Redo = False
    fint = dms(u,fr,X,C,EA,dofs)

    for idx in range(ns):
        if idx not in actives:
            xsi = xs[idx]
            gn = norm(xsi-Cxy) - Cr

            if gn < gn_eps:
                print("node",slaves[idx],"just entered. -> REDO")
                Redo = True
            else:
                continue
        else:
            si = s[idx]          
            nor = get_nor(si)
                        
            fnode = fint[dofs[slaves[idx]]]
            Fn = -fnode@nor

            if Fn<-Fn_eps:
                if sh[idx]==-1:
                    print("node",slaves[idx],"exceeded Fn_lim, Fn=",Fn," -> REDO")
                    Redo = True
                    sh[idx] = s_pre[idx]
                else:
                    print("node",slaves[idx],"hooked")
                    f_proj = fnode - (fnode@nor)*nor
                    shnew[idx] = si
            elif Fn>Fn_eps:
                print("node",slaves[idx],"exited, Fn=",Fn," -> REDO")
                Redo = True
                exited.append(idx)
                continue

            else:
                if sh[idx]!=-1:
                    print("node",slaves[idx],"now frictionles, Fn=",Fn," -> REDO")
                    sh[idx] = -1
                    Redo = True
                else:
                    print("node",slaves[idx],"frictionles")
                
        act.append(idx)

    if Redo:
        shnew = sh.copy()
        shnew[exited] = -1*np.ones(len(exited))

    return act,shnew,Redo


def getProjs(xs,Circ):
    if len(xs.shape)<2:     # if only one point...
        Cxy = Circ[0]
        dxyi = xs-Cxy
        si=atan(dxyi[1]/dxyi[0])/pi
        return si if si>=0 else 1+si
    else:                   # multiple points...
        s = np.zeros(len(xs))
        for idx, xsi in enumerate(xs):
            s[idx] = getProjs(xsi,Circ)
        return s

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
        csvwriter.writerow(['time','nfev','nact','nhooked', 'trials', 'try_last'])

    with open(filename2, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
        csvwriter = csv.writer(csvfile)
        row = [time,eval('nevals[-1]'),len(idx_act),len(idx_hooked),trials]
        try: row.append(NM.nfev)
        except NameError: row.append(trials)
        csvwriter.writerow(row)



X, C = Truss(1,1,0.75,0.75)
dim = len(X[0])
dofs = np.array(range(2*len(X))).reshape(-1,2)
ndofs = len(dofs.ravel())
X = Translate(X,[-0.75,-0.75])
EA = 1.0
ratio = 1.0     # resist_compr/resist_tens


slaves = SelectFlatSide(X,"+y")
base = SelectFlatSide(X,"-y")
dofs_base_x = dofs[base][:,0]
dofs_base_y = dofs[base][:,1]
ns = len(slaves)
k_pen = 1000
mu = 0.5

# Rigid Circle
Cxy = np.array([1.0 , 3.0])
Cr  = 3.1
Circ = [Cxy,Cr]


# import pickle
# pickle.dump({'pos':X,'connect':C,"EA":1.0,'mu':0.5,"ID_slaves":slaves,"ID_base":base,"Cir_pos":Cxy,"Cir_R":Cr,'dirichlet_base':(4.0,0.0)},open("Model.dat","wb"))
# set_trace()

#Step1  (Right)
bc1 = [dofs_base_x,"dir",4.0, 0.0,1.0]
bc2 = [dofs_base_y,"dir",0.0, 0.0,1.0]
BCs = [bc1,bc2]

kn = kt = 1e3
tol = 1e-8
nincr = 50
t0, tf = 0, 1
dt = (tf-t0)/nincr
tpre = t0

TT = []
iters_tot = []
trials = []
dX, FN, FFN, FT, RES,SS1,SS2= [],[],[],[],[],[],[]
NEVALS = []
folder = "Minimization"+"_"+strftime("%y%m%d-%H%M")
if not os.path.exists(folder):
    os.mkdir(folder)


Fbase = np.zeros((nincr,2),dtype=float)
f = np.zeros(ndofs)
u = np.zeros(ndofs)

# Initiallize contact variables
xs = X[slaves]+u[dofs[slaves]]
s = getProjs(xs,Circ)
sh = -1*np.ones_like(s)
actives = []

for incr in range(nincr):
    Redo = True
    iter_out = 0
    trials.append(1)
    iters_tot.append(0)
    t = t0 + (incr/nincr)*(tf-t0)

    print("------------")
    print("Increment:",incr)
    print("------------")
    while Redo:
        iter_out += 1
        du,df,di = ApplyBCs(tpre,t,BCs,ndofs)
        f += df
        u += du

        fr = np.delete(np.arange(ndofs), di)       # free DOFs 'b'

        idxs_hooked = [idx for idx in range(ns) if sh[idx]!=-1]

        s_pre = s.copy()

        if len(idxs_hooked)>0:
            s_opt = s[idxs_hooked] + 0.5*ds[idxs_hooked]
            breakpoint()
            s_opt,iters,u = CG_mf_constr(u,X,C,fr,actives,mu,ratio,k_pen,Circ,slaves,dofs,s_opt,sh,1e3,1e-4,1e-8)
            s[idxs_hooked] = s_opt
            iters_tot[-1] += iters

        else:
            iters = 1
            iters_tot[-1] += 1
                
            u = solve_for_a_given_s_wrapper(u,fr,Circ,s,actives)

        unused = [i for i in range(ns) if i not in idxs_hooked]
        xs = X[slaves]+u[dofs[slaves]]
        s[unused]=getProjs(xs[unused],Circ)

        actives, sh, Redo = checkContact(u,Circ,actives,sh,s,s_pre, X,dofs,slaves,C,ns,mu,ratio)

        idxs_hooked = [idx for idx in range(ns) if sh[idx]!=-1]

        if Redo:
            trials[-1] += 1
            ds = np.zeros_like(s)
            print("redoing...")
            # t -= dt
        else:
            tpre = t
            # t += dt
            ds = s-s_pre

            plotMS(u,X,C, ax=None, undef=False, show=False, save=True,sline = None)

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)



# # for idx, mu in enumerate(MUs):
# for idx in range(NSIM):
#     ax1.plot(TT,FN[idx],label=meth[idx])
#     ax2.plot(TT,FT[idx],label=meth[idx])
#     ax3.plot(TT,NEVALS[idx],label=meth[idx])
#     ax4.plot(TT,RES[idx],label=meth[idx])
# ax1.legend(),ax2.legend(),ax3.legend(),ax4.legend()
# ax1.set_xlabel("time")
# ax1.set_ylabel("Fx")
# ax2.set_xlabel("time")
# ax2.set_ylabel("Fy")
# ax3.set_xlabel("time")
# ax3.set_ylabel("# of evals")
# ax4.set_xlabel("time")
# ax4.set_ylabel("log10(res)")
# # plt.show()
# plt.savefig(folder+"/results.png")

# print('# of f_evals: (counting redone increments):')
# for idx in range(NSIM):
#     print(meth[idx],": ",sum(NEVALS[idx]))


print("--FINISHED--")
