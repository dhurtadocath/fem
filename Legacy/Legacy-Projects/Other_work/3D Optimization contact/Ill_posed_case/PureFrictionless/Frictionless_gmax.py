from pdb import set_trace
# from re import X
import numpy as np
from numpy.linalg import norm
from math import sqrt, sin, cos, atan,atan2, acos, pi, log10,log
from scipy.optimize import minimize,LinearConstraint, newton, Bounds, HessianUpdateStrategy,approx_fprime
import os, sys, pickle, csv
from functools import lru_cache
from time import strftime
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

def plotTruss(X,C, show_nodes=False,show_id=False,bar_types=None,showtypes=-1, ax = None):
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
        ax.plot(XX[:,0],XX[:,1],XX[:,2], c="blue",linewidth=0.5)
        if show_id: 
            alp=  0.3
            avg = XX[0]*(1-alp) + XX[1]*alp
            ax.text(avg[0],avg[1],avg[2],i,c="blue")
    if show_nodes:
        actives = [slaves[slid] for slid in idx_act]
        for i, x in enumerate(X):
            if i not in slaves: continue
            color = "red" if  i in actives else "blue" 
            ax.scatter(x[0],x[1],x[2],c=color,s=5)
            if show_id: ax.text(x[0],x[1],x[2],i)

    if show:  plt.show()

def plotSphere(Circ,ax=None):
    if ax is None:        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

    Cxy, Cr = Circ
    a = np.linspace(0,pi,50)
    b = np.linspace(pi/2,3*pi/2,50)
    x = Cr*np.outer(np.cos(a),np.sin(b)) + Cxy[0]
    y = Cr*np.outer(np.sin(a),np.sin(b)) + Cxy[1]
    z = Cr*np.outer(np.ones(np.size(a)),np.cos(b)) + Cxy[2]

    ax.plot_surface(x,y,z,linewidth=0.5, color=(0.5,0.5,0.5,0.5))
    if ax is None:  plt.show()

def plotMS(u,X,C, ax=None, undef=False, show=False, save=None,sline = None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

    plotTruss(X+u,C,ax=ax,bar_types=btypes,show_nodes=True,show_id=False,showtypes=0)
    plotSphere(Circ,ax=ax)

    ax.set_xlim([-1.5,3.5])
    ax.set_ylim([-2.5,2.5])
    ax.set_zlim([-1.5,2.0])

    if show: plt.show()

    if save is not None:
        if not os.path.exists(folder+"/plots"):
            os.mkdir(folder+"/plots")

        plt.savefig(folder+"/plots//img"+str(tt)+".png")
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


def ms(u,X,C,EA,dofs,Circ,idx_act):
    
    x=X+u.reshape(-1,3)
    ms=0
    for (i,j) in C:
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        dL = max(L-L0,0.0)
        if etype=="spring":
            ms+=(EA*dL**2)/(2*L0) # spring
        elif etype=="truss":
            ms+=EA*L0*(0.5*(log((dL+L0)/L0))**2)    # ABQ's truss
    return ms

def dms(u,X,C,EA,dofs,Circ,idx_act):
    x=X+u.reshape(-1,3)
    dms=np.zeros(ndofs)
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        dL = max(L-L0,0.0)
        if etype=="spring":
            dms[np.ix_(dofij)]+=EA*dL/L0*dLdu # spring
        elif etype=="truss":
            dms[np.ix_(dofij)]+=(EA*L0/L)*log((dL+L0)/L0)*dLdu    # ABQ's truss

    dms[di] = 0.0
    return dms

def ddms(u,X,C,EA,dofs,Circ,idx_act):
    x=X+u.reshape(-1,3)
    ddms=np.zeros((ndofs,ndofs))
    dadu = np.array([[ 1.0, 0.0, 0.0,-1.0, 0.0, 0.0],
                     [ 0.0, 1.0, 0.0, 0.0,-1.0, 0.0],
                     [ 0.0, 0.0, 1.0, 0.0, 0.0,-1.0],
                     [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                     [ 0.0,-1.0, 0.0, 0.0, 1.0, 0.0],
                     [ 0.0, 0.0,-1.0, 0.0, 0.0, 1.0]])
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dL = max(L-L0,0.0)
        dLdu = a/L
        d2Ldu2 = dadu/L - np.outer(a,a)/(L**3)
        if etype=="spring":
            ddms[np.ix_(dofij,dofij)] += EA*(np.outer(dLdu,dLdu)+dL*d2Ldu2)/L0    # spring
        elif etype=="truss":
            ddms[np.ix_(dofij,dofij)] += EA*L0*((1-log((dL+L0)/L0))/L**2*np.outer(dLdu,dLdu)+log((dL+L0)/L0)/L*d2Ldu2) # ABQ's truss
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
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    s=s.reshape(-1,2)
    Cxy, Cr = Circ
    mOBJ,cnt_act = 0.0 ,0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act]
        cnt_act+=1
        nor = get_nor(si)
        nor_h = get_nor(sh[idx])
        xc = np.array(Cxy)+Cr*nor
        xch= Cxy+Cr*nor_h
        gt = norm(xc-xch)

        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Fn = (F@nor)*nor
        Ft = F-Fn
        
        # Tau = (xch-xc)/norm(xc-xch)     # vector pointing TOWARDS the hook
        dsh = xch-xc
        Taup = dsh - (dsh@nor)*nor
        Tau = Taup/norm(Taup)

        mOBJ +=  gt**2 + 1/2*k*(Ft@Tau-norm(Ft))**2 + 1/2*k*((mu*norm(Fn)-norm(Ft))**2)*(norm(Ft)>mu*norm(Fn))

    print("u(",s,") = \t", "\tm =",mOBJ)
    # print("Fn =",Fn,"\tFt =",Ft)
    return log10(mOBJ+1)
    # return mOBJ

def NR(u0,fint,K,args=(),tol=1e-10,verbose=0):
    res, cnt = 10*5, 0
    u = u0

    if type(u) is tuple:
        u = np.array(u)
    u_pre = u.copy()
    f = fint(u,*args)
    while res>tol:
        df = K(u_pre,*args)

        dua = u[di] - u_pre[di]      # difference due to dirichlet BCs applied
        Kba=df[np.ix_(fr,di)]
        Kbb=df[np.ix_(fr,fr)]
        finb=f[fr]
                
        dub =np.linalg.solve(Kbb,-finb-Kba.dot(dua))
        u[fr] += dub
        f = fint(u,*args)

        u_pre = u

        cnt += 1
        res = norm(f)

        if cnt>10:
            return False

    if verbose: print("NR performed",cnt,"iterations")
    return u


@lru_cache(maxsize=20)
def solve_for_a_given_s(u0,Circ,s,idx_act,cache_key):
    s = np.array(s)
    # u_iter = NR(u0,f_rstr,k_rstr,args=(Circ,s.reshape(-1,2),idx_act))
    u_iter = minimize(m_rstr,u0,args=(Circ,s.reshape(-1,2),idx_act),method='Newton-CG',jac=f_rstr,hess=k_rstr,tol=1e-7,options={'disp':False}).x
    # if type(u_iter) != np.ndarray:
    #     print("NR failed, trying Newton-CG...")
    #     u_iter = minimize(m_rstr,u0,args=(Circ,s.reshape(-1,2),idx_act),method='Newton-CG',jac=f_rstr,hess=k_rstr,options={'disp':False}).x
    return u_iter

def solve_for_a_given_s_wrapper(u0,Circ,s,idx_act):
    cache_key = (tuple(u0),((Circ[0][0],Circ[0][1],Circ[0][2]),Circ[1]),tuple(idx_act))
    return solve_for_a_given_s(tuple(u0),((Circ[0][0],Circ[0][1],Circ[0][2]),Circ[1]),tuple(s.ravel()),tuple(idx_act),cache_key)

def u_of_s(s,Circ,u0,idx_act):
    return solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)


def mgt2(s,u0,idx_act,sh,Circ,kc1,tracePoints=True):
    if tracePoints: PairsUsed.append(s)

    Cxy, Cr = Circ
    s = s.reshape(-1,2)
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

    print("s : ",s.ravel(),"\tm : ",mOBJ)
    return mOBJ

def mgt2c1(s,u0,idx_act,sh,Circ, kc1, tracePoints=True):
    if tracePoints: PairsUsed.append(s)

    Cxy, Cr = Circ
    s = s.reshape(-1,2)
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    mgt,mc1,cnt_act = 0.0, 0.0 ,0
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
def c1(s,u0,idx_act,sh,Circ, kc1, eval = None):
    s = s.reshape(-1,2)
    if eval is None:
        u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    else: u = eval
    Cxy, Cr = Circ
    c1 = np.zeros(len(idx_act))
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
            Tau = np.zeros(3)
        else:
            Taup = (xch-xc)-((xch-xc)@nor)*nor     # projection vector pointing TOWARDS the hook
            Tau = Taup/norm(Taup)     # normalized vector on the 'nor'-plane pointing towards the hook

        c1[cnt_act] =  norm(Ft)-Ft@Tau if not np.allclose(xch-xc,0.0) else 0.0
        cnt_act += 1        # counter must continue even if later happens that gn>0

    return c1     #stric 'eq' (equality) constraint. Complicates the computation

# inequality    Ft <= mu*Fn
def c2(s,u0,idx_act,sh,Circ,kc1,eval= None):
    s=s.reshape(-1,2)
    if eval is None:
        u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    else: u = eval
    c2 = np.zeros(len(idx_act))
    cnt_act = 0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act]
        nor = get_nor(si)

        F = dms(u,X,C,EA,dofs)[dofs[slaves[idx]]]
        Fn = (F@nor)*nor
        Ft = F - Fn

        c2[cnt_act] = mu*norm(Fn) - norm(Ft) # must be positive
        cnt_act += 1        # counter must continue even if later happens that gn>0
    return c2


def m_rstr(u,Circ,s,idx_act, show=False):
    s = np.array(s).reshape(-1,2)
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    mN, mT = 0.0 , 0.0
    cnt_act = 0
    for idx in idx_act:
        xsi = xs[idx]
        gn = norm(xsi-Cxy)-Cr
        mN += 1/2*kn*gn**2
        if sh[idx] is None: continue
        si = s[cnt_act]
        cnt_act+=1
        nor = get_nor(si)
        xc = np.array(Cxy)+Cr*nor
        dxsi = xsi-xc
        dxs_proj = dxsi - (dxsi@nor)*nor
        mT += 1/2*kt*norm(dxs_proj)**2
    m = ms(u,X,C,EA) + mN + mT
    if show: print("here in m (r) \t m =",m)

    return m

def f_of_s(s,u,Circ,idx_act):
    return f_rstr(u,Circ,s,idx_act)

def f_rstr(u,Circ,s,idx_act, show=False):
    s = np.array(s).reshape(-1,2)
    Cxy, Cr = Circ
    f = np.zeros_like(u)
    f[fr] = dms(u,X,C,EA,dofs,Circ,idx_act)[fr]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in idx_act:
        xsi = xs[idx]
        gn = norm(xsi-Cxy)-Cr
        dgndu = (xsi-Cxy)/norm(xsi-Cxy)
        f[dofs[slaves[idx]]] += kn*gn*dgndu                  # += fN
        if sh[idx] is None: continue
        si = s[cnt_act]
        cnt_act+=1
        nor = get_nor(si)
        xc = np.array(Cxy)+Cr*nor
        dxsi = xsi-xc
        dxs_proj = dxsi - (dxsi@nor)*nor
        
        f[dofs[slaves[idx]]] += kt*dxs_proj@(np.eye(3)-np.outer(nor,nor))  # += fT

    if show: print("here in Fi (r) \tRES:",norm(f))
        
    return f

def k_rstr(u,Circ,s,idx_act, show=False):
    s = np.array(s).reshape(-1,2)
    Cxy, Cr = Circ
    kij = np.zeros((len(u),len(u)))
    kij[np.ix_(fr,fr)] = ddms(u,X,C,EA,dofs)[np.ix_(fr,fr)]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in idx_act:
        xsi = xs[idx]
        dsc = norm(xsi-Cxy)
        gn = dsc -Cr
        dgndu = (xsi-Cxy)/dsc
        d2gndu2 = (np.eye(3) - np.outer(dgndu,dgndu))/dsc
        kij[np.ix_(dofs[slaves[idx]],dofs[slaves[idx]])] += kn*(np.outer(dgndu,dgndu)+gn*d2gndu2)                   # += fN
        if sh[idx] is None: continue
        si = s[cnt_act]
        cnt_act+=1
        theta = pi*(1-si[1])
        phi = pi*si[0]
        nor = -np.array([cos(phi)*sin(theta), cos(theta),sin(theta)*sin(phi)])
        mij = np.eye(3)-np.outer(nor,nor)
        # set_trace()
        kij[np.ix_(dofs[slaves[idx]],dofs[slaves[idx]])] += kt*mij@mij  # += kij
    
    if show: print("here in Kij (r)")
    
    return kij

def m_free(u,X,C,EA,dofs,Circ,idx_act, show=False):
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    mN = 0.0
    for idx in idx_act:
        xsi = xs[idx]
        gn = norm(xsi-Cxy)-Cr
        mN += 1/2*kn*a_k[idx]*gn**2
    m = ms(u,X,C,EA,dofs,Circ,idx_act) + mN
    if show: print("here in m (r) \t m =",m)

    return m

def f_free(u,X,C,EA,dofs,Circ,idx_act, show=False):
    Cxy, Cr = Circ
    f = np.zeros_like(u)
    f[fr] = dms(u,X,C,EA,dofs,Circ,idx_act)[fr]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    for idx in idx_act:
        xsi = xs[idx]
        gn = norm(xsi-Cxy)-Cr
        dgndu = (xsi-Cxy)/norm(xsi-Cxy)
        f[dofs[slaves[idx]]] += kn*a_k[idx]*gn*dgndu                  # += fN

    if show: print("here in Fi (r) \tRES:",norm(f))
        
    return f

def k_free(u,X,C,EA,dofs,Circ,idx_act, show=False):
    Cxy, Cr = Circ
    kij = np.zeros((len(u),len(u)))
    kij[np.ix_(fr,fr)] = ddms(u,X,C,EA,dofs,Circ,idx_act)[np.ix_(fr,fr)]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    for idx in idx_act:
        xsi = xs[idx]
        dsc = norm(xsi-Cxy)
        gn = dsc -Cr
        dgndu = (xsi-Cxy)/dsc
        d2gndu2 = (np.eye(3) - np.outer(dgndu,dgndu))/dsc
        kij[np.ix_(dofs[slaves[idx]],dofs[slaves[idx]])] += kn*a_k[idx]*(np.outer(dgndu,dgndu)+gn*d2gndu2)                   # += fN
    
    if show: print("here in Kij (r)")
    
    return kij

def Cgn(u,X,C,EA,dofs,Circ,idx_act):
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    GN = np.zeros(len(idx_act))
    for ii, idx in enumerate(idx_act):
        xsi = xs[idx]
        GN[ii] = norm(xsi-Cxy)-Cr
    # print("GN:",GN)
    # if len(idx_act):set_trace()
    return GN

def dCgndu(u,X,C,EA,dofs,Circ,idx_act):
    Cxy, _ = Circ
    dGNdu = np.zeros((len(idx_act),len(u)))
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    for ii, idx in enumerate(idx_act):
        xsi = xs[idx]
        dGNdu[ii,dofs[slaves[idx]]]= (xsi-Cxy)/norm(xsi-Cxy)
    # print("dGNdu:",dGNdu)
    return dGNdu


def get_nor(s):
    theta = pi*(1-s[1])
    phi = pi*s[0]
    return -np.array([cos(phi)*sin(theta), cos(theta),sin(theta)*sin(phi)])

def get_dnords(s):
    theta = pi*(1-s[1])
    phi = pi*s[0]
    dndtheta = -np.array([cos(phi)*cos(theta), -sin(theta),cos(theta)*sin(phi)])
    dndphi   = -np.array([-sin(phi)*sin(theta),       0.0 ,sin(theta)*cos(phi)]) 
    return pi*np.array([dndphi,-dndtheta]).T

def get_dmds(s,u0,idx_act,sh,Circ,kc1):
    Cxy,Cr = Circ
    s = s.reshape(-1,2)
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


def get_duds(s,u0,idx_act,sh,Circ,kc1):
    Cxy,Cr = Circ
    s = s.reshape(-1,2)
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    KK = k_rstr(u,Circ,s,idx_act,)
    dfds,cnt_act = np.zeros((KK.shape[0],len(s),2)) ,0
    duds = np.zeros((ndofs,2*len(s)))
    for idx in idx_act:
        if sh[idx] is None: continue
        xsi = xs[idx]
        si = s[cnt_act]
        xc = np.array(Cxy)+Cr*get_nor(si)
        dxcds = Cr*get_dnords(si)
        dnords = get_dnords(si)
        nor = get_nor(si)
        dsi = xsi - xc
        dsip = dsi - (dsi@nor)*nor
        ddsipddsi = np.eye(3)-np.outer(nor,nor)
        # ddsipdn = (dsi@nor)*np.eye(3) - np.outer(nor,dsi)
        ddsipdn = (dsi@nor)*np.eye(3) - np.outer(dsi,nor)
        ddsids = -dxcds
        ddsipds = ddsipddsi@ddsids + ddsipdn@dnords
        NI = np.multiply.outer(nor,np.eye(3))
        dnndn = NI + NI.swapaxes(0,2)
        dofi = dofs[slaves[idx]]
        dfds[dofi,cnt_act,:] = kt*(np.eye(3)-np.outer(nor,nor))@ddsipds - kt*dsip@(dnndn@dnords)
        cnt_act+=1
    duds[fr] = np.linalg.solve(KK[np.ix_(fr,fr)],-dfds.reshape(KK.shape[0],2*len(s))[fr,:])
    
    return duds


def dc1ds(s,u0,idx_act,sh,Circ,kc1):
    Cxy,Cr = Circ
    s = s.reshape(-1,2)
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    dc1ds,cnt_act = np.zeros((len(s),2*len(s))) ,0
    fint = dms(u,X,C,EA,dofs)
    kint = ddms(u,X,C,EA,dofs)
    duds = get_duds(s,u0,idx_act,sh,Circ,kc1)
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
        dftdf = np.eye(3)-np.outer(nor,nor)
        dftdn = -(fnode@nor)*np.eye(3)-np.outer(fnode,nor)
        dtaupdxc = -dftdf
        dtaupdn = -(dsh@nor)*np.eye(3)-np.outer(dsh,nor)

        dtaudtaup = np.eye(3)/norm(taup) - np.outer(taup,taup)/norm(taup)**3

        dc1dft = ft/norm(ft) - tau
        dc1dtau = -ft

        dc1ds[cnt_act,2*cnt_act:2*cnt_act+2] += dc1dft@dftdn@dnords + dc1dtau@dtaudtaup@(dtaupdn@dnords+dtaupdxc@dxcds)
        dc1ds[cnt_act] += dc1dft@dftdf@dfdu@duds

        cnt_act+=1

    return dc1ds

def dc2ds(s,u0,idx_act,sh,Circ,kc1):
    s = s.reshape(-1,2)
    u = solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)
    dc2ds,cnt_act = np.zeros((len(s),2*len(s))) ,0
    fint = dms(u,X,C,EA,dofs)
    kint = ddms(u,X,C,EA,dofs)
    duds = get_duds(s,u0,idx_act,sh,Circ,kc1)
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
        dfndn = (fnode@nor)*np.eye(3)+np.outer(fnode,nor)
        dfndf = np.outer(nor,nor)
        dftdf = np.eye(3)
        dftdfn = -np.eye(3)
        dc2dfn = mu*fn/norm(fn)
        dc2dft = -ft/norm(ft)

        dfdu = kint[dofi]

        dc2ds[cnt_act,2*cnt_act:2*cnt_act+2] += (dc2dfn + dc2dft@dftdfn)@dfndn@dnords
        dc2ds[cnt_act] += (dc2dfn@dfndf+dc2dft@(dftdf+dftdfn@dfndf))@dfdu@duds

        cnt_act+=1

    return dc2ds


def verdict(u,Circ,idx_act,sh,s):
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    act, shnew, Redo = [], [None]*ns, False
    fint = dms(u,X,C,EA,dofs,Circ,idx_act)
    for idx, xsi in enumerate(xs):

        geps, feps = 1e-4, 1e-9

        gn = norm(xsi-np.array(Cxy))-Cr
        print("gn for node",slaves[idx],":",gn)

        if idx in idx_act:      # was active
            si = s[idx]
            nor = get_nor(si)
            fnode = fint[dofs[slaves[idx]]]
            Fn = fnode@nor
            print("Fn for node",slaves[idx],":",Fn)

            if Fn<-feps:     # being pulled (out)
                print("node",slaves[idx],"exited")
                Redo = True
                continue
            else:
                ft = fnode - Fn*nor
                Ft = norm(ft)
                if Ft/(mu*Fn)<0.99:    # elastic case
                    shnew[idx] = sh[idx]
                    print("node",slaves[idx],"continues elastic")
                else:
                    shnew[idx] = si
                    print("node",slaves[idx],"continues plastic")

        elif gn<-geps:
            print("node",slaves[idx],"just entered")
            Redo = True
        else:
            continue
        act.append(idx)
                
    return act, shnew, Redo

def verdict_fless(u,Circ,idx_act):
    Cxy, Cr = Circ
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    act, shnew, Redo = [], [None]*ns, False
    fint = dms(u,X,C,EA,dofs,Circ,idx_act)
    for idx, xsi in enumerate(xs):

        gn = norm(xsi-np.array(Cxy))-Cr
        print("gn for node",slaves[idx],":",gn)

        if idx in idx_act:      # was active
            if gn<-gmax:
                Redo=True
                a_k[idx] *= 1.5*abs(gn/gmax)
                print("too much penetration in node ",slaves[idx])
            else:
                si = getProjs(xsi,Circ)
                nor = get_nor(si)
                fnode = fint[dofs[slaves[idx]]]
                Fn = fnode@nor
                print("Fn for node",slaves[idx],":",Fn)

                if Fn<-feps:     # being pulled (out)
                    print("node",slaves[idx],"exited")
                    Redo = True
                    continue    # node doesn't append the list
                elif 0>gn>-gmin and a_k[idx]>1:
                    Redo=True
                    a_k[idx] = max(((2/3)*abs(gn)/gmin)*a_k[idx],1)
                    print("too little penetration in node ",slaves[idx])
        elif gn<-geps:
            print("node",slaves[idx],"just entered")
            Redo = True
        else:
            continue
        act.append(idx)
                
    return act, shnew, Redo


def getProjs(xs,Circ):
    if len(xs.shape)<2:     # if only one point...
        Cxy = Circ[0]
        dxyi = xs-Cxy
        si1 = np.sign(-dxyi[2])*acos(-dxyi[0]/sqrt(dxyi[0]**2+dxyi[2]**2))/pi
        si2 = 1-acos(-dxyi[1]/norm(dxyi))/pi
        return [si1,si2]
    else:                   # multiple points...
        s = np.zeros((len(xs),2))
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


X = np.zeros((70,3))
X[:,:2], C = Truss2D(6,9,0.25,0.25,diags=False)

btypes = None
dim = len(X[0])
dofs = np.array(range(dim*len(X))).reshape(-1,dim)
ndofs = len(dofs.ravel())
X = Translate(X,[-1.5,-2.25/2, 0.0])
EA = 1.0
etype = "truss"

base = [0,6,63,69]
slaves = list(set(range(70))-set(base))
dofs_base_x = dofs[base][:,0]
dofs_base_y = dofs[base][:,1]
dofs_base_z = dofs[base][:,2]
ns = len(slaves)
mu = 0.0

# penetration control
gmax = 0.0001
gmin = 0.5*gmax
geps, feps = 1e-4, 1e-9 # At least feps seems to fix problems in FLESS case
a_k  = np.ones(ns)


# Rigid Circle
Cxy = [1.0, 0.0, 3.0]
Cr  = 3.1
Circ = [Cxy,Cr]

u = np.zeros(ndofs)
idx_act = []
plotMS(u.reshape(-1,3),X,C, show=True, ax=None)

#Step1  (Right)
bc1 = [dofs_base_x,"dir", 4.0, 0.0,1.0]
bc2 = [dofs_base_y,"dir", 0.0, 0.0,1.0]
# bc3 = [dofs[base[0::2]][:,2],"dir", 0.0, 0.0,0.1]
# bc4 = [dofs[base[1::2]][:,2],"dir",-0.05, 0.0,0.1]
bc5 = [dofs_base_z,"dir", 0.0, 0.0,1.0]
# BCs = [bc1,bc2,bc3,bc4,bc5]
BCs = [bc1,bc2,bc5]

# kn = kt = 1e3
kn = kt = 0.1
tol = 1e-8

TT = []
dX, FN, FFN, FT, RES,SS1,SS2= [],[],[],[],[],[],[]
NEVALS = []

meth = 'BFGS'

folder = meth+"_"+strftime("%y%m%d-%H%M")
if not os.path.exists(folder):
    os.mkdir(folder)
idx_act = []
u = np.zeros(ndofs)
fint = np.zeros(ndofs)

t,tt=0.0, 0
dt=0.01
xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]

nevals = []
trials = 0
trace, InRepeat=False, False
idx_act,idx_act_pre = [],[]

while t+dt<=1.0+1e-4:
    print("\n----\ntime:", round(t,4), " to ",round(t+dt,4),"\n----")
    # Apply BCs
    du,df,di=ApplyBCs(t,t+dt, BCs,ndofs)
    fr = np.delete(np.arange(ndofs), di)       # free DOFs 'b'
    u += du
    fint += df

    idx_hooked = [i for i in idx_act if sh[i] is not None]

    PairsUsed = []

    args = (X,C,EA,dofs,Circ,idx_act)
    cons = ({'type':'eq', 'fun':Cgn, 'jac':dCgndu, 'args':args})
    if meth=="SLSQP":
        pass
        # MinPro = minimize(ms,u,args=args,method=meth,jac=dms,hess=ddms, constraints=cons,tol=1e-5,options={'disp':True}).x
    elif meth=="BFGS":
        MinPro = minimize(m_free,u,args=args,method=meth,jac=f_free,hess=k_free,constraints=cons,tol=1e-8,options={'disp':True,'maxiter':10000})
    u = MinPro.x
    nevals.append(MinPro.nfev)

    # set_trace()
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    FF = MinPro.jac


    print("BEFORE VERDICT:")
    print("act: ",[slaves[i] for i in idx_act])
    
    # idx_act, sh, Redo = verdict(u,Circ,idx_act,sh,s.copy()) # 's' is just input. it MUST be entered as copy
    idx_act, sh, Redo = verdict_fless(u,Circ,idx_act) # 's' is just input. it MUST be entered as copy

    print("AFTER VERDICT:")
    print("act: ",[slaves[i] for i in idx_act])

    if Redo:    
        u = np.array(u_pre)
        xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
        InRepeat = True
        trials += 1
        print("REDOING INCREMENT...")
        Redo = False
        continue

    idx_act_pre = idx_act
    u_pre = np.array(u)

    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]

    print("residual out:",norm(FF))

    fint = dms(u,X,C,EA,dofs,Circ,idx_act)
    F = fint[dofs[base]].reshape(-1,3).sum(axis=0)
    
    savedata(t)
    plotMS(u.reshape(-1,3),X,C,undef=True,save=folder)

    t += dt
    tt+=1
    InRepeat=False
    trials = 0      # if here means that it converged into the right contact config

    TT.append(t)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(221)

# for idx, mu in enumerate(MUs):
ax.plot(TT,nevals)
ax.set_xlabel("time")
ax.set_ylabel("n_fevals")
plt.savefig(folder+"/results.png")

print("--FINISHED--")

