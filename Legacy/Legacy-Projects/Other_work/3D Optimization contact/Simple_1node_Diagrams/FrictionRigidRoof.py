from pdb import set_trace
# from re import X
import numpy as np
from numpy.linalg import norm
from math import sqrt, sin, cos, atan,atan2, acos, pi, log10
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
    if ax is None:        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

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

    if ax is None:  plt.show()

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

def plotRoof(z_roof,ax=None):
    if ax is None:        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')


    x = np.array([[-3,3],[-3,3]])
    y = x.T
    z = z_roof*np.ones((2,2))
    ax.plot_surface(x,y,z,linewidth=0.5, color=(0.5,0.5,0.5,0.5))
    if ax is None:  plt.show()

def plotMS(u,X,C, ax=None, undef=False, show=False, save=None,sline = None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

    plotTruss(X+u,C,ax=ax,bar_types=btypes,show_nodes=True,show_id=False,showtypes=0)
    plotRoof(z_roof,ax=ax)

    ax.set_xlim([-2,2])
    ax.set_ylim([-2.0,2.0])
    ax.set_zlim([-1.0,1.0])

    if show: plt.show()

    if save is not None:
        if not os.path.exists(folder+"/plots"):
            os.mkdir(folder+"/plots")

        plt.savefig(folder+"/plots//img"+str(tt)+".png")
        plt.close(fig)

def plotMinimization(s,objf,xrange,yrange,n,tt,sm):
    if not os.path.exists(folder+"/plots"):
        os.mkdir(folder+"/plots")

    dels=1e-2
    sx = np.linspace(max(sm[0]-15*dels,0),min(sm[0]+15*dels,1),n)
    sy = np.linspace(max(sm[1]-15*dels,0),min(sm[1]+15*dels,1),n)
    Sx = np.linspace(xrange[0],xrange[1],n)
    Sy = np.linspace(yrange[0],yrange[1],n)
    mz =np.zeros((n,n))
    colors  = np.zeros((n,n,4))
    if len(idx_hooked)>0:
        mz = [[objf(np.array([s1i,s2j]),u,idx_act,sh,z_roof,1) for s2j in Sy] for s1i in Sx]
    for idx,s1 in enumerate(sx):
        for jdx,s2 in enumerate(sy):
            # mz[idx,jdx] = log10(mNM([s1,s2],u,idx_act,sh,Circ)+1)
            if len(idx_hooked)>0:
                # mz[idx,jdx] = objf(np.array([s1,s2]),u,idx_act,sh,Circ,1)
                C1 = c1(np.array([s1,s2]),u,idx_act,sh,z_roof,1)
                C2 = c2(np.array([s1,s2]),u,idx_act,sh,z_roof,1)
            else: 
                C1 = C2 = [-1,-1]
            # if out of both constraints
            if all([abs(cc1)<1e-2 for cc1 in C1]) and all([cc2>-1e-4 for cc2 in C2]):
                # colors[idx,jdx,:] = (0.5,0.0,0.5,1.0)   # purlple
                colors[idx,jdx,:] = (1.0,0.5,0.2,1.0)   # orange
            elif all([abs(cc1)<1e-2 for cc1 in C1]):
                colors[idx,jdx,:] = (1.0,0.0,0.0,1.0)   # red
            elif all([cc2>-1e-4 for cc2 in C2]):
                colors[idx,jdx,:] = (0.0,0.0,1.0,1.0)   # blue
            else:
                colors[idx,jdx,:] = (1.0,1.0,1.0,0.0)   # white/transparent
    pickle.dump([Sx,Sy,sx,sy,mz,colors],open(folder+"/plots/Minimization.dat","wb"))

    Sx,Sy,sx,sy,mz,Colors = pickle.load(open("/home/diego/fem/3D Optimization contact/Simple_1node_Diagrams/"+folder+"/plots/Minimization.dat","rb"))

    import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(20,12))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212,projection='3d')

    sx,sy = np.meshgrid(sx,sy)
    Sx,Sy = np.meshgrid(Sx,Sy)
    ax.contour(Sx.T,Sy.T,mz,50,alpha=0.25)
    Colors[:,:,3]=0.3
    ax.pcolormesh(sx.T,sy.T,Colors,shading='nearest')
    sm = s_of_xc(xm,xl,xu,yl,yu)
    xtot = X+u.reshape(-1,3)
    xxa = s_of_xc(xtot[1],xl,xu,yl,yu)
    xxb = s_of_xc(xtot[2],xl,xu,yl,yu)
    xxc = s_of_xc(xtot[3],xl,xu,yl,yu)
    xxd = s_of_xc(xtot[4],xl,xu,yl,yu)
    # ax.scatter([sm[0]-0.1,sm[0]+0.1,sm[0]+0.1,sm[0]-0.1],[sm[1]-0.1,sm[1]+0.1,sm[1]-0.1,sm[1]+0.1],c='k',s=30)
    ax.plot([xxa[0],s[0],xxd[0]],[xxa[1],s[1],xxd[1]],c='b',linewidth=0.5,zorder=0)
    ax.plot([xxb[0],s[0],xxc[0]],[xxb[1],s[1],xxc[1]],c='b',linewidth=0.5,zorder=0)
    ax.scatter(s[0],s[1],marker="o",s=50,c='r'if len(idx_act)>0 else 'b',zorder=10)
    if sh[0] is not None:
        ax.scatter(sh[0][0],sh[0][1],marker="x",s=50,c='k',zorder=20)
    # ax.margins(y=1000)
    # PU = np.array(PairsUsed)
    # ax.scatter(PU[:,0],PU[:,1],color=(0.5,0.5,0.5,0.6),s=10)
    plotMS(u.reshape(-1,3),X,C, ax=ax2)
    ax2.xaxis.set_ticklabels([])      #no numbers on axis
    ax2.yaxis.set_ticklabels([])      #no numbers on axis
    ax2.zaxis.set_ticklabels([])
    ax2.dist=10
    # ax2.axis('off')
    plt.savefig(folder+"/plots/Minimization"+str(tt)+".png",format='png',dpi=200, bbox_inches='tight')
    # plt.show()
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


def ms(u,X,C,EA):
    
    x=X+u.reshape(-1,3)
    ms=0
    for (i,j) in C:
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        ms+=(EA*(L-L0)**2)/(2*L0)
    return ms

def dms(u,X,C,EA,dofs,ret = 0):
    x=X+u.reshape(-1,3)
    dms=np.zeros(ndofs)
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        dms[np.ix_(dofij)]+=EA*(L-L0)/L0*dLdu

    return dms

def ddms(u,X,C,EA,dofs,ret=1):
    x=X+u.reshape(-1,3)
    ddms=np.zeros((ndofs,ndofs))
    for (i,j) in C:
        dofij = np.append(dofs[i],dofs[j])
        L0= norm(X[i]-X[j])
        L = norm(x[i]-x[j])
        a = np.array([x[i]-x[j],x[j]-x[i]]).ravel()
        dLdu = a/L
        dadu = np.array([[1.0,0.0,0.0,-1.0,0.0,0.0],[0.0,1.0,0.0,0.0,-1.0,0.0],[0.0,0.0,1.0,0.0,0.0,-1.0],[-1.0,0.0,0.0,1.0,0.0,0.0],[0.0,-1.0,0.0,0.0,1.0,0.0],[0.0,0.0,-1.0,0.0,0.0,1.0]])
        d2Ldu2 = dadu/L - np.outer(a,a)/(L**3)
        ddms[np.ix_(dofij,dofij)] += (np.outer(dLdu,dLdu)+(L-L0)*d2Ldu2)/L0
    return EA*ddms


def ddms_fd(u,X,C,EA,dofs, du=1e-7):
    ddms = np.zeros((len(u),len(u)))
    for idx in range(len(u)):
        u1 = np.array(u)
        u1[idx]+=du
        ddms[idx,:] = (dms(u1,X,C,EA,dofs) - dms(u,X,C,EA,dofs))/du
    return ddms


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
def solve_for_a_given_s(u0,z_roof,s,idx_act,cache_key):
    s = np.array(s)
    u_iter = NR(u0,f_rstr,k_rstr,args=(z_roof,s.reshape(-1,2),idx_act))
    # u_iter = minimize(m_rstr,u0,args=(Circ,s.reshape(-1,2),idx_act),method='Newton-CG',jac=f_rstr,hess=k_rstr,tol=1e-7,options={'disp':False}).x
    if type(u_iter) != np.ndarray:
        print("NR failed, trying Newton-CG...")
        u_iter = minimize(m_rstr,u0,args=(z_roof,s.reshape(-1,2),idx_act),method='Newton-CG',jac=f_rstr,hess=k_rstr,options={'disp':False}).x
    return u_iter

def solve_for_a_given_s_wrapper(u0,z_roof,s,idx_act):
    cache_key = (tuple(u0),z_roof,tuple(idx_act))
    # set_trace()
    return solve_for_a_given_s(tuple(u0),z_roof,tuple(s.ravel()),tuple(idx_act),cache_key)

def u_of_s(s,Circ,u0,idx_act):
    return solve_for_a_given_s_wrapper(u0,Circ,s,idx_act)


def mgt2(s,u0,idx_act,sh,z_roof,kc1,tracePoints=True):
    if tracePoints: PairsUsed.append(s)

    s = s.reshape(-1,2)
    # u = solve_for_a_given_s_wrapper(u0,z_roof,s,idx_act)
    mOBJ,cnt_act = 0.0 ,0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act]
        cnt_act+=1
        xc = xc_of_s(si,xl,xu,yl,yu)
        xch= xc_of_s(sh[idx],xl,xu,yl,yu)

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
def c1(s,u0,idx_act,sh,z_roof, kc1, eval = None):
    s = s.reshape(-1,2)
    if eval is None:
        u = solve_for_a_given_s_wrapper(u0,z_roof,s,idx_act)
    else: u = eval
    c1 = np.zeros(len(idx_act))
    cnt_act = 0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act]
        nor = get_nor(si)
        xc = xc_of_s(si,xl,xu,yl,yu)
        xch = xc_of_s(sh[idx],xl,xu,yl,yu)
   
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
    return 1e-3-c1**2

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


def m_rstr(u,z_roof,s,idx_act, show=False):
    s = np.array(s).reshape(-1,2)
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    mN, mT = 0.0 , 0.0
    cnt_act = 0
    for idx in idx_act:
        xsi = xs[idx]
        gn = z_roof - xsi[2]
        mN += 1/2*kn*gn**2
        if sh[idx] is None: continue
        si = s[cnt_act]
        cnt_act+=1
        nor = np.array([0,0,-1])
        xc = xc_of_s(si,xl,xu,yl,yu)
        dxsi = xsi-xc
        dxs_proj = dxsi - (dxsi@nor)*nor
        mT += 1/2*kt*norm(dxs_proj)**2
    m = ms(u,X,C,EA) + mN + mT
    if show: print("here in m (r) \t m =",m)

    return m

def f_of_s(s,u,z_roof,idx_act):
    return f_rstr(u,z_roof,s,idx_act)

def f_rstr(u,z_roof,s,idx_act, show=False):
    s = np.array(s).reshape(-1,2)
    f = np.zeros_like(u)
    f[fr] = dms(u,X,C,EA,dofs)[fr]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in idx_act:
        xsi = xs[idx]
        gn = z_roof - xsi[2]
        dgndu = np.array([0,0,-1])
        f[dofs[slaves[idx]]] += kn*gn*dgndu                  # += fN
        if sh[idx] is None: continue
        si = s[cnt_act]
        cnt_act+=1
        nor = np.array([0,0,-1])
        xc = xc_of_s(si,xl,xu,yl,yu)
        dxsi = xsi-xc
        dxs_proj = dxsi - (dxsi@nor)*nor
        
        f[dofs[slaves[idx]]] += kt*dxs_proj@(np.eye(3)-np.outer(nor,nor))  # += fT

    if show: print("here in Fi (r) \tRES:",norm(f))
        
    return f

def k_rstr(u,z_roof,s,idx_act, show=False):
    s = np.array(s).reshape(-1,2)
    kij = np.zeros((len(u),len(u)))
    kij[np.ix_(fr,fr)] = ddms(u,X,C,EA,dofs)[np.ix_(fr,fr)]
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    cnt_act = 0
    for idx in idx_act:
        xsi = xs[idx]
        gn = z_roof - xsi[2]
        dgndu = np.array([0,0,-1])
        d2gndu2 = 0.0
        kij[np.ix_(dofs[slaves[idx]],dofs[slaves[idx]])] += kn*(np.outer(dgndu,dgndu)+gn*d2gndu2)                   # += fN
        if sh[idx] is None: continue
        cnt_act+=1
        nor = np.array([0,0,-1])
        mij = np.eye(3)-np.outer(nor,nor)
        # set_trace()
        kij[np.ix_(dofs[slaves[idx]],dofs[slaves[idx]])] += kt*mij@mij  # += kij
    
    if show: print("here in Kij (r)")
    
    return kij

"""
def xc(s1,s2,Circ):
    Cxy,Cr = Circ
    theta = pi*(1-s2)
    phi = pi*s1
    return Cxy - Cr*np.array([cos(phi)*sin(theta), cos(theta),sin(theta)*sin(phi)])

def dxc(s1,s2,Circ):
    theta = pi*(1-s2)
    phi = pi*s1
    tau1 = -np.array([-sin(phi)*sin(theta), 0.0,sin(theta)*cos(phi)])
    tau2 = -np.array([-cos(phi)*cos(theta), sin(theta),-cos(theta)*sin(phi)])
    return np.array([tau1,tau2])
"""
def get_nor(s):
    return np.array([0,0,-1])

def get_dnords(s):
    return np.zeros((3,2))

def get_dmds(s,u0,idx_act,sh,z_roof,kc1):
    s = s.reshape(-1,2)
    dmds,cnt_act = np.zeros_like(s) ,0
    for idx in idx_act:
        if sh[idx] is None: continue
        si = s[cnt_act]
        xc = xc_of_s(si,xl,xu,yl,yu)
        dxcds = np.array([[xu-xl,0.0],[0.0,yu-yl],[0,0]])
        xch= xc_of_s(sh[idx],xl,xu,yl,yu)
        dmds[cnt_act]=2*(xc-xch)@dxcds
        cnt_act+=1

    return dmds


def get_duds(s,u0,idx_act,sh,z_roof,kc1):
    s = s.reshape(-1,2)
    u = solve_for_a_given_s_wrapper(u0,z_roof,s,idx_act)
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    KK = k_rstr(u,z_roof,s,idx_act,)
    dfds,cnt_act = np.zeros((KK.shape[0],len(s),2)) ,0
    duds = np.zeros((ndofs,2*len(s)))
    for idx in idx_act:
        if sh[idx] is None: continue
        xsi = xs[idx]
        si = s[cnt_act]
        xc = xc_of_s(si,xl,xu,yl,yu)
        dxcds = np.array([[xu-xl,0.0],[0.0,yu-yl],[0.0,0.0]])
        dnords = get_dnords(si)
        nor = get_nor(si)
        dsi = xsi - xc
        dsip = dsi - (dsi@nor)*nor
        ddsipddsi = np.eye(3)-np.outer(nor,nor)
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


def dc1ds(s,u0,idx_act,sh,z_roof,kc1):
    s = s.reshape(-1,2)
    u = solve_for_a_given_s_wrapper(u0,z_roof,s,idx_act)
    dc1ds,cnt_act = np.zeros((len(s),2*len(s))) ,0
    fint = dms(u,X,C,EA,dofs)
    kint = ddms(u,X,C,EA,dofs)
    duds = get_duds(s,u0,idx_act,sh,z_roof,kc1)
    for idx in idx_act:
        if sh[idx] is None: continue
        dofi = dofs[slaves[idx]]
        si = s[cnt_act]
        nor = get_nor(si)
        xc = xc_of_s(si,xl,xu,yl,yu)
        xch= xc_of_s(sh[idx],xl,xu,yl,yu)
        
        fnode = fint[dofi]
        ft = fnode - (fnode@nor)*nor
        dsh = (xch-xc)
        taup = dsh - (dsh@nor)*nor
        tau = taup/norm(taup)

        dxcds = np.array([[xu-xl,0.0],[0.0,yu-yl],[0,0]])
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

def dc2ds(s,u0,idx_act,sh,z_roof,kc1):
    s = s.reshape(-1,2)
    u = solve_for_a_given_s_wrapper(u0,z_roof,s,idx_act)
    dc2ds,cnt_act = np.zeros((len(s),2*len(s))) ,0
    fint = dms(u,X,C,EA,dofs)
    kint = ddms(u,X,C,EA,dofs)
    duds = get_duds(s,u0,idx_act,sh,z_roof,kc1)
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


def verdict(u,z_roof,idx_act,sh,s):
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    act, shnew, Redo = [], [None]*ns, False
    fint = dms(u,X,C,EA,dofs)
    for idx, xsi in enumerate(xs):

        gn = z_roof - xsi[2]
        
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
                    shnew[idx] = si.copy()
                else:
                    fnode = fint[dofs[slaves[idx]]]
                    f_proj = fnode - (fnode@nor)*nor
                    
                    if norm(f_proj) <= abs(mu*fnode@nor):    # elastic case
                        shnew[idx] = sh[idx].copy()
                        print("node",slaves[idx],"continues elastic")
                    else:
                        # shnew[idx] = 0.5*(si+sh[idx])
                        shnew[idx] = si.copy()
                        print("node",slaves[idx],"continues plastic")
                    print("fnode",fnode)

            else:
                print("node",slaves[idx],"just entered")
                Redo = True
            
            act.append(idx)
    
    return act, shnew, Redo



def getProjs(xs,z_roof):
    if len(xs.shape)<2:     # if only one point...
        return s_of_xc(xs,xl,xu,yl,yu)
    else:                   # multiple points...
        s = np.zeros((len(xs),2))
        for idx, xsi in enumerate(xs):
            s[idx] = getProjs(xsi,z_roof)
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
    simplex[0] = x0-delta*np.ones_like(x0)
    for i in range(1, n+1):
        simplex[i] = x0.copy()
        # if type(delta) not in [int,float]: set_trace()
        deli = delta if type(delta) in [int,float] else delta[i-1]
        deli = 0.001 if abs(deli)<0.001 else deli   # helps to prevent delta=0
        simplex[i] += [(alpha/(1+mu))**(abs(i-j))*deli for j in range(1, n+1)]
    return simplex

def xc_of_s(s,xl,xu,yl,yu):
    xc = np.array([xl+s[0]*(xu-xl),
                   yl+s[1]*(yu-yl),
                   z_roof])
    return xc

def s_of_xc(xc,xl,xu,yl,yu):
    s1 = (xc[0]-xl)/(xu-xl)
    s2 = (xc[1]-yl)/(yu-yl)
    return np.array([s1,s2])

X = np.array([[ 0.0, 0.0,-0.2],
              [-1.0,-1.0,-1.5],
              [ 1.0,-1.0,-1.5],
              [ 1.0, 1.0,-1.5],
              [-1.0, 1.0,-1.5]])

X = Translate(X,[-1,0.0, 0.0])
C = [[0,1],[0,2],[0,3],[0,4]]

xl,xu = -3,3
yl,yu = -3,3


btypes = None
dim = len(X[0])
dofs = np.array(range(dim*len(X))).reshape(-1,dim)
ndofs = len(dofs.ravel())
EA = 1.0

base = [1,2,3,4]
slaves = [0]
dofs_base_x = dofs[base][:,0]
dofs_base_y = dofs[base][:,1]
dofs_base_z = dofs[base][:,2]
ns = len(slaves)
mu = 1.0

# Rigid Circle
# Cxy = [2.0, 0.0, 3.0]
# Cr  = 3.1
# Circ = [Cxy,Cr]
z_roof = 0.1

u = np.zeros(ndofs)
idx_act = []
plotMS(u.reshape(-1,3),X,C, show=True, ax=None)



#Step1  (Right)
zup, dis = 1.0, 3
bc1 = [dofs_base_x,"dir", 0.0, 0.0,0.4]
bc2 = [dofs_base_y,"dir", 0.0, 0.0,1.0]
bc3 = [dofs_base_z,"dir", zup, 0.0,0.4]
bc4 = [dofs_base_x,"dir", dis, 0.4,1.0]
bc5 = [dofs_base_z,"dir", 0.0, 0.4,1.0]

BCs = [bc1,bc2,bc3,bc4,bc5]

kn = kt = 1e3
tol = 1e-8

TT = []
dX, FN, FFN, FT, RES,SS1,SS2= [],[],[],[],[],[],[]
NEVALS = []


meth = ['SLSQP-con']


NSIM = len(meth)


for idx in range(NSIM):
    folder = meth[idx]+"_"+strftime("%y%m%d-%H%M")
    if not os.path.exists(folder):
        os.mkdir(folder)
    idx_act = []
    u = np.zeros(ndofs)
    u_pre = u.copy()
    fint = np.zeros(ndofs)
    sh = [None]*ns   # Initial hook

    t,tt=0.0, 0
    dt=0.01
    xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
    s = getProjs(xs,z_roof)

    # plotMS(u.reshape(-1,3),X,C,undef=True,show=True)

    s_pre = s.copy()
    ds = np.zeros_like(s)
    ds_pre = np.zeros_like(s)
    sh_pre = sh.copy()
    print("first dummy 's':",s)
    
    FNi, FFNi, FTi, RESi = [],[],[],[]
    nevals = []
    trials = 0
    trace, InRepeat=False, False
    idx_act,idx_act_pre = [],[]

    while t+dt<=1.0+1e-4:
        print("\n----\ntime:", round(t,4), " to ",round(t+dt,4),"\n----")
        print("nowSH: ",sh)
        # Apply BCs
        du,df,di=ApplyBCs(t,t+dt, BCs,ndofs)
        fr = np.delete(np.arange(ndofs), di)       # free DOFs 'b'
        u += du
        fint += df

        ds = 1e-5*np.ones(ns)

        idx_hooked = [i for i in idx_act if sh[i] is not None]
        s0 = (s_pre+(ds))[idx_hooked]

        PairsUsed = []

        kc1=10
        args = (u,idx_hooked,sh,z_roof,kc1)
        cons = ({'type':'eq', 'fun':c1, 'jac':dc1ds, 'args':args},
                {'type':'ineq', 'fun':c2, 'jac':dc2ds, 'args':args})
        bds = Bounds(lb=s0.ravel()-np.maximum(0.5*np.abs(ds[idx_hooked]).ravel(),0.05*np.ones(2*len(idx_hooked))),
                     ub=s0.ravel()+np.maximum(0.5*np.abs(ds[idx_hooked]).ravel(),0.05*np.ones(2*len(idx_hooked))))
        
        if meth[idx]=="Nelder-Mead":
            if len(idx_hooked)>0:
                simplex = get_simplex2(s0.ravel(),0.001,0.6)
                NM = minimize(mNM,s0.ravel(),method='Nelder-Mead',args=(u,idx_act,sh,z_roof,True),options={'bounds':bds,'disp':False,'initial_simplex':simplex,'adaptive':len(idx_act)>0},tol=1e-5)
        elif meth[idx]=='SLSQP-con':
            if len(idx_hooked)>0:
                NM = minimize(mgt2,s0.ravel(),method='SLSQP',jac=get_dmds,args=args, constraints=cons,bounds= bds,tol=1e-3,
                              options={'disp':True,"iprint":2})
                

        if len(idx_hooked)>0:
            s[idx_hooked] = NM.x.reshape(-1,2)
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

        s=s.reshape(-1,2)
        u = solve_for_a_given_s_wrapper(u,z_roof,s[idx_hooked],idx_act)
        xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
        FF = f_rstr(u,z_roof,s[idx_hooked],idx_act)

        if t>0:
            sx, sy = s[idx_hooked[0]] if len(idx_hooked)>0 else s[0]
            xb = (X+u.reshape(-1,3))[1:]
            xm = np.average(xb,axis=0)
            sm = s_of_xc(xm,xl,xu,yl,yu)
            plotMinimization(s[0],mgt2,[0,1],
                                        [0.3,0.7],
                                        100,tt,sm)




        print("BEFORE VERDICT:")
        print("act: ",idx_act)
        print("sh : ",sh)
        
        idx_act, sh, Redo = verdict(u,z_roof,idx_act,sh.copy(),s.copy()) # 's' is just input. it MUST be entered as copy

        print("AFTER VERDICT:")
        print("act: ",idx_act)
        print("sh : ",sh)

        # if t>0.04999: set_trace()

        if Redo:    
            u = np.array(u_pre)
            s, ds = s_pre.copy(), ds_pre.copy()
            for i in range(ns):
                if not (sh[i] is None or sh_pre[i] is None):
                    sh[i] = sh_pre[i].copy()
            
            xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]
            InRepeat = True
            trials += 1
            print("REDOING INCREMENT...")
            Redo = False
            continue


        unused  = list(set(range(ns))-set(idx_hooked))
        # s = getProjs(xs,Circ)
        s[unused] = getProjs(xs,z_roof)[unused]


        sh_pre = sh.copy()
        idx_act_pre = idx_act
        ds_pre = ds.copy()
        ds = s-s_pre
        u_pre = np.array(u)
        s_pre = s.copy()

        xs = X.ravel()[dofs[slaves]]+u[dofs[slaves]]


        print("residual out:",norm(FF))

        fint = dms(u,X,C,EA,dofs)
        F = fint[dofs[base]].reshape(-1,3).sum(axis=0)
        
        for i, (xsi, shi) in enumerate(zip(xs,sh)):
            if shi is not None:
                fnode = fint[dofs[slaves[i]]]
                nor = np.array([0,0,-1])
                fn = fnode@nor
                ft = fnode - (fnode@nor)*nor
                print("fnode",fnode)
                print("node",slaves[i],"\t Ft =",norm(ft),"\tµFn =",mu*fn, ["%.2E" %i for i in ft])
            else:
                fn=ft=0

        FNi.append(fn)
        FTi.append(norm(ft))
        RESi.append(log10(norm(FF)))

        savedata(t)
        # plotMS(u.reshape(-1,3),X,C,undef=True,save=folder)


        t += dt
        tt+=1
        InRepeat=False
        trials = 0      # if here means that it converged into the right contact config

        if idx==0: TT.append(t)


    FN.append(FNi)
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
    ax1.plot(TT,FN[idx],label='Fn')
    ax1.plot(TT,FT[idx],label='Ft')
    ax2.plot(TT,NEVALS[idx],label=meth[idx])
    ax3.plot(TT,RES[idx],label=meth[idx])
ax1.legend(),ax2.legend(),ax3.legend(),ax4.legend()
ax1.set_xlabel("time")
ax1.set_ylabel("F")
ax2.set_xlabel("time")
ax2.set_ylabel("# of evals")
ax3.set_xlabel("time")
ax3.set_ylabel("log10(res)")
plt.show()
plt.savefig(folder+"/results.png")

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