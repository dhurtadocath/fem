#################
### FUNCTIONS ###
#################
from numpy.matrixlib import defmatrix
import numpy as np
from scipy.special import comb, factorial
import math
from pdb import set_trace

# def RandDispl(dirs = dirs, range = interv,steps= steps, X0 = [0.0, 0.0, 0.0]):
#     x = np.array(X0)
#     disps = np.empty((steps,3))
#     for i in range(len(disps)):
#         if "x" in dirs or "X" in dirs:
#             dx = 0
#             while x[0] + dx

def relDiff(A,B, disp=False, tol=1e-10, reciproc=True):
    "Computes the relative difference between matrices"
    A = np.where(abs(A)<tol,0.0,A)
    B = np.where(abs(B)<tol,0.0,B)
    diffRel = np.where(A==0.0, 0.0, np.divide(B-A,A))
    # diffRel = np.where(np.isfinite(diffRel),diffRel,0.0)
    if len(diffRel.shape)==2:
        mins = [min(diffRel[i]) for i in range(len(diffRel))]
        maxs = [max(diffRel[i]) for i in range(len(diffRel))]
        MIN = abs(min(mins))
        MAX = abs(max(maxs))
    elif len(diffRel.shape)==1:
        MIN = min(diffRel)
        MAX = max(diffRel)

    MAX = max(MIN,MAX)

    if reciproc:
        MAX = max(MAX, relDiff(B,A,reciproc=False))

    printif(disp,"maxRelDiff: ",MAX)
    return MAX


def flatList(l1):

    ls = []
    for li in l1:
        if type(li)==list:
            new = flatList(li)
        else:
            new = [li]

        ls.extend(new)

    return ls

def printif(cond, *args):
    if cond:
        print(*args)

def Bernstein(n,k,x):
    return comb(n,k)*(x**k)*((1-x)**(n-k))

def dBernstein(n,k,x, cases = True):
    if cases:
        if k==0:
            dB = comb(n,k)*( (x**k)*(-1*(n-k)*(1-x)**(n-k-1)) )
        elif k==n:
            dB = comb(n,k)*( (k*x**(k-1))*((1-x)**(n-k)) )
        else:
            dB = comb(n,k)*( (k*x**(k-1))*((1-x)**(n-k)) + (x**k)*(-1*(n-k)*(1-x)**(n-k-1)) )
    else:
        dB = comb(n,k)*( (k*x**(k-1))*((1-x)**(n-k)) + (x**k)*(-1*(n-k)*(1-x)**(n-k-1)) )

    return dB

def d2Bernstein(n,k,x, cases = True):
    if cases:           # without cases there are number/0 problems   for x=0
        if k==0:
            d2B = comb(n,k)*( (n-k)*(n-k-1)*x**k*(1-x)**(n-k-2) )
        elif k==1:
            d2B = comb(n,k)*( -2*k*(n-k)*x**(k-1)*(1-x)**(n-k-1) + (n-k)*(n-k-1)*x**k*(1-x)**(n-k-2) )
        elif k==n-1:
            d2B = comb(n,k)*( k*(k-1)*x**(k-2)*(1-x)**(n-k) - 2*k*(n-k)*x**(k-1)*(1-x)**(n-k-1) )
        elif k==n:
            d2B = comb(n,k)*( k*(k-1)*x**(k-2)*(1-x)**(n-k) )
        else:
            d2B = comb(n,k)*( k*(k-1)*x**(k-2)*(1-x)**(n-k) - 2*k*(n-k)*x**(k-1)*(1-x)**(n-k-1) + (n-k)*(n-k-1)*x**k*(1-x)**(n-k-2) )
    else:
        d2B = comb(n,k)*( k*(k-1)*x**(k-2)*(1-x)**(n-k) - 2*k*(n-k)*x**(k-1)*(1-x)**(n-k-1) + (n-k)*(n-k-1)*x**k*(1-x)**(n-k-2) )

    return d2B

def float_to_fraction (x, error=0.000001):
    n = int(math.floor(x))
    x -= n
    if x < error:
        return (n, 1)
    elif 1 - error < x:
        return (n+1, 1)

    # The lower fraction is 0/1
    lower_n = 0
    lower_d = 1
    # The upper fraction is 1/1
    upper_n = 1
    upper_d = 1
    while True:
        # The middle fraction is (lower_n + upper_n) / (lower_d + upper_d)
        middle_n = lower_n + upper_n
        middle_d = lower_d + upper_d
        # If x + error < middle
        if middle_d * (x + error) < middle_n:
            # middle is our new upper
            upper_n = middle_n
            upper_d = middle_d
        # Else If middle < x - error
        elif middle_n < (x - error) * middle_d:
            # middle is our new lower
            lower_n = middle_n
            lower_d = middle_d
        # Else middle is our best fraction
        else:
            return (n * middle_d + middle_n, middle_d)    

def skew(vector):

    if isinstance(vector, np.ndarray):
        vector = vector.tolist()

    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])

def contrVar(a,b):
    """
    computes the contravariant vectors. Outputs two 3D vectors    
    """
    ax,ay,az = a
    bx,by,bz = b


    den1 = (ay**2*bx**2+az**2*bx**2-2*ax*ay*bx*by+ax**2*by**2+az**2*by**2-2*ax*az*bx*bz-2*ay*az*by*bz+ax**2*bz**2+ay**2*bz**2)

    tau1p = np.array([  -(ay*bx*by-ax*by**2+az*bx*bz-ax*bz**2),
                        -((-ay)*bx**2+ax*bx*by+az*by*bz-ay*bz**2),
                        -((-az)*bx**2-az*by**2+ax*bx*bz+ay*by*bz)])/den1

    tau2p = np.array([  -((-ay**2)*bx-az**2*bx+ax*ay*by+ax*az*bz),
                        -(ax*ay*bx-ax**2*by-az**2*by+ay*az*bz),
                        -(ax*az*bx+ay*az*by-ax**2*bz-ay**2*bz)])/den1

    den2 = (ax**2*(by**2+bz**2)-2*ay*by*(ax*bx+az*bz)-2*ax*az*bx*bz+ay**2*(bx**2+bz**2)+az**2*(bx**2+by**2))**2

    dtau1pdtau1 = np.array([[-ax**2*(by**2+bz**2)**2+2*ay*by*(ax*bx*(by**2+bz**2)-az*bz*(2*bx**2+by**2+bz**2))+2*ax*az*bx*bz*(by**2+bz**2)+ay**2*(bx**2*(bz**2-by**2)+bz**2*(by**2+bz**2))+az**2*(bx**2*(by**2-bz**2)+by**2*(by**2+bz**2)), by*(ax**2*bx*(by**2+bz**2)+2*ax*az*bz*(by**2+bz**2)-az**2*bx*(bx**2+by**2+2*bz**2))-2*ay*(bx**2+bz**2)*(ax*(by**2+bz**2)-az*bx*bz)+ay**2*bx*by*(bx**2+bz**2),   ax**2*bx*bz*(by**2+bz**2)+2*ay*by*(ax*bz*(by**2+bz**2)+az*bx*(bx**2+by**2))-2*ax*az*(bx**2+by**2)*(by**2+bz**2)-ay**2*bx*bz*(bx**2+2*by**2+bz**2)+az**2*bx*bz*(bx**2+by**2)],
                            [by*(ax**2*bx*(by**2+bz**2)+2*ax*az*bz*(by**2+bz**2)-az**2*bx*(bx**2+by**2+2*bz**2))-2*ay*(bx**2+bz**2)*(ax*(by**2+bz**2)-az*bx*bz)+ay**2*bx*by*(bx**2+bz**2),	ax**2*(bx**2*(bz**2-by**2)+bz**2*(by**2+bz**2))+2*ay*by*(bx**2+bz**2)*(ax*bx+az*bz)-2*ax*az*bx*bz*(bx**2+2*by**2+bz**2)-ay**2*(bx**2+bz**2)**2+az**2*(bx**2*(by**2+bz**2)+bx**4-by**2*bz**2),	by*(ax**2*(-bz)*(2*bx**2+by**2+bz**2)+2*ax*az*bx*(bx**2+by**2)+az**2*bz*(bx**2+by**2))-2*ay*(bx**2+bz**2)*(az*(bx**2+by**2)-ax*bx*bz)+ay**2*by*bz*(bx**2+bz**2)],
                            [ax**2*bx*bz*(by**2+bz**2)+2*ay*by*(ax*bz*(by**2+bz**2)+az*bx*(bx**2+by**2))-2*ax*az*(bx**2+by**2)*(by**2+bz**2)-ay**2*bx*bz*(bx**2+2*by**2+bz**2)+az**2*bx*bz*(bx**2+by**2),	by*(ax**2*(-bz)*(2*bx**2+by**2+bz**2)+2*ax*az*bx*(bx**2+by**2)+az**2*bz*(bx**2+by**2))-2*ay*(bx**2+bz**2)*(az*(bx**2+by**2)-ax*bx*bz)+ay**2*by*bz*(bx**2+bz**2),	ax**2*(bx**2*(by**2-bz**2)+by**2*(by**2+bz**2))-2*ay*by*(ax*bx*(bx**2+by**2+2*bz**2)-az*bz*(bx**2+by**2))+2*ax*az*bx*bz*(bx**2+by**2)+ay**2*(bx**2*(by**2+bz**2)+bx**4-by**2*bz**2)-az**2*(bx**2+by**2)**2]])/den2

    dtau1pdtau2 = np.array([[ay*by*(ax**2*(by**2+bz**2)+az**2*(bx**2-by**2+2*bz**2))+az*(ax**2*bz*(by**2+bz**2)-2*ax*az*bx*(by**2+bz**2)+az**2*bz*(bx**2-by**2))+ay**2*(az*bz*(bx**2+2*by**2-bz**2)-2*ax*bx*(by**2+bz**2))+ay**3*by*(bx**2-bz**2),	ay*(ax**2*bx*(bz**2-by**2)+2*ax*az*bz*(bz**2-by**2)-az**2*bx*(bx**2-by**2+2*bz**2))+2*az*by*(ax**2*(-bx)*bz+ax*az*(bx**2-bz**2)+az**2*bx*bz)+2*ax*ay**2*by*(bx**2+bz**2)+ay**3*(-bx)*(bx**2+bz**2),	-az*(ax**2*bx*(bz**2-by**2)-2*ax*az*bz*(bx**2+by**2)+az**2*bx*(bx**2+by**2))+ay**2*(2*ax*bz*(bx**2-by**2)-az*bx*(bx**2+2*by**2-bz**2))-2*ax*ay*by*(ax*bx*bz+az*(bz**2-by**2))+2*ay**3*bx*by*bz],
                            [2*ax**2*ay*bx*(by**2+bz**2)+ax**3*(-by)*(by**2+bz**2)+ax*(ay**2*by*(bz**2-bx**2)+2*ay*az*bz*(bz**2-bx**2)+az**2*by*(bx**2-by**2-2*bz**2))+2*az*bx*(ay**2*(-by)*bz+ay*az*(by**2-bz**2)+az**2*by*bz),	ax**2*(az*bz*(2*bx**2+by**2-bz**2)-2*ay*by*(bx**2+bz**2))+ax**3*bx*(by**2-bz**2)+ax*bx*(ay**2*(bx**2+bz**2)+az**2*(-bx**2+by**2+2*bz**2))+az*(ay**2*bz*(bx**2+bz**2)-2*ay*az*by*(bx**2+bz**2)+az**2*bz*(by**2-bx**2)),	ax**2*(2*ay*bz*(by**2-bx**2)-az*by*(2*bx**2+by**2-bz**2))+2*ax**3*bx*by*bz-2*ax*ay*bx*(ay*by*bz+az*(bz**2-bx**2))-az*(ay**2*by*(bz**2-bx**2)-2*ay*az*bz*(bx**2+by**2)+az**2*by*(bx**2+by**2))],
                            [2*ax**2*az*bx*(by**2+bz**2)+ax**3*(-bz)*(by**2+bz**2)+ax*(ay**2*bz*(bx**2-2*by**2-bz**2)+2*ay*az*by*(by**2-bx**2)+az**2*bz*(by**2-bx**2))+2*ay*bx*(ay**2*by*bz+ay*az*(bz**2-by**2)-az**2*by*bz),	ax**2*(ay*bz*(-2*bx**2+by**2-bz**2)+2*az*by*(bz**2-bx**2))+2*ax**3*bx*by*bz-2*ax*az*bx*(ay*(by**2-bx**2)+az*by*bz)-ay*(ay**2*bz*(bx**2+bz**2)-2*ay*az*by*(bx**2+bz**2)+az**2*bz*(by**2-bx**2)),	ax**2*(ay*by*(2*bx**2-by**2+bz**2)-2*az*bz*(bx**2+by**2))+ax**3*bx*(bz**2-by**2)+ax*bx*(ay**2*(-bx**2+2*by**2+bz**2)+az**2*(bx**2+by**2))+ay*(ay**2*by*(bz**2-bx**2)-2*ay*az*bz*(bx**2+by**2)+az**2*by*(bx**2+by**2))]])/den2

    dtau2pdtau1 = np.array([[ay*by*(ax**2*(by**2+bz**2)+az**2*(bx**2-by**2+2*bz**2))+az*(ax**2*bz*(by**2+bz**2)-2*ax*az*bx*(by**2+bz**2)+az**2*bz*(bx**2-by**2))+ay**2*(az*bz*(bx**2+2*by**2-bz**2)-2*ax*bx*(by**2+bz**2))+ay**3*by*(bx**2-bz**2),	2*ax**2*ay*bx*(by**2+bz**2)+ax**3*(-by)*(by**2+bz**2)+ax*(ay**2*by*(bz**2-bx**2)+2*ay*az*bz*(bz**2-bx**2)+az**2*by*(bx**2-by**2-2*bz**2))+2*az*bx*(ay**2*(-by)*bz+ay*az*(by**2-bz**2)+az**2*by*bz),	2*ax**2*az*bx*(by**2+bz**2)+ax**3*(-bz)*(by**2+bz**2)+ax*(ay**2*bz*(bx**2-2*by**2-bz**2)+2*ay*az*by*(by**2-bx**2)+az**2*bz*(by**2-bx**2))+2*ay*bx*(ay**2*by*bz+ay*az*(bz**2-by**2)-az**2*by*bz)],
                            [ay*(ax**2*bx*(bz**2-by**2)+2*ax*az*bz*(bz**2-by**2)-az**2*bx*(bx**2-by**2+2*bz**2))+2*az*by*(ax**2*(-bx)*bz+ax*az*(bx**2-bz**2)+az**2*bx*bz)+2*ax*ay**2*by*(bx**2+bz**2)+ay**3*(-bx)*(bx**2+bz**2),	ax**2*(az*bz*(2*bx**2+by**2-bz**2)-2*ay*by*(bx**2+bz**2))+ax**3*bx*(by**2-bz**2)+ax*bx*(ay**2*(bx**2+bz**2)+az**2*(-bx**2+by**2+2*bz**2))+az*(ay**2*bz*(bx**2+bz**2)-2*ay*az*by*(bx**2+bz**2)+az**2*bz*(by**2-bx**2)),	ax**2*(ay*bz*(-2*bx**2+by**2-bz**2)+2*az*by*(bz**2-bx**2))+2*ax**3*bx*by*bz-2*ax*az*bx*(ay*(by**2-bx**2)+az*by*bz)-ay*(ay**2*bz*(bx**2+bz**2)-2*ay*az*by*(bx**2+bz**2)+az**2*bz*(by**2-bx**2))],
                            [-az*(ax**2*bx*(bz**2-by**2)-2*ax*az*bz*(bx**2+by**2)+az**2*bx*(bx**2+by**2))+ay**2*(2*ax*bz*(bx**2-by**2)-az*bx*(bx**2+2*by**2-bz**2))-2*ax*ay*by*(ax*bx*bz+az*(bz**2-by**2))+2*ay**3*bx*by*bz,	ax**2*(2*ay*bz*(by**2-bx**2)-az*by*(2*bx**2+by**2-bz**2))+2*ax**3*bx*by*bz-2*ax*ay*bx*(ay*by*bz+az*(bz**2-bx**2))-az*(ay**2*by*(bz**2-bx**2)-2*ay*az*bz*(bx**2+by**2)+az**2*by*(bx**2+by**2)),	ax**2*(ay*by*(2*bx**2-by**2+bz**2)-2*az*bz*(bx**2+by**2))+ax**3*bx*(bz**2-by**2)+ax*bx*(ay**2*(-bx**2+2*by**2+bz**2)+az**2*(bx**2+by**2))+ay*(ay**2*by*(bz**2-bx**2)-2*ay*az*bz*(bx**2+by**2)+az**2*by*(bx**2+by**2))]])/den2

    dtau2pdtau2 = np.array([[ay**2*(ax**2*(bz**2-by**2)+2*ax*az*bx*bz+az**2*(-2*bx**2+by**2+bz**2))-2*ay*az*by*(2*ax**2*bz-ax*az*bx+az**2*bz)+az**2*(ax**2*(by**2-bz**2)+2*ax*az*bx*bz+az**2*(by**2-bx**2))+2*ay**3*by*(ax*bx-az*bz)+ay**4*(bz**2-bx**2),	-2*ax**2*bx*by*(ay**2+az**2)+ax**3*(ay*(by**2-bz**2)+2*az*by*bz)+ax*(ay**3*(bx**2-bz**2)+ay*az**2*(bx**2+by**2-2*bz**2)+2*az**3*by*bz)+2*az*bx*(ay**2+az**2)*(ay*bz-az*by),	-2*ax**2*bx*bz*(ay**2+az**2)+ax**3*(2*ay*by*bz+az*(bz**2-by**2))+ax*(ay**2*az*(bx**2-2*by**2+bz**2)+2*ay**3*by*bz+az**3*(bx**2-by**2))-2*ay*bx*(ay**2+az**2)*(ay*bz-az*by)],
                            [-2*ax**2*bx*by*(ay**2+az**2)+ax**3*(ay*(by**2-bz**2)+2*az*by*bz)+ax*(ay**3*(bx**2-bz**2)+ay*az**2*(bx**2+by**2-2*bz**2)+2*az**3*by*bz)+2*az*bx*(ay**2+az**2)*(ay*bz-az*by),	ax**2*(ay**2*(bz**2-bx**2)+2*ay*az*by*bz+az**2*(bx**2-2*by**2+bz**2))+2*ax**3*bx*(ay*by-az*bz)+ax**4*(bz**2-by**2)-2*ax*az*bx*(2*ay**2*bz-ay*az*by+az**2*bz)+az**2*(ay**2*(bx**2-bz**2)+2*ay*az*by*bz+az**2*(bx**2-by**2)),	ax**2*(-2*ay**2*by*bz+ay*az*(-2*bx**2+by**2+bz**2)-2*az**2*by*bz)+2*ax**3*bx*(ay*bz+az*by)-2*ax**4*by*bz+2*ax*bx*(ay**3*bz+az**3*by)+ay*az*(ay**2*(bz**2-bx**2)-2*ay*az*by*bz+az**2*(by**2-bx**2))],
                            [-2*ax**2*bx*bz*(ay**2+az**2)+ax**3*(2*ay*by*bz+az*(bz**2-by**2))+ax*(ay**2*az*(bx**2-2*by**2+bz**2)+2*ay**3*by*bz+az**3*(bx**2-by**2))-2*ay*bx*(ay**2+az**2)*(ay*bz-az*by),	ax**2*(-2*ay**2*by*bz+ay*az*(-2*bx**2+by**2+bz**2)-2*az**2*by*bz)+2*ax**3*bx*(ay*bz+az*by)-2*ax**4*by*bz+2*ax*bx*(ay**3*bz+az**3*by)+ay*az*(ay**2*(bz**2-bx**2)-2*ay*az*by*bz+az**2*(by**2-bx**2)),	ax**2*(ay**2*(bx**2+by**2-2*bz**2)+2*ay*az*by*bz+az**2*(by**2-bx**2))+ax**3*(2*az*bx*bz-2*ay*bx*by)+ax**4*(by**2-bz**2)-2*ax*ay*bx*(ay**2*by-ay*az*bz+2*az**2*by)+ay**2*(ay**2*(bx**2-bz**2)+2*ay*az*by*bz+az**2*(bx**2-by**2))]])/den2

    return np.array([tau1p,tau2p]).T


def plot_coords(ax,orig=(0.0,0.0,0.0),labels=['x','y','z']):
    x, y, z = np.array(orig)
    # I want to plot 3 arrows, each of them starting from the origin and pointing in directions x, y and z
    dl = 1
    objs = []
    objs.append(ax.quiver(x, y, z,dl,0,0))
    objs.append(ax.quiver(x, y, z,0,dl,0))
    objs.append(ax.quiver(x, y, z,0,0,dl))
    objs.append(ax.text(x+dl,y,z,labels[0]))
    objs.append(ax.text(x,y+dl,z,labels[1]))
    objs.append(ax.text(x,y,z+dl,labels[2]))

    return objs

def dnBernstein(n,k,x,p):
    coef = factorial(n)/factorial(n-p)
    desde = max(0,k+p-n)
    hasta = min(k,p)
    dnB = coef*sum([(-1)**(i+p)*comb(p,i)*Bernstein(n-p,k-i,x) for i in range(desde,hasta+1)])
    return dnB

def Unchain(Inputlist,exclude=[]):
    if len(Inputlist)<=1:
        return []
    # elif len(Inputlist) == 3:
    elif len(Inputlist) in [2,3]:           # Changed for 2d surface. points at the edges (quad: 0  ,  node: 1)
        for pair in Inputlist:
            if set(pair)!=set(exclude):
                for elem in pair:
                    if elem not in exclude:
                        return [elem]
    Cleanlist = []
    for pair in Inputlist:
        if pair[0] not in exclude and pair[1] not in exclude:
            Cleanlist.append(pair)
    sortedList = Cleanlist.pop(0)
    
    # if len(Cleanlist)!=0:
    #     sortedList = Cleanlist.pop(0)
    # else:
    #     sortedList = []
 
    i=0
    while len(Cleanlist)>0:
        if i >=len(Cleanlist):
            i=0
        if Cleanlist[i][1] == sortedList[0]:
            sortedList.insert(0,Cleanlist[i][0])
            del Cleanlist[i]
            i=0
        elif Cleanlist[i][0] == sortedList[-1]:
            sortedList.append([Cleanlist[i][1]])
            del Cleanlist[i]
            i=0
        i+=1
    return sortedList

def flatmesh(x,y,z,dv1,dv2,nx,ny):
    import numpy as np
    quads = []
    XYZ = []
    for j in range(ny+1):
        for i in range(nx+1):
            if i < nx and j < ny:
                quad = [(nx+1)*j+i , (nx+1)*j+i+1 , (nx+1)*(j+1)+i+1 , (nx+1)*(j+1)+i]
                quads.append(quad)
            dx, dy, dz = i/(nx)*dv1 + j/(ny)*dv2
            xyz = [x+dx,y+dy,z+dz]
            XYZ.append(xyz)

    return np.array(XYZ) , quads

# def scipy_to_petsc(A):
#     """Converts SciPy CSR matrix to PETSc serial matrix."""
#     nrows = A.shape[0]
#     ncols = A.shape[1]

#     ai, aj, av = A.indptr, A.indices, A.data
#     mat = PETSc.Mat()
#     mat.createAIJWithArrays(size=(nrows, ncols), csr=(ai, aj, av))
#     mat.assemble()

#     return mat

def checkVarsSize():
    import sys
    def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                            key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))







###############
### BACKUPS ###
###############

"""        def BS_Reduction( self, ii, jj, list1 = "ALL", list2= "ALL" ):       # Recursive reduction function for GlobalBS pairs


            body1, body2 = self.bodies[ii], self.bodies[jj]

            if list1 =="ALL" and list2 == "ALL":
                list1 = range(len(body1.patches))
                list2 = range(len(body2.patches)) 

            # Creates (sub)body sphere BSB
            xmin, ymin, zmin = body2.patches[list2[0]].BS.x
            xmax, ymax, zmax = body2.patches[list2[0]].BS.x
            for idx2 in list2:
                BSj = body2.patches[idx2].BS
                ri, xi = BSj.r, BSj.x

                if xi[0]-ri<xmin:
                    xmin = xi[0] - ri
                elif xi[0]+ri>xmax:
                    xmax = xi[0] + ri
                if xi[1]-ri<ymin:
                    ymin = xi[1] - ri
                elif xi[1]+ri>ymax:
                    ymax = xi[1] + ri
                if xi[2]-ri<zmin:
                    zmin = xi[2] - ri
                elif xi[2]+ri>zmax:
                    zmax = xi[2] + ri

            xm = 0.5*np.array( [ xmin+xmax , ymin+ymax, zmin+zmax ] )
            r  = 0.5*sqrt( (xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2 )
            BSB = BoundingSphere(xm,r)

            # Finds patch spheres of body 1 in BSB
            newlist1 = []
            for idx1 in list1:
                if body1.patches[idx1].BS.Collision(BSB):
                    newlist1.append(idx1)

            # Creates (sub)body sphere BSA
            xmin, ymin, zmin = body1.patches[newlist1[0]].BS.x
            xmax, ymax, zmax = body1.patches[newlist1[0]].BS.x
            for idx1 in newlist1:
                BSi = body1.patches[idx1].BS
                ri, xi = BSi.r, BSi.x

                if xi[0]-ri<xmin:
                    xmin = xi[0] - ri
                elif xi[0]+ri>xmax:
                    xmax = xi[0] + ri
                if xi[1]-ri<ymin:
                    ymin = xi[1] - ri
                elif xi[1]+ri>ymax:
                    ymax = xi[1] + ri
                if xi[2]-ri<zmin:
                    zmin = xi[2] - ri
                elif xi[2]+ri>zmax:
                    zmax = xi[2] + ri

            xm = 0.5*np.array( [ xmin+xmax , ymin+ymax, zmin+zmax ] )
            r  = 0.5*sqrt( (xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2 )
            BSA = BoundingSphere(xm,r)

            # Finds patch spheres of body 2 in BSA
            newlist2 = []
            for idx2 in list2:
                if body2.patches[idx2].BS.Collision(BSA):
                    newlist2.append(idx2)


            # Recursion:
            if set(newlist1) == set(list1) and set(newlist2) == set(list2):
                return newlist1, newlist2
            else:
                return self.BS_Reduction(ii,jj,list1=newlist1,list2=newlist2)

"""
