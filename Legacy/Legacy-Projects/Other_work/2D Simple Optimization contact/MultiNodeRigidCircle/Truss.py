from pdb import set_trace
import numpy as np

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

# chatGPTversion
def double_braced_system(nx, ny, dx, dy):
    nodes = np.arange(1, nx * ny * 4 + 1).reshape(nx * ny, 4)
    pos = np.zeros((nx * ny * 4, 2))
    for i in range(nx):
        for j in range(ny):
            # pos[nodes[i * ny + j, :]-1, 0] = [i * dx, j * dy]
            pos[nodes[i * ny + j, :2]-1, :] = [i * dx, j * dy]
            pos[nodes[i * ny + j, :2]-1, :] = [(i + 1) * dx, j * dy]
            pos[nodes[i * ny + j, :2]-1, :] = [(i + 1) * dx, (j + 1) * dy]
            pos[nodes[i * ny + j, :2]-1, :] = [i * dx, (j + 1) * dy]

    bars = set()
    for i in range(nx):
        for j in range(ny):
            if i < nx - 1:
                bars.add(frozenset([nodes[i * ny + j, 0], nodes[i * ny + j, 1]]))
                bars.add(frozenset([nodes[i * ny + j, 2], nodes[i * ny + j, 3]]))
            if j < ny - 1:
                bars.add(frozenset([nodes[i * ny + j, 1], nodes[i * ny + j, 2]]))
                bars.add(frozenset([nodes[i * ny + j, 0], nodes[i * ny + j, 3]]))
            if i < nx - 1 and j < ny - 1:
                bars.add(frozenset([nodes[i * ny + j, 0], nodes[i * ny + j + ny, 2]]))
                bars.add(frozenset([nodes[i * ny + j, 2], nodes[i * ny + j + ny, 0]]))
    
    conn = np.zeros((len(bars), 2))
    for i, bar in enumerate(bars):
        conn[i, :] = list(bar)
        
    return conn.astype(int), pos

    

nx = 10
ny = 6
dx = 2
dy = 2

X, C = Truss(nx,ny,dx,dy)
Cgpt, Xgpt = double_braced_system(nx,ny,dx,dy)


plotTruss(X,C, show_nodes=True,show_id=True)
plotTruss(Xgpt,Cgpt, show_nodes=True,show_id=True)


set_trace()


