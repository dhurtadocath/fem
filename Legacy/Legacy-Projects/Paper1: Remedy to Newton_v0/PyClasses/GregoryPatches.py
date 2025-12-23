# from numpy.core.numeric import tensordot
from typing import Set
from numpy.core.numeric import outer
from PyClasses.Utilities import Bernstein, dnBernstein, Unchain, printif, flatList, skew, contrVar
import numpy as np
from numpy.linalg import norm
import time,csv

from pdb import set_trace


class GrgPatch:
    def __init__(self,surf,iquad, masterNodes = False):
        # GLOBALS
        self.surf = surf
        self.iquad = iquad
        self.quad  = surf.quads[iquad]
        # LOCALS
        self.masterNodes = masterNodes
        self.getLocals()    #attrs: squad, facets, num_outnodes, sorrounding_nodes
        self.X = self.surf.X[self.squad]
        self.eps = 1e-11             # treatment for undefined cases

    def getLocals(self):
        quad = self.quad
        superquad = quad[:]
        num_outnodes = []
        facets = []
        sorrounding_nodes = []
        for nid, node in enumerate(quad):
            loc_id = self.surf.nodes.index(node)    # local id of the node in the surface
            neighbors = self.surf.NeighborNodes[loc_id]
            outerneighbors = Unchain(neighbors,exclude = [quad[nid-1],quad[nid-3]])
            if self.masterNodes != False:   # No C1 continuity out of subsurf
                sorrounding_nodes.append( [node for node in outerneighbors if node in self.masterNodes] )
            else:                           # C1 continuity out of subsurf
                sorrounding_nodes.append( outerneighbors )
            superquad.extend(sorrounding_nodes[nid])
            facets.append([[superquad.index(nodi) for nodi in pair] for pair in neighbors if (pair[0] in superquad and pair[1] in superquad)])
            num_outnodes.append(len(sorrounding_nodes[nid]))
            
        self.sorrounding_nodes = sorrounding_nodes
        self.squad = superquad
        self.facets = facets
        self.num_outnodes = num_outnodes

    def Y0Y3(self,side_idx):
        CP = self.CtrlPts
        x1 = [[0,0],[3,0],[3,3],[0,3]][side_idx]
        x2 = [[3,0],[3,3],[0,3],[0,0]][side_idx]
        return CP[x1[0]][x1[1]] , CP[x2[0]][x2[1]]

    def Y1Y2(self,side_idx):
        CP = self.CtrlPts
        x1 = [[1,0],[3,1],[2,3],[0,2]][side_idx]
        x2 = [[2,0],[3,2],[1,3],[0,1]][side_idx]
        return CP[x1[0]][x1[1]] , CP[x2[0]][x2[1]]

    def Yp0Yp3(self,side_idx):
        return self.Y1Y2(side_idx-1)[1], self.Y1Y2(side_idx-3)[0]


    def getNormals(self,u, compute_deriv=True):
        Ne = len(self.squad)
        # normals = []
        # W = []
        # dW = []
        normals = np.empty((4,3),dtype=float)
        W       = np.empty((4,3,3*Ne),dtype=float)
        dW      = np.empty((4,3,3*Ne,3*Ne),dtype=float)

        for idxm in range(4):
            nodeid = self.quad[idxm]
            Wi = np.zeros( ( 3 , 3*Ne ) )
            dWi= np.zeros( ( 3 , 3*Ne , 3*Ne ) )
            # Xi = self.surf.X[nodeid]
            # ui = u[self.surf.DoFs[nodeid]]
            Xi = self.X[idxm]
            ui = u[self.surf.body.DoFs[nodeid]]
            ntmp = 0
            n0mp = 0
            ntm = 0
            
            for pair in self.facets[idxm]:

                Xna, Xnb = [self.X[i] for i in pair]
                una, unb = [     u[self.surf.body.DoFs[self.squad[i]]] for i in pair]

                Xa, Xb = (Xna)-(Xi) , (Xnb)-(Xi)
                xa, xb = Xa + una - ui , Xb + unb - ui

                ntm_e = np.cross(xb,xa)
                n0m_e = np.cross(Xb,Xa)
                w0m_e = norm(n0m_e)

                ntmp += ntm_e / w0m_e
                n0mp += n0m_e / w0m_e

                # set_trace()
                if compute_deriv:
                    idx1, idx2 = pair
                    
                    w2m = wij(Xnb+unb,Xi+ui)
                    w1m = wij(Xna+una,Xi+ui)
                    w12 = wij(Xna+una,Xnb+unb)

                    Wi[ : , 3*idx1:3*idx1+3] += w2m/w0m_e
                    Wi[ : , 3*idx2:3*idx2+3] -= w1m/w0m_e
                    Wi[ : , 3*idxm:3*idxm+3] += w12/w0m_e

                    # TODO: Below, use sparse matrices instead (almost only zeros)

                    dwij = np.array([ [[0.0,0.0,0.0],[0.0,0.0,-1.0],[0.0,1.0,0.0]] , [[0.0,0.0,1.0],[0.0,0.0,0.0],[-1.0,0.0,0.0]] , [[0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,0.0]] ])/w0m_e

                    dWi[ : , 3*idx1:3*idx1+3 , 3*idx2:3*idx2+3] += dwij
                    dWi[ : , 3*idx1:3*idx1+3 , 3*idxm:3*idxm+3] -= dwij
                    dWi[ : , 3*idx2:3*idx2+3 , 3*idx1:3*idx1+3] -= dwij
                    dWi[ : , 3*idx2:3*idx2+3 , 3*idxm:3*idxm+3] += dwij
                    dWi[ : , 3*idxm:3*idxm+3 , 3*idx1:3*idx1+3] += dwij
                    dWi[ : , 3*idxm:3*idxm+3 , 3*idx2:3*idx2+3] -= dwij

            wm = norm( n0mp )
            
            ntm = ntmp / wm
            # normals.append(ntm)
            normals[idxm] = ntm

            if compute_deriv:
                Wi = Wi/wm
                dWi = dWi/wm
                # W.append(Wi)
                # dW.append(dWi)
                W [idxm] = Wi
                dW[idxm] = dWi

        self.normals = normals
        self.W = W
        self.dW = dW                                  # if not computed, W = []

    def getCtrlPts(self, u, compute_deriv=True, BS_update=True):


        self.getNormals(u, compute_deriv=compute_deriv)
        
        X, u = self.X, u[self.surf.body.DoFs[self.squad]]

        start=[]
        edge=[]
        inner = []

        # going through every edge of the patch to find edge points
        for idx in range(4):
            y0, y3 = X[range(4)[idx-1]]+u[range(4)[idx-1]] , X[idx]+u[idx]
            m0, m3 = self.normals[idx-1] , self.normals[idx]

            c0 = 1/3*((y3-y0)-np.matmul((y3-y0),np.outer(m0,m0)/np.dot(m0,m0)))
            c2 = 1/3*((y3-y0)-np.matmul((y3-y0),np.outer(m3,m3)/np.dot(m3,m3)))
            y1, y2 = y0+c0, y3-c2

            start.append(y0)
            edge.append([y1,y2])

        # going through every edge of the patch to find inner points
        for idx in range(4):
            y0, y3 = X[range(4)[idx-1]]+u[range(4)[idx-1]] , X[idx]+u[idx]
            m0, m3 = self.normals[idx-1] , self.normals[idx]
            
            y1, y2 = edge[idx][0], edge[idx][1]
            y0p , y3p = edge[idx-1][1], edge[idx-3][0]

            c0, c1, c2 = y1-y0, y2-y1, y3-y2
            b0, b3 = y0p-y0 , y3p-y3

            a0 = np.cross(m0,(y3-y0))/norm( np.cross(m0,(y3-y0)) )
            a3 = np.cross(m3,(y3-y0))/norm( np.cross(m3,(y3-y0)) )

            k0, k1 = np.dot(a0,b0), np.dot(a3,b3)
            h0, h1 = np.dot(c0,b0)/np.dot(c0,c0) , np.dot(c2,b3)/np.dot(c2,c2)

            b1 = 1/3*((k1+k0)*a0+k0*a3+2*h0*c1+h1*c0)
            b2 = 1/3*((k1+k0)*a3+k1*a0+2*h1*c1+h0*c2)

            y1p, y2p = y1+b1 , y2+b2

            inner.append([y1p,y2p])

        # Assembly of CtrlPts matrix of the patch
        row0 = [ start[1]  ,         edge[0][1]        ,         edge[0][0]        ,  start[0]  ]     #This "matrix" HAS to be transposed (for indexing purposes)
        row1 = [edge[1][0] , [inner[1][0],inner[0][1]] , [inner[3][1],inner[0][0]] , edge[3][1] ]
        row2 = [edge[1][1] , [inner[1][1],inner[2][0]] , [inner[3][0],inner[2][1]] , edge[3][0] ]
        row3 = [ start[2]  ,         edge[2][0]        ,         edge[2][1]        ,  start[3]  ]

        self.CtrlPts = [row0,row1,row2,row3]

        from PyClasses.BoundingSpheres import BS
        if hasattr(self, "BS"):
            self.BS.x, self.BS.r = self.BS.wrapPatch(self, method = "avgXmaxR")
        else:
            # set_trace()
            self.BS = BS(self, eps=0.15)


    def flatCtrlPts(self):
        cp = self.CtrlPts
        CtrlPts = [
            cp[0][0], cp[3][0], cp[3][3], cp[0][3],
            cp[1][0], cp[2][0], cp[3][1], cp[3][2],cp[2][3], cp[1][3], cp[0][2], cp[0][1],
            cp[1][1][0], cp[1][1][1], cp[2][1][0], cp[2][1][1],cp[2][2][0], cp[2][2][1], cp[1][2][0], cp[1][2][1],
        ]

        return CtrlPts

    def groupCtrlPts(self, cp = 0):

        if type(cp) == int:
            cp = self.CtrlPts
        row0 = [ cp[0] ,      cp[11]     ,       cp[10]    , cp[3] ]            #This "matrix" HAS to be transposed (for indexing purposes)
        row1 = [ cp[4] , [cp[12],cp[13]] , [cp[18],cp[19]] , cp[9] ]
        row2 = [ cp[5] , [cp[14],cp[15]] , [cp[16],cp[17]] , cp[8] ]
        row3 = [ cp[1] ,       cp[6]     ,       cp[7]     , cp[2] ]
        
        ctrls = [row0,row1,row2,row3]
        self.CtrlPts = ctrls


    def Grg(self, t, deriv = 0):                        # Normalized normal vector at (u,v) with treatment for undefinition at nodes
        u,v = t
        p       = np.array([0.0 , 0.0 , 0.0])
        D1p     = np.array([0.0 , 0.0 , 0.0])
        D2p     = np.array([0.0 , 0.0 , 0.0])
        D1D1p   = np.array([0.0 , 0.0 , 0.0])
        D1D2p   = np.array([0.0 , 0.0 , 0.0])
        D2D2p   = np.array([0.0 , 0.0 , 0.0])
        D1D1D1p = np.array([0.0 , 0.0 , 0.0])
        D1D1D2p = np.array([0.0 , 0.0 , 0.0])
        D1D2D2p = np.array([0.0 , 0.0 , 0.0])
        D2D2D2p = np.array([0.0 , 0.0 , 0.0])
        n, m = len(self.CtrlPts)-1, len(self.CtrlPts[0])-1

        for i  in range(n+1):
            for j in range(m+1):
                # Inner nodes: values, derivatives and treatments
                if i in [1,2] and j in [1,2]:
                    if i==1 and j ==1:
                        x110, x111 = self.CtrlPts[1][1]
                        den = max(self.eps,u+v)  
                        xij = (u*x110+v*x111)/(den)
                        if deriv > 0:
                            D1xij = x110/(den) - (u*x110 + v*x111)/((den)**2)
                            D2xij = x111/(den) - (u*x110 + v*x111)/((den)**2)
                        if deriv > 1:
                            D11xij =-2*v*(x110 - x111)/(den)**3
                            D12xij = (u - v)*(x110 - x111)/(den)**3
                            D22xij = 2*u*(x110 - x111)/(den)**3
                        if deriv > 2:
                            D111xij = 6*v*(x110-x111)/(den)**4
                            D112xij = -(2*(u-2*v)*(x110-x111)/(den)**4)
                            D122xij = -(2*(2*u-v)*(x110-x111)/(den)**4)
                            D222xij = -(6*u*(x110-x111)/(den)**4)
                            
                    elif i==1 and j==2:
                        x120, x121 = self.CtrlPts[1][2]
                        den = max(self.eps,u+1-v)
                        xij = (u*x120+(1-v)*x121)/(den)
                        if deriv > 0:
                            D1xij = x120/(den) - (u*x120 + (1-v)*x121)/((den)**2)
                            D2xij = -x121/(den) + (u*x120 + (1-v)*x121)/((den)**2)
                        if deriv > 1:
                            D11xij = (2*(-1 + v)*(x120 - x121))/(den)**3
                            D12xij = -(((-1 + u + v)*(x120 - x121))/(den)**3)
                            D22xij = (2*u*(x120 - x121))/(den)**3
                        if deriv > 2:
                            D111xij = -(6*(-1+v)*(x120-x121)/(den)**4)
                            D112xij = (2*(-2+u+2*v)*(x120-x121)/(den)**4)
                            D122xij = -(2*(-1+2*u+v)*(x120-x121)/(den)**4)
                            D222xij = (6*u*(x120-x121)/(den)**4)

                    elif i==2 and j==1:
                        x210, x211 = self.CtrlPts[2][1]
                        den = max(self.eps, v+1-u)
                        xij = ((1-u)*x210+v*x211)/(den)
                        if deriv > 0:
                            D1xij = -x210/(den) + ((1-u)*x210 + v*x211)/((den)**2)
                            D2xij = x211/(den) - ((1-u)*x210 + v*x211)/((den)**2)
                        if deriv > 1:
                            D11xij = (2*v*(x210 - x211))/(-den)**3
                            D12xij = ((-1 + u + v)*(x210 - x211))/(den)**3
                            D22xij = (2*(-1 + u)*(x210 - x211))/(-den)**3
                        if deriv > 2:
                            D111xij = -(6*v*(x210-x211)/(den)**4)
                            D112xij = (2*(-1+u+2*v)*(x210-x211)/(den)**4)
                            D122xij = -(2*(-2+2*u+v)*(x210-x211)/(den)**4)
                            D222xij = (6*(-1+u)*(x210-x211)/(den)**4)

                    else:
                        x220, x221 = self.CtrlPts[2][2]
                        den = max(self.eps, 2-u-v)
                        xij = ((1-u)*x220+(1-v)*x221)/(den)
                        if deriv > 0:
                            D1xij = -x220/(den) + ((1-u)*x220 + (1-v)*x221)/((den)**2)
                            D2xij = -x221/(den) + ((1-u)*x220 + (1-v)*x221)/((den)**2)
                        if deriv > 1:
                            D11xij = -((2*(-1 + v)*(x220 - x221))/(-den)**3)
                            D12xij = ((u - v)*(x220 - x221))/(-den)**3
                            D22xij = (2*(-1 + u)*(x220 - x221))/(-den)**3
                        if deriv > 2:
                            D111xij = 6*(-1+v)*(x220-x221)/(-den)**4
                            D112xij = -(2*(1+u-2*v)*(x220-x221)/(-den)**4)
                            D122xij = -(2*(-1+2*u-v)*(x220-x221)/(-den)**4)
                            D222xij = -(6*(-1+u)*(x220-x221)/(-den)**4)

                else:
                    xij = self.CtrlPts[i][j]
                    D1xij, D2xij = 0.0 , 0.0
                    D11xij, D12xij, D22xij = 0.0 , 0.0 , 0.0
                    D111xij, D112xij, D122xij, D222xij = 0.0 , 0.0 , 0.0 , 0.0

                # Bernstein polynomials
                Bi     =   Bernstein(n, i, u)
                Bj     =   Bernstein(m, j, v)

                p += Bi*Bj*xij

                # Tangent Derivatives w/r to LOCAL parameters
                if deriv > 0:
                    D1Bi     =  dnBernstein(n, i, u, 1)
                    D2Bj     =  dnBernstein(m, j, v, 1)
                    D1p += D1Bi*Bj*xij + Bi*Bj*D1xij
                    D2p += Bi*D2Bj*xij + Bi*Bj*D2xij

                if deriv > 1:
                    DD1Bi = dnBernstein(n, i, u, 2)
                    DD2Bj = dnBernstein(m, j, v, 2)
                    D1D1p += (DD1Bi*xij + 2*D1Bi*D1xij + Bi*D11xij)*Bj
                    D1D2p += D1Bi*D2Bj*xij + D1Bi*Bj*D2xij + Bi*D2Bj*D1xij + Bi*Bj*D12xij
                    D2D2p += (DD2Bj*xij + 2*D2Bj*D2xij + Bj*D22xij)*Bi

                if deriv > 2:
                    DDD1Bi = dnBernstein(n, i, u, 3)
                    DDD2Bj = dnBernstein(m, j, v, 3)
                    D1D1D1p += (DDD1Bi*xij + 3*DD1Bi*D1xij + 3*D1Bi*D11xij + Bi*D111xij)*Bj
                    D1D1D2p += (DD1Bi*D2xij + 2*D1Bi*D12xij + Bi*D112xij)*Bj + (DD1Bi*xij + 2*D1Bi*D1xij + Bi*D11xij)*D2Bj
                    D1D2D2p += (DD2Bj*D1xij + 2*D2Bj*D12xij + Bj*D122xij)*Bi + (DD2Bj*xij + 2*D2Bj*D2xij + Bj*D22xij)*D1Bi
                    D2D2D2p += (DDD2Bj*xij + 3*DD2Bj*D2xij + 3*D2Bj*D22xij + Bj*D222xij)*Bi

        if deriv==0:
            return p
        if deriv==1:
            return p, np.array([D1p, D2p]).T
        elif deriv == 2:
            return p, np.array([D1p, D2p]).T, np.array([[D1D1p, D1D2p], [D1D2p, D2D2p]]).T
        elif deriv == 3:
            return p, np.array([D1p, D2p]).T, np.array([[D1D1p, D1D2p], [D1D2p, D2D2p]]).T, np.array([[[D1D1D1p,D1D1D2p],[D1D1D2p,D1D2D2p]], [[D1D1D2p,D1D2D2p],[D1D2D2p,D2D2D2p]]]).T
        else:
            print("Derivative order not (yet) implemented")
            set_trace()

    def Grg0(self, t):                        # Normalized normal vector at (u,v) with treatment for undefinition at nodes
        u,v = t
        p       = np.array([0.0 , 0.0 , 0.0])
        n, m = len(self.CtrlPts)-1, len(self.CtrlPts[0])-1

        for i  in range(n+1):
            for j in range(m+1):
                # Inner nodes: values, derivatives and treatments
                if i in [1,2] and j in [1,2]:
                    if i==1 and j ==1:
                        x110, x111 = self.CtrlPts[1][1]
                        den = max(self.eps,u+v)  
                        xij = (u*x110+v*x111)/(den)
                            
                    elif i==1 and j==2:
                        x120, x121 = self.CtrlPts[1][2]
                        den = max(self.eps,u+1-v)
                        xij = (u*x120+(1-v)*x121)/(den)

                    elif i==2 and j==1:
                        x210, x211 = self.CtrlPts[2][1]
                        den = max(self.eps, v+1-u)
                        xij = ((1-u)*x210+v*x211)/(den)

                    else:
                        x220, x221 = self.CtrlPts[2][2]
                        den = max(self.eps, 2-u-v)
                        xij = ((1-u)*x220+(1-v)*x221)/(den)

                else:
                    xij = self.CtrlPts[i][j]

                # Bernstein polynomials
                Bi     =   Bernstein(n, i, u)
                Bj     =   Bernstein(m, j, v)

                p += Bi*Bj*xij


        return p

    def D3Grg(self,t, normalize = True):                        # Normalized normal vector at (u,v) with treatment for undefinition at nodes
        D1p, D2p = self.Grg(t, deriv = 1)[1].T
        D3p = np.cross(D1p,D2p)
        if norm(D3p) ==0:
            set_trace()
        if normalize:
            D3p = D3p/norm(D3p)       # The NORMALIZED vectors are continuous from patch to patch (doesnt make sense otherwise)
        return D3p

    def dndxi(self,t,degree=1):
        _, dxcdt= self.Grg(t, deriv = 1)
        tau1, tau2 = dxcdt.T
        N = np.cross(tau1,tau2)
        normN = norm(N)
        dndN   = np.eye(3)/normN - np.outer(N,N)/(normN**3)
        dNdtau1 =-skew(tau2)        # signs verified with Autograd
        dNdtau2 = skew(tau1)        # signs verified with Autograd
        Dtau1Dxi, Dtau2Dxi  = self.d2xcdxidt(t).T       # shape (20,) each
        dtau1dxi = np.outer(np.eye(3),Dtau1Dxi).reshape(3,3,20)     # shape (3,3,20)
        dtau2dxi = np.outer(np.eye(3),Dtau2Dxi).reshape(3,3,20)
        
        dNdxi = np.tensordot(dNdtau1,dtau1dxi,axes=[1,0]) + np.tensordot(dNdtau2,dtau2dxi,axes=[1,0])             # shape (3,3,20)
        dndxi = np.tensordot(dndN,dNdxi,axes=[1,0])

        if degree==1:
            return dndxi.swapaxes(0,2).swapaxes(1,2)   #returns shape (20,3,3)

        else:
            NI = np.multiply.outer(N,np.eye(3))
            d2nd2N = 3*np.multiply.outer(np.outer(N,N),N)/normN**5 - (NI+NI.swapaxes(0,1)+NI.swapaxes(0,2))/normN**3
            
            # check symmetry 
            d2Ndtau1dtau2 = -np.array([ [[0.0,0.0,0.0],[0.0,0.0,-1.0],[0.0,1.0,0.0]] , [[0.0,0.0,1.0],[0.0,0.0,0.0],[-1.0,0.0,0.0]] , [[0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,0.0]] ])
            d2Ndtau2dtau1 = -d2Ndtau1dtau2
            d2Ndxidxj = (np.tensordot(np.tensordot(d2Ndtau1dtau2,dtau1dxi,axes=[1,0]),dtau2dxi,axes=[1,0]) + np.tensordot(np.tensordot(d2Ndtau2dtau1,dtau2dxi,axes=[1,0]),dtau1dxi,axes=[1,0])).swapaxes(2,3)   #shape (3,3,3,20,20)
            d2ndxidxj = np.tensordot(np.tensordot(d2nd2N,dNdxi,axes=[1,0]),dNdxi,axes=[1,0]).swapaxes(2,3)+np.tensordot(dndN,d2Ndxidxj,axes=[1,0])
            
            return dndxi.swapaxes(0,2).swapaxes(1,2), d2ndxidxj   #returns shapes (20,3,3), (20,20,3,3,3)

    def dndt(self,t,degree=1):
        if degree==1:
            _, dxcdt,d2xcdt2 = self.Grg(t, deriv = 2)
        elif degree==2:
            _, dxcdt,d2xcdt2,d3xcdt3 = self.Grg(t, deriv = 3)

        tau1, tau2 = dxcdt.T
        N = np.cross(tau1,tau2)
        normN = norm(N)
        nor = N/normN
        dndN   = (np.eye(3) - np.outer(nor,nor))/normN
        dNdtau1 =-skew(tau2)        # signs verified with Autograd
        dNdtau2 = skew(tau1)        # signs verified with Autograd
        dtau1dt = d2xcdt2[:,:,0]
        dtau2dt = d2xcdt2[:,:,1]

        dNdt = dNdtau1@dtau1dt + dNdtau2@dtau2dt
        dndt = dndN@dNdt


        if degree==1:
            return dndt   #returns shape (20,3,3)

        elif degree==2:
            d2tau1d2t = d3xcdt3[:,:,:,0]
            d2tau2d2t = d3xcdt3[:,:,:,1]
            NI = np.multiply.outer(N,np.eye(3))
            d2nd2N = 3*np.multiply.outer(np.outer(N,N),N)/normN**5 - (NI+NI.swapaxes(0,1)+NI.swapaxes(0,2))/normN**3
            
            # check symmetry 
            d2Ndtau1dtau2 = -np.array([ [[0.0,0.0,0.0],[0.0,0.0,-1.0],[0.0,1.0,0.0]] , [[0.0,0.0,1.0],[0.0,0.0,0.0],[-1.0,0.0,0.0]] , [[0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,0.0]] ])
            d2Ndtau2dtau1 = -d2Ndtau1dtau2

            d2Nd2t = (np.tensordot(np.tensordot(d2Ndtau1dtau2,dtau1dt,axes=[1,0]),dtau2dt,axes=[1,0])                   \
                    + np.tensordot(np.tensordot(d2Ndtau2dtau1,dtau2dt,axes=[1,0]),dtau1dt,axes=[1,0])).swapaxes(1,2)    \
                    + np.tensordot(dNdtau1,d2tau1d2t,axes=[1,0])                                                        \
                    + np.tensordot(dNdtau2,d2tau2d2t,axes=[1,0])                                                        #shape (3,3,3,20,20) 
            
            d2nd2t = np.tensordot(np.tensordot(d2nd2N,dNdt,axes=[1,0]),dNdt,axes=[1,0]).swapaxes(1,2) + np.tensordot(dndN,d2Nd2t,axes=[1,0])

            # return dndt.swapaxes(0,2).swapaxes(1,2), d2nd2t   #returns shapes (20,3,3), (20,20,3,3,3)
            return dndt, d2nd2t   #returns shapes (20,3,3), (20,20,3,3,3)
        
        else:
            raise ValueError("In dndt: Only first and second order derivatives are implemented")

    def MinDist(self, x, seeding = 10,x0x1y0y1 = [0.0,1.0,0.0,1.0],recursive = False,recursionLevel=0,prev_t=None):
        umin = 0.0
        vmin = 0.0
        dmin = norm(x-self.CtrlPts[0][0])   # Starting point

        if recursive:
            x0,x1,y0,y1 = x0x1y0y1
            if recursive>1:
                seeding = recursive
            else:
                seeding = 4
            dx, dy = x1-x0, y1-y0   # they might change once the borders (0.0/1.0) have been considered
        else:
            x0,x1,y0,y1 = x0x1y0y1

        for u in np.linspace(x0,x1,seeding+1):
            for v in np.linspace(y0,y1,seeding+1):
                d = norm(x - self.Grg((u,v)))
                if d < dmin:
                    dmin, umin, vmin = d, u, v
        
        # if recursive and recursionLevel<8:
        if recursive and recursionLevel<8:
            if prev_t is None or (abs(prev_t[0]-umin)>5e-3 and abs(prev_t[1]-vmin)>5e-3):
                    # x0 = max(0.0,umin-dx*3/16)      # 3/16 is a bit less than 1/4
                    # x1 = min(1.0,umin+dx*3/16)      # ... and this is useful 
                    # y0 = max(0.0,vmin-dy*3/16)      # ... to increase variance 
                    # y1 = min(1.0,vmin+dy*3/16)      # ... and reduce redundance.
                    x0 = max(0.0,umin-3*dx/(8*seeding))      # 3/4*(dx/(2*seeding))...
                    x1 = min(1.0,umin+3*dx/(8*seeding))      # ... and this is useful 
                    y0 = max(0.0,vmin-3*dy/(8*seeding))      # ... to increase variance 
                    y1 = min(1.0,vmin+3*dy/(8*seeding))      # ... and reduce redundance.
            
                    return self.MinDist(x,seeding = seeding, x0x1y0y1=[x0,x1,y0,y1],recursive=recursive,recursionLevel=recursionLevel+1,prev_t=[umin,vmin])


        return umin , vmin      # Temporarily returning UV from the rough approximation on the grid. TODO: implement exact calculus

    def findProjection(self,xs, seeding=10, recursive=1, decimals = None,tracing =False):

        def proj_final_check(self,xs,t):
            # final check (for points at/beyond edges)
            if not (0<=t[0]<=1 and 0<=t[1]<=1):
                t1 = min(max(0.0,t[0]),1.0)     # trimming values
                t2 = min(max(0.0,t[1]),1.0)     # trimming values
                xc0= self.Grg0([t1,t2])
                nor0=self.D3Grg([t1,t2])
                x_tang = (xs-xc0)-(xs-xc0)@nor0
                if norm(x_tang)>2*self.BS.r/100:         # some considerable order of magnitude with respect to the patch 'size'
                    return np.array([-1.0,-1.0])
            return t


        if tracing: set_trace()
        t = np.array(self.MinDist(xs, seeding=seeding,recursive=recursive))

        tol = 1e-16
        res = 1+tol
        niter = 0
        tcandidate = t
        dist = norm(xs - self.Grg0(tcandidate))  # Initial guess for distance in case there is no convergence

        opa = 5e-2  # this allows for a certain percentage of out-patch-allowance for NR to iterate in.
        while res>tol and (0-opa<=t[0]<=1+opa and 0-opa<=t[1]<=1+opa):

            xc, dxcdt, d2xcd2t = self.Grg(t, deriv = 2)

            f = -2*(xs-xc)@dxcdt
            K =  2*(np.tensordot(-( xs-xc),d2xcd2t,axes=[[0],[0]]) + dxcdt.T @ dxcdt)

            dt=np.linalg.solve(-K,f)
            try:
                t+=dt
            except:
                set_trace()


            res = np.linalg.norm(f)

            niter +=1
            if niter > 10:
                dist_new = norm(xs - xc )
                if dist_new < dist:
                    dist = dist_new
                    tcandidate = t
                if niter> 13:
                    # return tcandidate
                    return proj_final_check(self,xs,tcandidate)
                
        t = proj_final_check(self,xs,tcandidate)
                
        return t if decimals is None else t.round(decimals)

    # xc derivs
    def dxcdxi(self,t):     # 20 scalars to be converted into diagonal matrices each
        t1,t2 = t
        N = np.zeros((20))

        c_edges = [[0,0],[3,0],[3,3],[0,3],[1,0],[2,0],[3,1],[3,2],[2,3],[1,3],[0,2],[0,1]]
        for ic, c in enumerate(c_edges):
            N[ic] = Bernstein(3, c[0], t1)*Bernstein(3, c[1], t2)

        den = max(self.eps,t1+t2)
        N[12] = Bernstein(3, 1, t1)*Bernstein(3, 1, t2)*(t1/den)
        N[13] = Bernstein(3, 1, t1)*Bernstein(3, 1, t2)*(t2/den)
        den = max(self.eps,1-t1+t2)
        N[14] = Bernstein(3, 2, t1)*Bernstein(3, 1, t2)*((1-t1)/den)
        N[15] = Bernstein(3, 2, t1)*Bernstein(3, 1, t2)*(t2/den)
        den = max(self.eps,2-t1-t2)
        N[16] = Bernstein(3, 2, t1)*Bernstein(3, 2, t2)*((1-t1)/den)
        N[17] = Bernstein(3, 2, t1)*Bernstein(3, 2, t2)*((1-t2)/den)
        den = max(self.eps,t1+1-t2)
        N[18] = Bernstein(3, 1, t1)*Bernstein(3, 2, t2)*(t1/den)
        N[19] = Bernstein(3, 1, t1)*Bernstein(3, 2, t2)*((1-t2)/den)

        return N

    def d2xcdxidt(self,t):
        t1,t2 = t
        N = np.zeros((20,2))
        c_edges = [[0,0],[3,0],[3,3],[0,3],[1,0],[2,0],[3,1],[3,2],[2,3],[1,3],[0,2],[0,1]]
        for ic, c in enumerate(c_edges):
            N[ic] = np.array([dnBernstein(3, c[0], t1, 1)*Bernstein(3, c[1], t2), Bernstein(3, c[0], t1)*dnBernstein(3, c[1], t2, 1)])

        IJ = [(1,1), (1,1), (2,1), (2,1), (2,2), (2,2), (1,2), (1,2)]
        DEN = [t1+t2, t1+t2, 1-t1+t2, 1-t1+t2, 2-t1-t2, 2-t1-t2, 1+t1-t2, 1+t1-t2]
        Xij = [t1, t2, 1-t1, t2, 1-t1, 1-t2, t1, 1-t2]
        Dixij = [t2, -t2, -t2, t2, -1+t2, 1-t2, 1-t2, -1+t2]
        Djxij = [-t1, t1, -1+t1, 1-t1, 1-t1, -1+t1, t1, -t1]
        
        for idn, (i,j), den, xij, dixij, djxij in zip(range(12,20), IJ, DEN, Xij, Dixij, Djxij):
            den = max(self.eps,den)
            xij = xij/den
            dixij = dixij/den**2
            djxij = djxij/den**2

            N[idn] = np.array([ (dnBernstein(3, i, t1, 1)*xij + Bernstein(3, i, t1)*dixij)*Bernstein(3, j, t2),
                                (dnBernstein(3, j, t2, 1)*xij + Bernstein(3, j, t2)*djxij)*Bernstein(3, i, t1)]) # I deleted den here

        return N

    def d3xcdxid2t(self,t):
        t1,t2 = t
        N = np.zeros((20,2,2))
        c_edges = [[0,0],[3,0],[3,3],[0,3],[1,0],[2,0],[3,1],[3,2],[2,3],[1,3],[0,2],[0,1]]
        for ic, c in enumerate(c_edges):
            N[ic] = np.array([[dnBernstein(3, c[0], t1, 2)*Bernstein(3, c[1], t2),dnBernstein(3, c[0], t1, 1)*dnBernstein(3, c[1], t2,1)], 
                              [dnBernstein(3, c[0], t1, 1)*dnBernstein(3, c[1], t2,1),Bernstein(3, c[0], t1)*dnBernstein(3, c[1], t2, 2)]])

        IJ = [(1,1), (1,1), (2,1), (2,1), (2,2), (2,2), (1,2), (1,2)]
        DEN = [t1+t2, t1+t2, 1-t1+t2, 1-t1+t2, 2-t1-t2, 2-t1-t2, 1+t1-t2, 1+t1-t2]
        Xij = [t1, t2, 1-t1, t2, 1-t1, 1-t2, t1, 1-t2]
        Dixij = [t2, -t2, -t2, t2, -1+t2, 1-t2, 1-t2, -1+t2]
        Djxij = [-t1, t1, -1+t1, 1-t1, 1-t1, -1+t1, t1, -t1]
        D2ixij = [-2*t2, 2*t2, -2*t2, 2*t2, -2+2*t2, 2-2*t2, -2+2*t2, 2-2*t2]
        Dijxij = [t1-t2, t2-t1, -1+t1+t2, 1-t1-t2, t2-t1, t1-t2, 1-t1-t2, -1+t1+t2]
        D2jxij = [2*t1, -2*t1, 2-2*t1, -2+2*t1, 2-2*t1, -2+2*t1, 2*t1, -2*t1]
        
        for idn, (i,j), den, xij, dixij, djxij, d2ixij, d2jxij, dijxij in zip(range(12,20), IJ, DEN, Xij, Dixij, Djxij, D2ixij, D2jxij, Dijxij):
            den = max(self.eps,den)
            xij = xij/den
            dixij = dixij/den**2
            djxij = djxij/den**2
            d2ixij = d2ixij/den**3
            dijxij = dijxij/den**3
            d2jxij = d2jxij/den**3
            Bi, dBi, d2Bi = Bernstein(3, i, t1), dnBernstein(3, i, t1, 1), dnBernstein(3, i, t1, 2)
            Bj, dBj, d2Bj = Bernstein(3, j, t2), dnBernstein(3, j, t2, 1), dnBernstein(3, j, t2, 2)


            N[idn] = np.array([ [ Bj*(d2Bi*xij + 2*dBi*dixij + Bi*d2ixij), 
                                  dBj*(Bi*dixij + dBi*xij) + Bj*(Bi*dijxij + dBi*djxij) ],
                                [ dBj*(Bi*dixij + dBi*xij) + Bj*(Bi*dijxij + dBi*djxij), 
                                  Bi*(d2Bj*xij + 2*dBj*djxij + Bj*d2jxij) ]])

        return N

    # CtrlPts derivs
    def dy1du(self,idx, inverse = False, tracing=False):
        """idx: index of the edge (0,..,3). 'inverse' option does the rest"""
        Ne = len(self.squad)
        dy1du = np.zeros((3,3*Ne))     # Here the DoF_slave is not considered

        if inverse:
            y3, y0 = self.Y0Y3(idx)
            idx3  = idx
            idx0  = [1,2,3,0][idx]
        else:
            y0, y3 = self.Y0Y3(idx)
            idx0 = idx
            idx3 = [1,2,3,0][idx]

        y0x,y0y,y0z = y0
        y3x,y3y,y3z = y3
        m0 = self.normals[idx0]
        m0x,m0y,m0z = m0
        dm0du = self.W[idx0]

        dy1dy0 = 2/3*np.eye(3)+np.outer(m0,m0)/(3*np.dot(m0,m0))
        dy1dy3 = np.eye(3)/3-np.outer(m0,m0)/(3*np.dot(m0,m0))
        dy1dm0 = np.array([[(2*m0x*(m0y**2+m0z**2)*(y0x-y3x)+(m0y**2+m0z**2)*(m0y*(y0y-y3y)+m0z*(y0z-y3z))+m0x**2*(m0y*(-y0y+y3y)+m0z*(-y0z+y3z))),(m0x*(2*m0x*m0y*(-y0x+y3x)+m0x**2*(y0y-y3y)+m0z**2*(y0y-y3y)+m0y**2*(-y0y+y3y)+2*m0y*m0z*(-y0z+y3z))),(m0x*(2*m0x*m0z*(-y0x+y3x)+2*m0y*m0z*(-y0y+y3y)+m0x**2*(y0z-y3z)+m0y**2*(y0z-y3z)+m0z**2*(-y0z+y3z)))],
                           [(m0y*((m0y**2+m0z**2)*(y0x-y3x)+m0x**2*(-y0x+y3x)-2*m0x*(m0y*(y0y-y3y)+m0z*(y0z-y3z)))),(m0x**3*(y0x-y3x)-m0x*(m0y**2-m0z**2)*(y0x-y3x)+m0x**2*(2*m0y*(y0y-y3y)+m0z*(y0z-y3z))+m0z*(2*m0y*m0z*(y0y-y3y)+m0z**2*(y0z-y3z)+m0y**2*(-y0z+y3z))),(m0y*(2*m0x*m0z*(-y0x+y3x)+2*m0y*m0z*(-y0y+y3y)+m0x**2*(y0z-y3z)+m0y**2*(y0z-y3z)+m0z**2*(-y0z+y3z)))],
                           [(m0z*((m0y**2+m0z**2)*(y0x-y3x)+m0x**2*(-y0x+y3x)-2*m0x*(m0y*(y0y-y3y)+m0z*(y0z-y3z)))),(m0z*(2*m0x*m0y*(-y0x+y3x)+m0x**2*(y0y-y3y)+m0z**2*(y0y-y3y)+m0y**2*(-y0y+y3y)+2*m0y*m0z*(-y0z+y3z))),(m0x**3*(y0x-y3x)+m0x*(m0y**2-m0z**2)*(y0x-y3x)+m0x**2*(m0y*(y0y-y3y)+2*m0z*(y0z-y3z))+m0y*(m0y**2*(y0y-y3y)+m0z**2*(-y0y+y3y)+2*m0y*m0z*(y0z-y3z)))]])/(3*(m0x**2+m0y**2+m0z**2)**2)

        dy1du[ : , 3*idx0:3*(idx0+1) ] += dy1dy0
        dy1du[ : , 3*idx3:3*(idx3+1) ] += dy1dy3
        dy1du +=  dy1dm0 @ dm0du

        return dy1du

    def dyp1du(self,idx, inverse = False, dEdyp = 0):
        """idx: index of the edge (0,..,3). 'inverse' option does the rest"""
        Ne = len(self.squad)
        dyp1du = np.zeros((3,3*Ne))     # Here the DoF_slave is not considered

        if inverse:
            y3,   y0   = self.Y0Y3(idx)
            y2,   y1   = self.Y1Y2(idx)
            yp3,  yp0  = self.Yp0Yp3(idx)
            idx3  = idx
            idx0  = [1,2,3,0][idx]

            idx00 = idx0
            idx03 = [3,0,1,2][idx]

        else:
            y0,   y3   = self.Y0Y3(idx)
            y1,   y2   = self.Y1Y2(idx)
            yp0,  yp3  = self.Yp0Yp3(idx)
            idx0 = idx
            idx3 = [1,2,3,0][idx]

            idx00 = [3,0,1,2][idx]
            idx03 = idx3

        m0, m3 = self.normals[idx0], self.normals[idx3]
        dm0du, dm3du = self.W[idx0], self.W[idx3]

        # vars:  a0, a3, b0,b3, c0, c1, c2
        a0 = np.cross(m0,(y3-y0))/norm( np.cross(m0,(y3-y0)) )
        a3 = np.cross(m3,(y3-y0))/norm( np.cross(m3,(y3-y0)) )
        b0, b3 = yp0-y0 , yp3-y3
        c0, c1, c2 = y1-y0, y2-y1, y3-y2

        # vars split
        m0x, m0y, m0z = m0
        m3x, m3y, m3z = m3
        y0x, y0y, y0z = y0
        y3x, y3y, y3z = y3
        a0x, a0y, a0z = a0
        a3x, a3y, a3z = a3
        b0x, b0y, b0z = b0
        b3x, b3y, b3z = b3
        c0x, c0y, c0z = c0
        c1x, c1y, c1z = c1
        c2x, c2y, c2z = c2

        # helpful things
        c0c0 = c0x**2+c0y**2+c0z**2
        c2c2 = c2x**2+c2y**2+c2z**2
        dy1du = self.dy1du(idx,inverse=inverse)
        dy2du = self.dy1du(idx,inverse= not inverse)
        dy00du = self.dy1du(idx00,inverse= not inverse)
        dy03du = self.dy1du(idx03,inverse= inverse)

        ###############
        # Derivatives   dyp1/d(var)
        ###############
        dyp1dy1 = np.eye(3)
        dyp1da0     = np.array([[1/3*(2*a0x*b0x+a0y*b0y+a0z*b0z+a3x*(b0x+b3x)+a3y*b3y+a3z*b3z),1/3*(a0x+a3x)*b0y,1/3*(a0x+a3x)*b0z],
                            [1/3*(a0y+a3y)*b0x,1/3*(a0x*b0x+2*a0y*b0y+a3y*b0y+a0z*b0z+a3x*b3x+a3y*b3y+a3z*b3z),1/3*(a0y+a3y)*b0z],
                            [1/3*(a0z+a3z)*b0x,1/3*(a0z+a3z)*b0y,1/3*(a0x*b0x+a0y*b0y+2*a0z*b0z+a3z*b0z+a3x*b3x+a3y*b3y+a3z*b3z)]])
        dyp1da3     = np.array([[a0y*b0y+a0z*b0z+a0x*(b0x+b3x),a0x*b3y,a0x*b3z],
                            [a0y*b3x,a0x*b0x+a0z*b0z+a0y*(b0y+b3y),a0y*b3z],
                            [a0z*b3x,a0z*b3y,a0x*b0x+a0y*b0y+a0z*(b0z+b3z)]])/3
        dyp1db0     = np.array([[a0x**2+a0x*a3x+(2*c0x*c1x)/c0c0,a0x*a0y+a0y*a3x+(2*c0y*c1x)/c0c0,a0x*a0z+a0z*a3x+(2*c0z*c1x)/c0c0],
                                [a0x*(a0y+a3y)+(2*c0x*c1y)/c0c0,a0y**2+a0y*a3y+(2*c0y*c1y)/c0c0,a0y*a0z+a0z*a3y+(2*c0z*c1y)/c0c0],
                                [a0x*(a0z+a3z)+(2*c0x*c1z)/c0c0,a0y*(a0z+a3z)+(2*c0y*c1z)/c0c0,a0z**2+a0z*a3z+(2*c0z*c1z)/c0c0]])/3
        dyp1db3     = np.array([[a0x*a3x+(c0x*c2x)/(c2c2),a0x*a3y+(c0x*c2y)/(c2c2),a0x*a3z+(c0x*c2z)/(c2c2)],
                                [a0y*a3x+(c0y*c2x)/(c2c2),a0y*a3y+(c0y*c2y)/(c2c2),a0y*a3z+(c0y*c2z)/(c2c2)],
                                [a0z*a3x+(c0z*c2x)/(c2c2),a0z*a3y+(c0z*c2y)/(c2c2),a0z*a3z+(c0z*c2z)/(c2c2)]])/3
        dyp1dc0     = np.array([[-((4*c0x*c1x*(b0x*c0x+b0y*c0y+b0z*c0z))/c0c0**2)+(2*b0x*c1x)/c0c0+(b3x*c2x+b3y*c2y+b3z*c2z)/(c2c2),(2*c1x*(b0y*(c0x**2-c0y**2+c0z**2)-2*c0y*(b0x*c0x+b0z*c0z)))/c0c0**2,(2*c1x*(b0z*(c0x**2+c0y**2-c0z**2)-2*c0z*(b0x*c0x+b0y*c0y)))/c0c0**2],
                                [(2*c1y*(b0x*(-c0x**2+c0y**2+c0z**2)-2*c0x*(b0y*c0y+b0z*c0z)))/c0c0**2,-((4*c0y*c1y*(b0x*c0x+b0y*c0y+b0z*c0z))/c0c0**2)+(2*b0y*c1y)/c0c0+(b3x*c2x+b3y*c2y+b3z*c2z)/(c2c2),(2*c1y*(b0z*(c0x**2+c0y**2-c0z**2)-2*c0z*(b0x*c0x+b0y*c0y)))/c0c0**2],
                                [(2*c1z*(b0x*(-c0x**2+c0y**2+c0z**2)-2*c0x*(b0y*c0y+b0z*c0z)))/c0c0**2,(2*c1z*(b0y*(c0x**2-c0y**2+c0z**2)-2*c0y*(b0x*c0x+b0z*c0z)))/c0c0**2,-((4*c0z*c1z*(b0x*c0x+b0y*c0y+b0z*c0z))/c0c0**2)+(2*b0z*c1z)/c0c0+(b3x*c2x+b3y*c2y+b3z*c2z)/(c2c2)]])/3
        dyp1dc1     = ((2*(b0x*c0x+b0y*c0y+b0z*c0z))/(c0c0))*np.eye(3)/3
        dyp1dc2 = np.array([[-c0x*(b3x*(c2x**2-c2y**2-c2z**2)+2*c2x*(b3y*c2y+b3z*c2z)),c0x*(b3y*(c2x**2-c2y**2+c2z**2)-2*c2y*(b3x*c2x+b3z*c2z)),c0x*(b3z*(c2x**2+c2y**2-c2z**2)-2*c2z*(b3x*c2x+b3y*c2y))],
                            [-c0y*(b3x*(c2x**2-c2y**2-c2z**2)+2*c2x*(b3y*c2y+b3z*c2z)),c0y*(b3y*(c2x**2-c2y**2+c2z**2)-2*c2y*(b3x*c2x+b3z*c2z)),c0y*(b3z*(c2x**2+c2y**2-c2z**2)-2*c2z*(b3x*c2x+b3y*c2y))],
                            [-c0z*(b3x*(c2x**2-c2y**2-c2z**2)+2*c2x*(b3y*c2y+b3z*c2z)),c0z*(b3y*(c2x**2-c2y**2+c2z**2)-2*c2y*(b3x*c2x+b3z*c2z)),c0z*(b3z*(c2x**2+c2y**2-c2z**2)-2*c2z*(b3x*c2x+b3y*c2y))]])/(3*c2c2**2)


        ###############
        # Derivatives d(var)/du
        ###############
        # da0du
        da0du = np.zeros((3,3*Ne))
        da0dm0 = np.array([[((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(-m0x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m0y*(y0x-y3x)*(y0y-y3y)+m0z*(y0x-y3x)*(y0z-y3z)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(-m0x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m0y*(y0x-y3x)*(y0y-y3y)+m0z*(y0x-y3x)*(y0z-y3z)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(-m0x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m0y*(y0x-y3x)*(y0y-y3y)+m0z*(y0x-y3x)*(y0z-y3z)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)],
                           [((m0y*(y0z-y3z)+m0z*(y3y-y0y))*(m0y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)-(y0y-y3y)*(m0x*(y0x-y3x)+m0z*(y0z-y3z))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*((y0y-y3y)*(m0x*(y0x-y3x)+m0z*(y0z-y3z))-m0y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),-(((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)-(y0y-y3y)*(m0x*(y0x-y3x)+m0z*(y0z-y3z))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2))],
                           [-(((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)-(y0z-y3z)*(m0x*(y0x-y3x)+m0y*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)-(y0z-y3z)*(m0x*(y0x-y3x)+m0y*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*((y0z-y3z)*(m0x*(y0x-y3x)+m0y*(y0y-y3y))-m0z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)]])
        da0dy0 = np.array([[-(((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2))],
                           [((m0y*(y0z-y3z)+m0z*(y3y-y0y))*(m0x**2*(y0y-y3y)+m0x*m0y*(y3x-y0x)+m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x**2*(y0y-y3y)+m0x*m0y*(y3x-y0x)+m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x**2*(y3y-y0y)+m0x*m0y*(y0x-y3x)+m0z*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)],
                           [((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x**2*(y3z-y0z)+m0x*m0z*(y0x-y3x)+m0y*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x**2*(y3z-y0z)+m0x*m0z*(y0x-y3x)+m0y*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),-(((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x**2*(y0z-y3z)+m0x*m0z*(y3x-y0x)+m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2))]])
        da0dy3 = np.array([[((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)],
                           [((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x**2*(y0y-y3y)+m0x*m0y*(y3x-y0x)+m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x**2*(y0y-y3y)+m0x*m0y*(y3x-y0x)+m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),-(((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x**2*(y3y-y0y)+m0x*m0y*(y0x-y3x)+m0z*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2))],
                           [((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x**2*(y0z-y3z)+m0x*m0z*(y3x-y0x)+m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x**2*(y3z-y0z)+m0x*m0z*(y0x-y3x)+m0y*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x**2*(y0z-y3z)+m0x*m0z*(y3x-y0x)+m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)]])
        da0du[ : , 3*idx0:3*(idx0+1)] += da0dy0
        da0du[ : , 3*idx3:3*(idx3+1)] += da0dy3
        da0du                         += da0dm0@dm0du
        # da3du
        da3du = np.zeros((3,3*Ne))
        da3dm3 = np.array([[((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(-m3x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m3y*(y0x-y3x)*(y0y-y3y)+m3z*(y0x-y3x)*(y0z-y3z)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(-m3x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m3y*(y0x-y3x)*(y0y-y3y)+m3z*(y0x-y3x)*(y0z-y3z)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(-m3x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m3y*(y0x-y3x)*(y0y-y3y)+m3z*(y0x-y3x)*(y0z-y3z)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)],
                           [((m3y*(y0z-y3z)+m3z*(y3y-y0y))*(m3y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)-(y0y-y3y)*(m3x*(y0x-y3x)+m3z*(y0z-y3z))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*((y0y-y3y)*(m3x*(y0x-y3x)+m3z*(y0z-y3z))-m3y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),-(((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)-(y0y-y3y)*(m3x*(y0x-y3x)+m3z*(y0z-y3z))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2))],
                           [-(((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)-(y0z-y3z)*(m3x*(y0x-y3x)+m3y*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)-(y0z-y3z)*(m3x*(y0x-y3x)+m3y*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*((y0z-y3z)*(m3x*(y0x-y3x)+m3y*(y0y-y3y))-m3z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)]])
        da3dy0 = np.array([[-(((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2))],
                           [((m3y*(y0z-y3z)+m3z*(y3y-y0y))*(m3x**2*(y0y-y3y)+m3x*m3y*(y3x-y0x)+m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x**2*(y0y-y3y)+m3x*m3y*(y3x-y0x)+m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x**2*(y3y-y0y)+m3x*m3y*(y0x-y3x)+m3z*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)],
                           [((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x**2*(y3z-y0z)+m3x*m3z*(y0x-y3x)+m3y*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x**2*(y3z-y0z)+m3x*m3z*(y0x-y3x)+m3y*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),-(((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x**2*(y0z-y3z)+m3x*m3z*(y3x-y0x)+m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2))]])
        da3dy3 = np.array([[((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)],
                           [((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x**2*(y0y-y3y)+m3x*m3y*(y3x-y0x)+m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x**2*(y0y-y3y)+m3x*m3y*(y3x-y0x)+m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),-(((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x**2*(y3y-y0y)+m3x*m3y*(y0x-y3x)+m3z*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2))],
                           [((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x**2*(y0z-y3z)+m3x*m3z*(y3x-y0x)+m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x**2*(y3z-y0z)+m3x*m3z*(y0x-y3x)+m3y*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x**2*(y0z-y3z)+m3x*m3z*(y3x-y0x)+m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)]])
        da3du[ : , 3*idx0:3*(idx0+1)] += da3dy0
        da3du[ : , 3*idx3:3*(idx3+1)] += da3dy3
        da3du                         += da3dm3@dm3du
        # db0du
        db0du                       = np.array(dy00du)
        db0du[:,3*idx0:3*(idx0+1)] -= np.eye(3)
        # db3du
        db3du                       = np.array(dy03du)
        db3du[:,3*idx3:3*(idx3+1)] -= np.eye(3)
        # dc0du
        dc0du                       = np.array(dy1du)
        dc0du[:,3*idx0:3*(idx0+1)] -= np.eye(3)
        # dc1du
        dc1du  = np.array(dy2du)
        dc1du -= dy1du
        # dc2du
        dc2du = -np.array(dy2du)
        dc2du[:,3*idx3:3*(idx3+1)] += np.eye(3)


        ###################
        # Contributions dyp1du  <----  y1,a0,a3,b0,b3,c0,c1,c2
        ###################
        # y1
        dyp1du += dyp1dy1@dy1du
        # a0
        dyp1du += dyp1da0@da0du
        # a3
        dyp1du += dyp1da3@da3du
        # b0
        dyp1du += dyp1db0@db0du
        # b3
        dyp1du += dyp1db3@db3du
        # c0
        dyp1du += dyp1dc0@dc0du
        # c1
        dyp1du += dyp1dc1@dc1du
        # c2
        dyp1du += dyp1dc2@dc2du


        return dyp1du

    def d2y1d2u(self,idx,inverse = False , tracing = False):
        """idx: index of the edge (0,..,3). 'inverse' option does the rest"""
        Ne = len(self.squad)
        d2y1d2u = np.zeros((3,3*Ne,3*Ne))     # Here the DoF_slave is not considered

        if inverse:
            y3, y0 = self.Y0Y3(idx)
            idx3  = idx
            idx0  = [1,2,3,0][idx]
        else:
            y0, y3 = self.Y0Y3(idx)
            idx0 = idx
            idx3 = [1,2,3,0][idx]

        y0x,y0y,y0z = y0
        y3x,y3y,y3z = y3
        m0 = self.normals[idx0]
        m0x,m0y,m0z = m0
        dm0du = self.W[idx0]
        d2m0d2u = self.dW[idx0]

        dy1dm0 = np.array([[(2*m0x*(m0y**2+m0z**2)*(y0x-y3x)+(m0y**2+m0z**2)*(m0y*(y0y-y3y)+m0z*(y0z-y3z))+m0x**2*(m0y*(-y0y+y3y)+m0z*(-y0z+y3z))),(m0x*(2*m0x*m0y*(-y0x+y3x)+m0x**2*(y0y-y3y)+m0z**2*(y0y-y3y)+m0y**2*(-y0y+y3y)+2*m0y*m0z*(-y0z+y3z))),(m0x*(2*m0x*m0z*(-y0x+y3x)+2*m0y*m0z*(-y0y+y3y)+m0x**2*(y0z-y3z)+m0y**2*(y0z-y3z)+m0z**2*(-y0z+y3z)))],
                           [(m0y*((m0y**2+m0z**2)*(y0x-y3x)+m0x**2*(-y0x+y3x)-2*m0x*(m0y*(y0y-y3y)+m0z*(y0z-y3z)))),(m0x**3*(y0x-y3x)-m0x*(m0y**2-m0z**2)*(y0x-y3x)+m0x**2*(2*m0y*(y0y-y3y)+m0z*(y0z-y3z))+m0z*(2*m0y*m0z*(y0y-y3y)+m0z**2*(y0z-y3z)+m0y**2*(-y0z+y3z))),(m0y*(2*m0x*m0z*(-y0x+y3x)+2*m0y*m0z*(-y0y+y3y)+m0x**2*(y0z-y3z)+m0y**2*(y0z-y3z)+m0z**2*(-y0z+y3z)))],
                           [(m0z*((m0y**2+m0z**2)*(y0x-y3x)+m0x**2*(-y0x+y3x)-2*m0x*(m0y*(y0y-y3y)+m0z*(y0z-y3z)))),(m0z*(2*m0x*m0y*(-y0x+y3x)+m0x**2*(y0y-y3y)+m0z**2*(y0y-y3y)+m0y**2*(-y0y+y3y)+2*m0y*m0z*(-y0z+y3z))),(m0x**3*(y0x-y3x)+m0x*(m0y**2-m0z**2)*(y0x-y3x)+m0x**2*(m0y*(y0y-y3y)+2*m0z*(y0z-y3z))+m0y*(m0y**2*(y0y-y3y)+m0z**2*(-y0y+y3y)+2*m0y*m0z*(y0z-y3z)))]])/(3*(m0x**2+m0y**2+m0z**2)**2)

        d2y1dm0dy0 = np.array([[[2*m0x*(m0y**2+m0z**2),m0y*(-m0x**2+m0y**2+m0z**2),m0z*(-m0x**2+m0y**2+m0z**2)],[-2*m0x**2*m0y,m0x*(m0x**2-m0y**2+m0z**2),-2*m0x*m0y*m0z],[-2*m0x**2*m0z,-2*m0x*m0y*m0z,m0x*(m0x**2+m0y**2-m0z**2)]],
                               [[m0y*(-m0x**2+m0y**2+m0z**2),-2*m0x*m0y**2,-2*m0x*m0y*m0z],[m0x*(m0x**2-m0y**2+m0z**2),2*m0y*(m0x**2+m0z**2),m0z*(m0x**2-m0y**2+m0z**2)],[-2*m0x*m0y*m0z,-2*m0y**2*m0z,m0y*(m0x**2+m0y**2-m0z**2)]],
                               [[m0z*(-m0x**2+m0y**2+m0z**2),-2*m0x*m0y*m0z,-2*m0x*m0z**2],[-2*m0x*m0y*m0z,m0z*(m0x**2-m0y**2+m0z**2),-2*m0y*m0z**2],[m0x*(m0x**2+m0y**2-m0z**2),m0y*(m0x**2+m0y**2-m0z**2),2*m0z*(m0x**2+m0y**2)]]])/(3*(m0x**2+m0y**2+m0z**2)**2)

        d2y1d2m0 = np.array([[ [2*(-3*m0x**2*(m0y**2+m0z**2)*(y0x-y3x)+m0x**3*(m0y*(y0y-y3y)+m0z*(y0z-y3z))-3*m0x*(m0y**2+m0z**2)*(m0y*(y0y-y3y)+m0z*(y0z-y3z))+(m0y**2+m0z**2)**2*(y0x-y3x)),6*m0x**2*m0y*(m0y*(y0y-y3y)+m0z*(y0z-y3z))+4*m0x**3*m0y*(y0x-y3x)+m0x**4*(y3y-y0y)-4*m0x*m0y*(m0y**2+m0z**2)*(y0x-y3x)-(m0y**2+m0z**2)*(m0y**2*(y0y-y3y)+2*m0y*m0z*(y0z-y3z)+m0z**2*(y3y-y0y)),6*m0x**2*m0z*(m0y*(y0y-y3y)+m0z*(y0z-y3z))+4*m0x**3*m0z*(y0x-y3x)+m0x**4*(y3z-y0z)-4*m0x*m0z*(m0y**2+m0z**2)*(y0x-y3x)+(m0y**2+m0z**2)*(m0y**2*(y0z-y3z)+2*m0y*m0z*(y3y-y0y)+m0z**2*(y3z-y0z))],
                               [6*m0x**2*m0y*(m0y*(y0y-y3y)+m0z*(y0z-y3z))+4*m0x**3*m0y*(y0x-y3x)+m0x**4*(y3y-y0y)-4*m0x*m0y*(m0y**2+m0z**2)*(y0x-y3x)-(m0y**2+m0z**2)*(m0y**2*(y0y-y3y)+2*m0y*m0z*(y0z-y3z)+m0z**2*(y3y-y0y)),-2*m0x*(m0x**2*(3*m0y*(y0y-y3y)+m0z*(y0z-y3z))+m0x**3*(y0x-y3x)-m0x*(3*m0y**2-m0z**2)*(y0x-y3x)+3*m0y**2*m0z*(y3z-y0z)+m0y**3*(y3y-y0y)+3*m0y*m0z**2*(y0y-y3y)+m0z**3*(y0z-y3z)),-2*m0x*(m0x**2*(m0y*(y0z-y3z)+m0z*(y0y-y3y))+4*m0x*m0y*m0z*(y3x-y0x)+3*m0y**2*m0z*(y3y-y0y)+m0y**3*(y0z-y3z)+3*m0y*m0z**2*(y3z-y0z)+m0z**3*(y0y-y3y))],
                               [6*m0x**2*m0z*(m0y*(y0y-y3y)+m0z*(y0z-y3z))+4*m0x**3*m0z*(y0x-y3x)+m0x**4*(y3z-y0z)-4*m0x*m0z*(m0y**2+m0z**2)*(y0x-y3x)+(m0y**2+m0z**2)*(m0y**2*(y0z-y3z)+2*m0y*m0z*(y3y-y0y)+m0z**2*(y3z-y0z)),-2*m0x*(m0x**2*(m0y*(y0z-y3z)+m0z*(y0y-y3y))+4*m0x*m0y*m0z*(y3x-y0x)+3*m0y**2*m0z*(y3y-y0y)+m0y**3*(y0z-y3z)+3*m0y*m0z**2*(y3z-y0z)+m0z**3*(y0y-y3y)),-2*m0x*(m0x**2*(m0y*(y0y-y3y)+3*m0z*(y0z-y3z))+m0x**3*(y0x-y3x)+m0x*(m0y**2-3*m0z**2)*(y0x-y3x)+3*m0y**2*m0z*(y0z-y3z)+m0y**3*(y0y-y3y)+3*m0y*m0z**2*(y3y-y0y)+m0z**3*(y3z-y0z))]],
                             [ [2*m0y*(3*m0x**2*(m0y*(y0y-y3y)+m0z*(y0z-y3z))+m0x**3*(y0x-y3x)-3*m0x*(m0y**2+m0z**2)*(y0x-y3x)-(m0y**2+m0z**2)*(m0y*(y0y-y3y)+m0z*(y0z-y3z))),6*m0x**2*m0y**2*(y0x-y3x)+m0x**3*(-4*m0y*y0y+4*m0y*y3y-2*m0z*y0z+2*m0z*y3z)+m0x**4*(y3x-y0x)+2*m0x*(3*m0y**2*m0z*(y0z-y3z)+2*m0y**3*(y0y-y3y)+2*m0y*m0z**2*(y3y-y0y)+m0z**3*(y3z-y0z))-(m0y**4-m0z**4)*(y0x-y3x),-2*m0y*(3*m0x**2*m0z*(y3x-y0x)+m0x**3*(y0z-y3z)+m0x*(m0y**2*(y0z-y3z)+4*m0y*m0z*(y3y-y0y)+3*m0z**2*(y3z-y0z))+m0z*(m0y**2+m0z**2)*(y0x-y3x))],
                               [6*m0x**2*m0y**2*(y0x-y3x)+m0x**3*(-4*m0y*y0y+4*m0y*y3y-2*m0z*y0z+2*m0z*y3z)+m0x**4*(y3x-y0x)+2*m0x*(3*m0y**2*m0z*(y0z-y3z)+2*m0y**3*(y0y-y3y)+2*m0y*m0z**2*(y3y-y0y)+m0z**3*(y3z-y0z))-(m0y**4-m0z**4)*(y0x-y3x),2*(m0x**2*(-3*m0y**2*(y0y-y3y)+3*m0y*m0z*(y3z-y0z)+2*m0z**2*(y0y-y3y))+3*m0x**3*m0y*(y3x-y0x)+m0x**4*(y0y-y3y)+m0x*m0y*(m0y**2-3*m0z**2)*(y0x-y3x)+m0z*(3*m0y**2*m0z*(y3y-y0y)+m0y**3*(y0z-y3z)+3*m0y*m0z**2*(y3z-y0z)+m0z**3*(y0y-y3y))),4*m0x**2*m0y*m0z*(y3y-y0y)+2*m0x**3*m0z*(y3x-y0x)+m0x**4*(y0z-y3z)-2*m0x*m0z*(m0z**2-3*m0y**2)*(y0x-y3x)+6*m0y**2*m0z**2*(y0z-y3z)+4*m0y**3*m0z*(y0y-y3y)+m0y**4*(y3z-y0z)+4*m0y*m0z**3*(y3y-y0y)+m0z**4*(y3z-y0z)],
                               [-2*m0y*(3*m0x**2*m0z*(y3x-y0x)+m0x**3*(y0z-y3z)+m0x*(m0y**2*(y0z-y3z)+4*m0y*m0z*(y3y-y0y)+3*m0z**2*(y3z-y0z))+m0z*(m0y**2+m0z**2)*(y0x-y3x)),4*m0x**2*m0y*m0z*(y3y-y0y)+2*m0x**3*m0z*(y3x-y0x)+m0x**4*(y0z-y3z)-2*m0x*m0z*(m0z**2-3*m0y**2)*(y0x-y3x)+6*m0y**2*m0z**2*(y0z-y3z)+4*m0y**3*m0z*(y0y-y3y)+m0y**4*(y3z-y0z)+4*m0y*m0z**3*(y3y-y0y)+m0z**4*(y3z-y0z),-2*m0y*(m0x**2*(m0y*(y0y-y3y)+3*m0z*(y0z-y3z))+m0x**3*(y0x-y3x)+m0x*(m0y**2-3*m0z**2)*(y0x-y3x)+3*m0y**2*m0z*(y0z-y3z)+m0y**3*(y0y-y3y)+3*m0y*m0z**2*(y3y-y0y)+m0z**3*(y3z-y0z))]],
                             [ [2*m0z*(3*m0x**2*(m0y*(y0y-y3y)+m0z*(y0z-y3z))+m0x**3*(y0x-y3x)-3*m0x*(m0y**2+m0z**2)*(y0x-y3x)-(m0y**2+m0z**2)*(m0y*(y0y-y3y)+m0z*(y0z-y3z))),-2*m0z*(3*m0x**2*m0y*(y3x-y0x)+m0x**3*(y0y-y3y)+m0x*(-3*m0y**2*(y0y-y3y)+4*m0y*m0z*(y3z-y0z)+m0z**2*(y0y-y3y))+m0y*(m0y**2+m0z**2)*(y0x-y3x)),m0x**3*(-2*m0y*y0y+2*m0y*y3y-4*m0z*y0z+4*m0z*y3z)+6*m0x**2*m0z**2*(y0x-y3x)+m0x**4*(y3x-y0x)+m0x*(4*m0y**2*m0z*(y3z-y0z)-2*m0y**3*(y0y-y3y)+6*m0y*m0z**2*(y0y-y3y)+4*m0z**3*(y0z-y3z))+(m0y**4-m0z**4)*(y0x-y3x)],
                               [-2*m0z*(3*m0x**2*m0y*(y3x-y0x)+m0x**3*(y0y-y3y)+m0x*(-3*m0y**2*(y0y-y3y)+4*m0y*m0z*(y3z-y0z)+m0z**2*(y0y-y3y))+m0y*(m0y**2+m0z**2)*(y0x-y3x)),-2*m0z*(m0x**2*(3*m0y*(y0y-y3y)+m0z*(y0z-y3z))+m0x**3*(y0x-y3x)-m0x*(3*m0y**2-m0z**2)*(y0x-y3x)+3*m0y**2*m0z*(y3z-y0z)+m0y**3*(y3y-y0y)+3*m0y*m0z**2*(y0y-y3y)+m0z**3*(y0z-y3z)),4*m0x**2*m0y*m0z*(y3z-y0z)+2*m0x**3*m0y*(y3x-y0x)+m0x**4*(y0y-y3y)-2*m0x*m0y*(m0y**2-3*m0z**2)*(y0x-y3x)+6*m0y**2*m0z**2*(y0y-y3y)+4*m0y**3*m0z*(y3z-y0z)+m0y**4*(y3y-y0y)+4*m0y*m0z**3*(y0z-y3z)+m0z**4*(y3y-y0y)],
                               [m0x**3*(-2*m0y*y0y+2*m0y*y3y-4*m0z*y0z+4*m0z*y3z)+6*m0x**2*m0z**2*(y0x-y3x)+m0x**4*(y3x-y0x)+m0x*(4*m0y**2*m0z*(y3z-y0z)-2*m0y**3*(y0y-y3y)+6*m0y*m0z**2*(y0y-y3y)+4*m0z**3*(y0z-y3z))+(m0y**4-m0z**4)*(y0x-y3x),4*m0x**2*m0y*m0z*(y3z-y0z)+2*m0x**3*m0y*(y3x-y0x)+m0x**4*(y0y-y3y)-2*m0x*m0y*(m0y**2-3*m0z**2)*(y0x-y3x)+6*m0y**2*m0z**2*(y0y-y3y)+4*m0y**3*m0z*(y3z-y0z)+m0y**4*(y3y-y0y)+4*m0y*m0z**3*(y0z-y3z)+m0z**4*(y3y-y0y),2*(m0x**2*(2*m0y**2*(y0z-y3z)+3*m0y*m0z*(y3y-y0y)+3*m0z**2*(y3z-y0z))+3*m0x**3*m0z*(y3x-y0x)+m0x**4*(y0z-y3z)+m0x*m0z*(m0z**2-3*m0y**2)*(y0x-y3x)+m0y*(3*m0y**2*m0z*(y3y-y0y)+m0y**3*(y0z-y3z)+3*m0y*m0z**2*(y3z-y0z)+m0z**3*(y0y-y3y)))]]])/(3*(m0x**2+m0y**2+m0z**2)**3)

        d2y1dudy0 = np.tensordot(d2y1dm0dy0,dm0du,axes=[1,0])

        d2y1d2u[ : , 3*idx0:3*(idx0+1) , : ] += d2y1dudy0
        d2y1d2u[ : , 3*idx3:3*(idx3+1) , : ] -= d2y1dudy0
        d2y1d2u[ : , : , 3*idx0:3*(idx0+1) ] += d2y1dudy0.swapaxes(1,2)
        d2y1d2u[ : , : , 3*idx3:3*(idx3+1) ] -= d2y1dudy0.swapaxes(1,2)
        d2y1d2u += np.tensordot(np.tensordot(d2y1d2m0,dm0du,axes=1),dm0du,axes=[1,0])
        d2y1d2u += np.tensordot(dy1dm0,d2m0d2u,axes=1)

        return d2y1d2u

    def d2yp1d2u(self,idx,inverse = False , tracing = False):
        """idx: index of the edge (0,..,3). 'inverse' option does the rest"""
        Ne = len(self.squad)
        d2yp1d2u = np.zeros((3,3*Ne,3*Ne))     # Here the DoF_slave is not considered

        if inverse:
            y3,   y0   = self.Y0Y3(idx)
            y2,   y1   = self.Y1Y2(idx)
            yp3,  yp0  = self.Yp0Yp3(idx)
            idx3  = idx
            idx0  = [1,2,3,0][idx]

            idx00 = idx0
            idx03 = [3,0,1,2][idx]

        else:
            y0,   y3   = self.Y0Y3(idx)
            y1,   y2   = self.Y1Y2(idx)
            yp0,  yp3  = self.Yp0Yp3(idx)
            idx0 = idx
            idx3 = [1,2,3,0][idx]

            idx00 = [3,0,1,2][idx]
            idx03 = idx3

        # m's
        m0, m3 = self.normals[idx0], self.normals[idx3]
        dm0du, dm3du = self.W[idx0], self.W[idx3]
        d2m0d2u, d2m3d2u = self.dW[idx0], self.dW[idx3]

        # a0, a3, b0, b3, c0, c1, c2 
        a0 = np.cross(m0,(y3-y0))/norm( np.cross(m0,(y3-y0)) )
        a3 = np.cross(m3,(y3-y0))/norm( np.cross(m3,(y3-y0)) )
        b0, b3 = yp0-y0 , yp3-y3
        c0, c1, c2 = y1-y0, y2-y1, y3-y2

        # vars split
        m0x, m0y, m0z = m0
        m3x, m3y, m3z = m3
        y0x, y0y, y0z = y0
        y3x, y3y, y3z = y3
        a0x, a0y, a0z = a0
        a3x, a3y, a3z = a3
        b0x, b0y, b0z = b0
        b3x, b3y, b3z = b3
        c0x, c0y, c0z = c0
        c1x, c1y, c1z = c1
        c2x, c2y, c2z = c2

        # helpful things
        c0c0 = c0x**2+c0y**2+c0z**2
        c2c2 = c2x**2+c2y**2+c2z**2
        dy1du = self.dy1du(idx,inverse=inverse)
        dy2du = self.dy1du(idx,inverse= not inverse)
        dy00du = self.dy1du(idx00,inverse= not inverse)
        dy03du = self.dy1du(idx03,inverse= inverse)
        d2y1d2u = self.d2y1d2u(idx,inverse=inverse)
        d2y2d2u = self.d2y1d2u(idx,inverse= not inverse)
        d2y00d2u = self.d2y1d2u(idx00,inverse= not inverse)
        d2y03d2u = self.d2y1d2u(idx03,inverse= inverse)


        ###############
        # Derivatives   dyp1/d(var) , d2y1/d2(var,var)
        ###############
        # y1
        dyp1dy1 = np.eye(3)
        # a0
        dyp1da0     = np.array([[1/3*(2*a0x*b0x+a0y*b0y+a0z*b0z+a3x*(b0x+b3x)+a3y*b3y+a3z*b3z),1/3*(a0x+a3x)*b0y,1/3*(a0x+a3x)*b0z],
                            [1/3*(a0y+a3y)*b0x,1/3*(a0x*b0x+2*a0y*b0y+a3y*b0y+a0z*b0z+a3x*b3x+a3y*b3y+a3z*b3z),1/3*(a0y+a3y)*b0z],
                            [1/3*(a0z+a3z)*b0x,1/3*(a0z+a3z)*b0y,1/3*(a0x*b0x+a0y*b0y+2*a0z*b0z+a3z*b0z+a3x*b3x+a3y*b3y+a3z*b3z)]])
        d2yp1d2a0   = np.array([[[(2*b0x),b0y,b0z],[b0y,0,0],[b0z,0,0]],
                              [[0,b0x,0],[b0x,(2*b0y),b0z],[0,b0z,0]],
                              [[0,0,b0x],[0,0,b0y],[b0x,b0y,(2*b0z)]]])/3
        d2yp1da0da3 = np.array([[[b0x+b3x,b3y,b3z],[b0y,0,0],[b0z,0,0]],
                                [[0,b0x,0],[b3x,b0y+b3y,b3z],[0,b0z,0]],
                                [[0,0,b0x],[0,0,b0y],[b3x,b3y,b0z+b3z]]])/3
        d2yp1da0db0 = np.array([[[2*a0x+a3x,a0y,a0z],[0,a0x+a3x,0],[0,0,a0x+a3x]],
                                [[a0y+a3y,0,0],[a0x,2*a0y+a3y,a0z],[0,0,a0y+a3y]],
                                [[a0z+a3z,0,0],[0,a0z+a3z,0],[a0x,a0y,2*a0z+a3z]]])/3
        d2yp1da0db3 = np.array([[[a3x,a3y,a3z],[0,0,0],[0,0,0]],
                                [[0,0,0],[a3x,a3y,a3z],[0,0,0]],
                                [[0,0,0],[0,0,0],[a3x,a3y,a3z]]])/3
        # a3
        dyp1da3     = np.array([[a0y*b0y+a0z*b0z+a0x*(b0x+b3x),a0x*b3y,a0x*b3z],
                            [a0y*b3x,a0x*b0x+a0z*b0z+a0y*(b0y+b3y),a0y*b3z],
                            [a0z*b3x,a0z*b3y,a0x*b0x+a0y*b0y+a0z*(b0z+b3z)]])/3
        d2yp1da3da0 = d2yp1da0da3.swapaxes(1,2)
        d2yp1da3db0 = np.array([[[a0x,a0y,a0z],[0,0,0],[0,0,0]],
                                [[0,0,0],[a0x,a0y,a0z],[0,0,0]],
                                [[0,0,0],[0,0,0],[a0x,a0y,a0z]]])/3
        d2yp1da3db3 = np.array([[[a0x,0,0],[0,a0x,0],[0,0,a0x]],
                                [[a0y,0,0],[0,a0y,0],[0,0,a0y]],
                                [[a0z,0,0],[0,a0z,0],[0,0,a0z]]])/3
        # b0                        
        dyp1db0     = np.array([[a0x**2+a0x*a3x+(2*c0x*c1x)/c0c0,a0x*a0y+a0y*a3x+(2*c0y*c1x)/c0c0,a0x*a0z+a0z*a3x+(2*c0z*c1x)/c0c0],
                                [a0x*(a0y+a3y)+(2*c0x*c1y)/c0c0,a0y**2+a0y*a3y+(2*c0y*c1y)/c0c0,a0y*a0z+a0z*a3y+(2*c0z*c1y)/c0c0],
                                [a0x*(a0z+a3z)+(2*c0x*c1z)/c0c0,a0y*(a0z+a3z)+(2*c0y*c1z)/c0c0,a0z**2+a0z*a3z+(2*c0z*c1z)/c0c0]])/3
        d2yp1db0da0 = d2yp1da0db0.swapaxes(1,2)
        d2yp1db0da3 = d2yp1da3db0.swapaxes(1,2)
        d2yp1db0dc0 = np.array([[[-((2*c1x*(c0x**2-c0y**2-c0z**2))/c0c0**2),-((4*c0x*c0y*c1x)/c0c0**2),-((4*c0x*c0z*c1x)/c0c0**2)],[-((4*c0x*c0y*c1x)/c0c0**2),(2*c1x*(c0x**2-c0y**2+c0z**2))/c0c0**2,-((4*c0y*c0z*c1x)/c0c0**2)],[-((4*c0x*c0z*c1x)/c0c0**2),-((4*c0y*c0z*c1x)/c0c0**2),(2*c1x*(c0x**2+c0y**2-c0z**2))/c0c0**2]],
                                [[-((2*c1y*(c0x**2-c0y**2-c0z**2))/c0c0**2),-((4*c0x*c0y*c1y)/c0c0**2),-((4*c0x*c0z*c1y)/c0c0**2)],[-((4*c0x*c0y*c1y)/c0c0**2),(2*c1y*(c0x**2-c0y**2+c0z**2))/c0c0**2,-((4*c0y*c0z*c1y)/c0c0**2)],[-((4*c0x*c0z*c1y)/c0c0**2),-((4*c0y*c0z*c1y)/c0c0**2),(2*c1y*(c0x**2+c0y**2-c0z**2))/c0c0**2]],
                                [[-((2*c1z*(c0x**2-c0y**2-c0z**2))/c0c0**2),-((4*c0x*c0y*c1z)/c0c0**2),-((4*c0x*c0z*c1z)/c0c0**2)],[-((4*c0x*c0y*c1z)/c0c0**2),(2*c1z*(c0x**2-c0y**2+c0z**2))/c0c0**2,-((4*c0y*c0z*c1z)/c0c0**2)],[-((4*c0x*c0z*c1z)/c0c0**2),-((4*c0y*c0z*c1z)/c0c0**2),(2*c1z*(c0x**2+c0y**2-c0z**2))/c0c0**2]]])/3
        d2yp1db0dc1 = np.array([[[(2*c0x)/c0c0,0,0],[(2*c0y)/c0c0,0,0],[(2*c0z)/c0c0,0,0]],
                                [[0,(2*c0x)/c0c0,0],[0,(2*c0y)/c0c0,0],[0,(2*c0z)/c0c0,0]],
                                [[0,0,(2*c0x)/c0c0],[0,0,(2*c0y)/c0c0],[0,0,(2*c0z)/c0c0]]])/3
        # b3
        dyp1db3     = np.array([[a0x*a3x+(c0x*c2x)/(c2c2),a0x*a3y+(c0x*c2y)/(c2c2),a0x*a3z+(c0x*c2z)/(c2c2)],
                                [a0y*a3x+(c0y*c2x)/(c2c2),a0y*a3y+(c0y*c2y)/(c2c2),a0y*a3z+(c0y*c2z)/(c2c2)],
                                [a0z*a3x+(c0z*c2x)/(c2c2),a0z*a3y+(c0z*c2y)/(c2c2),a0z*a3z+(c0z*c2z)/(c2c2)]])/3
        d2yp1db3da0 = d2yp1da0db3.swapaxes(1,2)
        d2yp1db3da3 = d2yp1da3db3.swapaxes(1,2)
        d2yp1db3dc0 = np.array([[[c2x/(c2c2),0,0],[c2y/(c2c2),0,0],[c2z/(c2c2),0,0]],
                                [[0,c2x/(c2c2),0],[0,c2y/(c2c2),0],[0,c2z/(c2c2),0]],
                                [[0,0,c2x/(c2c2)],[0,0,c2y/(c2c2)],[0,0,c2z/(c2c2)]]])/3
        d2yp1db3dc2 = np.array([[[c0x*(-c2x**2+c2y**2+c2z**2),-2*c0x*c2x*c2y,-2*c0x*c2x*c2z],[-2*c0x*c2x*c2y,c0x*(c2x**2-c2y**2+c2z**2),-2*c0x*c2y*c2z],[-2*c0x*c2x*c2z,-2*c0x*c2y*c2z,c0x*(c2x**2+c2y**2-c2z**2)]],
                                [[c0y*(-c2x**2+c2y**2+c2z**2),-2*c0y*c2x*c2y,-2*c0y*c2x*c2z],[-2*c0y*c2x*c2y,c0y*(c2x**2-c2y**2+c2z**2),-2*c0y*c2y*c2z],[-2*c0y*c2x*c2z,-2*c0y*c2y*c2z,c0y*(c2x**2+c2y**2-c2z**2)]],
                                [[c0z*(-c2x**2+c2y**2+c2z**2),-2*c0z*c2x*c2y,-2*c0z*c2x*c2z],[-2*c0z*c2x*c2y,c0z*(c2x**2-c2y**2+c2z**2),-2*c0z*c2y*c2z],[-2*c0z*c2x*c2z,-2*c0z*c2y*c2z,c0z*(c2x**2+c2y**2-c2z**2)]]])/(3*c2c2**2)
        # c0
        dyp1dc0     = np.array([[-((4*c0x*c1x*(b0x*c0x+b0y*c0y+b0z*c0z))/c0c0**2)+(2*b0x*c1x)/c0c0+(b3x*c2x+b3y*c2y+b3z*c2z)/(c2c2),(2*c1x*(b0y*(c0x**2-c0y**2+c0z**2)-2*c0y*(b0x*c0x+b0z*c0z)))/c0c0**2,(2*c1x*(b0z*(c0x**2+c0y**2-c0z**2)-2*c0z*(b0x*c0x+b0y*c0y)))/c0c0**2],
                                [(2*c1y*(b0x*(-c0x**2+c0y**2+c0z**2)-2*c0x*(b0y*c0y+b0z*c0z)))/c0c0**2,-((4*c0y*c1y*(b0x*c0x+b0y*c0y+b0z*c0z))/c0c0**2)+(2*b0y*c1y)/c0c0+(b3x*c2x+b3y*c2y+b3z*c2z)/(c2c2),(2*c1y*(b0z*(c0x**2+c0y**2-c0z**2)-2*c0z*(b0x*c0x+b0y*c0y)))/c0c0**2],
                                [(2*c1z*(b0x*(-c0x**2+c0y**2+c0z**2)-2*c0x*(b0y*c0y+b0z*c0z)))/c0c0**2,(2*c1z*(b0y*(c0x**2-c0y**2+c0z**2)-2*c0y*(b0x*c0x+b0z*c0z)))/c0c0**2,-((4*c0z*c1z*(b0x*c0x+b0y*c0y+b0z*c0z))/c0c0**2)+(2*b0z*c1z)/c0c0+(b3x*c2x+b3y*c2y+b3z*c2z)/(c2c2)]])/3
        d2yp1dc0db0 = d2yp1db0dc0.swapaxes(1,2)
        d2yp1dc0db3 = d2yp1db3dc0.swapaxes(1,2)
        d2yp1d2c0   = 4*np.array([[[c1x*(b0x*(c0x**3-3*c0x*(c0y**2+c0z**2))-(-3*c0x**2+c0y**2+c0z**2)*(b0y*c0y+b0z*c0z)),c1x*(-(b0x*c0y*(-3*c0x**2+c0y**2+c0z**2)+b0y*c0x*(c0x**2-3*c0y**2+c0z**2)-4*b0z*c0x*c0y*c0z)),c1x*(-(b0x*c0z*(-3*c0x**2+c0y**2+c0z**2)-4*b0y*c0x*c0y*c0z+b0z*c0x*(c0x**2+c0y**2-3*c0z**2)))],[c1x*(-(b0x*c0y*(-3*c0x**2+c0y**2+c0z**2)+b0y*c0x*(c0x**2-3*c0y**2+c0z**2)-4*b0z*c0x*c0y*c0z)),c1x*(-(b0x*c0x*(c0x**2-3*c0y**2+c0z**2)+b0y*c0y*(3*c0x**2-c0y**2+3*c0z**2)+b0z*c0z*(c0x**2-3*c0y**2+c0z**2))),c1x*(-(-4*b0x*c0x*c0y*c0z+b0y*c0z*(c0x**2-3*c0y**2+c0z**2)+b0z*c0y*(c0x**2+c0y**2-3*c0z**2)))],[c1x*(-(b0x*c0z*(-3*c0x**2+c0y**2+c0z**2)-4*b0y*c0x*c0y*c0z+b0z*c0x*(c0x**2+c0y**2-3*c0z**2))),c1x*(-(-4*b0x*c0x*c0y*c0z+b0y*c0z*(c0x**2-3*c0y**2+c0z**2)+b0z*c0y*(c0x**2+c0y**2-3*c0z**2))),c1x*(-(b0x*c0x*(c0x**2+c0y**2-3*c0z**2)+b0y*c0y*(c0x**2+c0y**2-3*c0z**2)+b0z*c0z*(3*c0x**2+3*c0y**2-c0z**2)))]],
                                  [[c1y*(b0x*(c0x**3-3*c0x*(c0y**2+c0z**2))-(-3*c0x**2+c0y**2+c0z**2)*(b0y*c0y+b0z*c0z)),c1y*(-(b0x*c0y*(-3*c0x**2+c0y**2+c0z**2)+b0y*c0x*(c0x**2-3*c0y**2+c0z**2)-4*b0z*c0x*c0y*c0z)),c1y*(-(b0x*c0z*(-3*c0x**2+c0y**2+c0z**2)-4*b0y*c0x*c0y*c0z+b0z*c0x*(c0x**2+c0y**2-3*c0z**2)))],[c1y*(-(b0x*c0y*(-3*c0x**2+c0y**2+c0z**2)+b0y*c0x*(c0x**2-3*c0y**2+c0z**2)-4*b0z*c0x*c0y*c0z)),c1y*(-(b0x*c0x*(c0x**2-3*c0y**2+c0z**2)+b0y*c0y*(3*c0x**2-c0y**2+3*c0z**2)+b0z*c0z*(c0x**2-3*c0y**2+c0z**2))),c1y*(-(-4*b0x*c0x*c0y*c0z+b0y*c0z*(c0x**2-3*c0y**2+c0z**2)+b0z*c0y*(c0x**2+c0y**2-3*c0z**2)))],[c1y*(-(b0x*c0z*(-3*c0x**2+c0y**2+c0z**2)-4*b0y*c0x*c0y*c0z+b0z*c0x*(c0x**2+c0y**2-3*c0z**2))),c1y*(-(-4*b0x*c0x*c0y*c0z+b0y*c0z*(c0x**2-3*c0y**2+c0z**2)+b0z*c0y*(c0x**2+c0y**2-3*c0z**2))),c1y*(-(b0x*c0x*(c0x**2+c0y**2-3*c0z**2)+b0y*c0y*(c0x**2+c0y**2-3*c0z**2)+b0z*c0z*(3*c0x**2+3*c0y**2-c0z**2)))]],
                                  [[c1z*(b0x*(c0x**3-3*c0x*(c0y**2+c0z**2))-(-3*c0x**2+c0y**2+c0z**2)*(b0y*c0y+b0z*c0z)),c1z*(-(b0x*c0y*(-3*c0x**2+c0y**2+c0z**2)+b0y*c0x*(c0x**2-3*c0y**2+c0z**2)-4*b0z*c0x*c0y*c0z)),c1z*(-(b0x*c0z*(-3*c0x**2+c0y**2+c0z**2)-4*b0y*c0x*c0y*c0z+b0z*c0x*(c0x**2+c0y**2-3*c0z**2)))],[c1z*(-(b0x*c0y*(-3*c0x**2+c0y**2+c0z**2)+b0y*c0x*(c0x**2-3*c0y**2+c0z**2)-4*b0z*c0x*c0y*c0z)),c1z*(-(b0x*c0x*(c0x**2-3*c0y**2+c0z**2)+b0y*c0y*(3*c0x**2-c0y**2+3*c0z**2)+b0z*c0z*(c0x**2-3*c0y**2+c0z**2))),c1z*(-(-4*b0x*c0x*c0y*c0z+b0y*c0z*(c0x**2-3*c0y**2+c0z**2)+b0z*c0y*(c0x**2+c0y**2-3*c0z**2)))],[c1z*(-(b0x*c0z*(-3*c0x**2+c0y**2+c0z**2)-4*b0y*c0x*c0y*c0z+b0z*c0x*(c0x**2+c0y**2-3*c0z**2))),c1z*(-(-4*b0x*c0x*c0y*c0z+b0y*c0z*(c0x**2-3*c0y**2+c0z**2)+b0z*c0y*(c0x**2+c0y**2-3*c0z**2))),c1z*(-(b0x*c0x*(c0x**2+c0y**2-3*c0z**2)+b0y*c0y*(c0x**2+c0y**2-3*c0z**2)+b0z*c0z*(3*c0x**2+3*c0y**2-c0z**2)))]]])/(3*c0c0**3)
        d2yp1dc0dc1 = np.array([[[2*b0x*(-c0x**2+c0y**2+c0z**2)-4*c0x*(b0y*c0y+b0z*c0z),0,0],[2*b0y*(c0x**2-c0y**2+c0z**2)-4*c0y*(b0x*c0x+b0z*c0z),0,0],[2*b0z*(c0x**2+c0y**2-c0z**2)-4*c0z*(b0x*c0x+b0y*c0y),0,0]],
                                [[0,2*b0x*(-c0x**2+c0y**2+c0z**2)-4*c0x*(b0y*c0y+b0z*c0z),0],[0,2*b0y*(c0x**2-c0y**2+c0z**2)-4*c0y*(b0x*c0x+b0z*c0z),0],[0,2*b0z*(c0x**2+c0y**2-c0z**2)-4*c0z*(b0x*c0x+b0y*c0y),0]],
                                [[0,0,2*b0x*(-c0x**2+c0y**2+c0z**2)-4*c0x*(b0y*c0y+b0z*c0z)],[0,0,2*b0y*(c0x**2-c0y**2+c0z**2)-4*c0y*(b0x*c0x+b0z*c0z)],[0,0,2*b0z*(c0x**2+c0y**2-c0z**2)-4*c0z*(b0x*c0x+b0y*c0y)]]])/(3*c0c0**2)
        valc0c2     = np.array([b3x*(-c2x**2+c2y**2+c2z**2)-2*c2x*(b3y*c2y+b3z*c2z),b3y*(c2x**2-c2y**2+c2z**2)-2*c2y*(b3x*c2x+b3z*c2z),b3z*(c2x**2+c2y**2-c2z**2)-2*c2z*(b3x*c2x+b3y*c2y)])
        d2yp1dc0dc2 = np.array([[valc0c2,[0,0,0],[0,0,0]],
                                [[0,0,0],valc0c2,[0,0,0]],
                                [[0,0,0],[0,0,0],valc0c2]])/(3*c2c2**2)
        # c1
        dyp1dc1     = ((2*(b0x*c0x+b0y*c0y+b0z*c0z))/(c0c0))*np.eye(3)/3
        d2yp1dc1db0 = d2yp1db0dc1.swapaxes(1,2)
        d2yp1dc1dc0 = d2yp1dc0dc1.swapaxes(1,2)
        # c2
        dyp1dc2 = np.array([[-c0x*(b3x*(c2x**2-c2y**2-c2z**2)+2*c2x*(b3y*c2y+b3z*c2z)),c0x*(b3y*(c2x**2-c2y**2+c2z**2)-2*c2y*(b3x*c2x+b3z*c2z)),c0x*(b3z*(c2x**2+c2y**2-c2z**2)-2*c2z*(b3x*c2x+b3y*c2y))],
                            [-c0y*(b3x*(c2x**2-c2y**2-c2z**2)+2*c2x*(b3y*c2y+b3z*c2z)),c0y*(b3y*(c2x**2-c2y**2+c2z**2)-2*c2y*(b3x*c2x+b3z*c2z)),c0y*(b3z*(c2x**2+c2y**2-c2z**2)-2*c2z*(b3x*c2x+b3y*c2y))],
                            [-c0z*(b3x*(c2x**2-c2y**2-c2z**2)+2*c2x*(b3y*c2y+b3z*c2z)),c0z*(b3y*(c2x**2-c2y**2+c2z**2)-2*c2y*(b3x*c2x+b3z*c2z)),c0z*(b3z*(c2x**2+c2y**2-c2z**2)-2*c2z*(b3x*c2x+b3y*c2y))]])/(3*c2c2**2)
        d2yp1dc2db3 = d2yp1db3dc2.swapaxes(1,2)
        d2yp1dc2dc0 = d2yp1dc0dc2.swapaxes(1,2)
        d2yp1d2c2   = np.array([[[2*c0x*(b3x*(c2x**3-3*c2x*(c2y**2+c2z**2))-(-3*c2x**2+c2y**2+c2z**2)*(b3y*c2y+b3z*c2z)),-2*c0x*(b3x*c2y*(-3*c2x**2+c2y**2+c2z**2)+b3y*c2x*(c2x**2-3*c2y**2+c2z**2)-4*b3z*c2x*c2y*c2z),-2*c0x*(b3x*c2z*(-3*c2x**2+c2y**2+c2z**2)-4*b3y*c2x*c2y*c2z+b3z*c2x*(c2x**2+c2y**2-3*c2z**2))],[-2*c0x*(b3x*c2y*(-3*c2x**2+c2y**2+c2z**2)+b3y*c2x*(c2x**2-3*c2y**2+c2z**2)-4*b3z*c2x*c2y*c2z),-2*c0x*(b3x*c2x*(c2x**2-3*c2y**2+c2z**2)+b3y*c2y*(3*c2x**2-c2y**2+3*c2z**2)+b3z*c2z*(c2x**2-3*c2y**2+c2z**2)),-2*c0x*(-4*b3x*c2x*c2y*c2z+b3y*c2z*(c2x**2-3*c2y**2+c2z**2)+b3z*c2y*(c2x**2+c2y**2-3*c2z**2))],[-2*c0x*(b3x*c2z*(-3*c2x**2+c2y**2+c2z**2)-4*b3y*c2x*c2y*c2z+b3z*c2x*(c2x**2+c2y**2-3*c2z**2)),-2*c0x*(-4*b3x*c2x*c2y*c2z+b3y*c2z*(c2x**2-3*c2y**2+c2z**2)+b3z*c2y*(c2x**2+c2y**2-3*c2z**2)),-2*c0x*(b3x*c2x*(c2x**2+c2y**2-3*c2z**2)+b3y*c2y*(c2x**2+c2y**2-3*c2z**2)+b3z*c2z*(3*c2x**2+3*c2y**2-c2z**2))]],
                                [[2*c0y*(b3x*(c2x**3-3*c2x*(c2y**2+c2z**2))-(-3*c2x**2+c2y**2+c2z**2)*(b3y*c2y+b3z*c2z)),-2*c0y*(b3x*c2y*(-3*c2x**2+c2y**2+c2z**2)+b3y*c2x*(c2x**2-3*c2y**2+c2z**2)-4*b3z*c2x*c2y*c2z),-2*c0y*(b3x*c2z*(-3*c2x**2+c2y**2+c2z**2)-4*b3y*c2x*c2y*c2z+b3z*c2x*(c2x**2+c2y**2-3*c2z**2))],[-2*c0y*(b3x*c2y*(-3*c2x**2+c2y**2+c2z**2)+b3y*c2x*(c2x**2-3*c2y**2+c2z**2)-4*b3z*c2x*c2y*c2z),-2*c0y*(b3x*c2x*(c2x**2-3*c2y**2+c2z**2)+b3y*c2y*(3*c2x**2-c2y**2+3*c2z**2)+b3z*c2z*(c2x**2-3*c2y**2+c2z**2)),-2*c0y*(-4*b3x*c2x*c2y*c2z+b3y*c2z*(c2x**2-3*c2y**2+c2z**2)+b3z*c2y*(c2x**2+c2y**2-3*c2z**2))],[-2*c0y*(b3x*c2z*(-3*c2x**2+c2y**2+c2z**2)-4*b3y*c2x*c2y*c2z+b3z*c2x*(c2x**2+c2y**2-3*c2z**2)),-2*c0y*(-4*b3x*c2x*c2y*c2z+b3y*c2z*(c2x**2-3*c2y**2+c2z**2)+b3z*c2y*(c2x**2+c2y**2-3*c2z**2)),-2*c0y*(b3x*c2x*(c2x**2+c2y**2-3*c2z**2)+b3y*c2y*(c2x**2+c2y**2-3*c2z**2)+b3z*c2z*(3*c2x**2+3*c2y**2-c2z**2))]],
                                [[2*c0z*(b3x*(c2x**3-3*c2x*(c2y**2+c2z**2))-(-3*c2x**2+c2y**2+c2z**2)*(b3y*c2y+b3z*c2z)),-2*c0z*(b3x*c2y*(-3*c2x**2+c2y**2+c2z**2)+b3y*c2x*(c2x**2-3*c2y**2+c2z**2)-4*b3z*c2x*c2y*c2z),-2*c0z*(b3x*c2z*(-3*c2x**2+c2y**2+c2z**2)-4*b3y*c2x*c2y*c2z+b3z*c2x*(c2x**2+c2y**2-3*c2z**2))],[-2*c0z*(b3x*c2y*(-3*c2x**2+c2y**2+c2z**2)+b3y*c2x*(c2x**2-3*c2y**2+c2z**2)-4*b3z*c2x*c2y*c2z),-2*c0z*(b3x*c2x*(c2x**2-3*c2y**2+c2z**2)+b3y*c2y*(3*c2x**2-c2y**2+3*c2z**2)+b3z*c2z*(c2x**2-3*c2y**2+c2z**2)),-2*c0z*(-4*b3x*c2x*c2y*c2z+b3y*c2z*(c2x**2-3*c2y**2+c2z**2)+b3z*c2y*(c2x**2+c2y**2-3*c2z**2))],[-2*c0z*(b3x*c2z*(-3*c2x**2+c2y**2+c2z**2)-4*b3y*c2x*c2y*c2z+b3z*c2x*(c2x**2+c2y**2-3*c2z**2)),-2*c0z*(-4*b3x*c2x*c2y*c2z+b3y*c2z*(c2x**2-3*c2y**2+c2z**2)+b3z*c2y*(c2x**2+c2y**2-3*c2z**2)),-2*c0z*(b3x*c2x*(c2x**2+c2y**2-3*c2z**2)+b3y*c2y*(c2x**2+c2y**2-3*c2z**2)+b3z*c2z*(3*c2x**2+3*c2y**2-c2z**2))]]])/(3*c2c2**3)


        ###############
        # Derivatives d(var)/du , d2(var)/d2u
        ###############
        # da0du
        da0du = np.zeros((3,3*Ne))
        da0dm0 = np.array([[((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(-m0x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m0y*(y0x-y3x)*(y0y-y3y)+m0z*(y0x-y3x)*(y0z-y3z)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(-m0x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m0y*(y0x-y3x)*(y0y-y3y)+m0z*(y0x-y3x)*(y0z-y3z)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(-m0x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m0y*(y0x-y3x)*(y0y-y3y)+m0z*(y0x-y3x)*(y0z-y3z)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)],
                           [((m0y*(y0z-y3z)+m0z*(y3y-y0y))*(m0y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)-(y0y-y3y)*(m0x*(y0x-y3x)+m0z*(y0z-y3z))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*((y0y-y3y)*(m0x*(y0x-y3x)+m0z*(y0z-y3z))-m0y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),-(((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)-(y0y-y3y)*(m0x*(y0x-y3x)+m0z*(y0z-y3z))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2))],
                           [-(((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)-(y0z-y3z)*(m0x*(y0x-y3x)+m0y*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)-(y0z-y3z)*(m0x*(y0x-y3x)+m0y*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*((y0z-y3z)*(m0x*(y0x-y3x)+m0y*(y0y-y3y))-m0z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)]])
        da0dy0 = np.array([[-(((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2))],
                           [((m0y*(y0z-y3z)+m0z*(y3y-y0y))*(m0x**2*(y0y-y3y)+m0x*m0y*(y3x-y0x)+m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x**2*(y0y-y3y)+m0x*m0y*(y3x-y0x)+m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x**2*(y3y-y0y)+m0x*m0y*(y0x-y3x)+m0z*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)],
                           [((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x**2*(y3z-y0z)+m0x*m0z*(y0x-y3x)+m0y*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x**2*(y3z-y0z)+m0x*m0z*(y0x-y3x)+m0y*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),-(((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x**2*(y0z-y3z)+m0x*m0z*(y3x-y0x)+m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2))]])
        da0dy3 = np.array([[((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x*m0y*(y3y-y0y)+m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+m0y**2*(y0x-y3x)))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)],
                           [((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x**2*(y0y-y3y)+m0x*m0y*(y3x-y0x)+m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),-(((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x**2*(y0y-y3y)+m0x*m0y*(y3x-y0x)+m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)),-(((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x**2*(y3y-y0y)+m0x*m0y*(y0x-y3x)+m0z*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2))],
                           [((m0y*(y3z-y0z)+m0z*(y0y-y3y))*(m0x**2*(y0z-y3z)+m0x*m0z*(y3x-y0x)+m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3z-y0z)+m0z*(y0x-y3x))*(m0x**2*(y3z-y0z)+m0x*m0z*(y0x-y3x)+m0y*(m0y*(y3z-y0z)+m0z*(y0y-y3y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2),((m0x*(y3y-y0y)+m0y*(y0x-y3x))*(m0x**2*(y0z-y3z)+m0x*m0z*(y3x-y0x)+m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y))))/((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)**(3/2)]])
        da0du[ : , 3*idx0:3*(idx0+1)] += da0dy0
        da0du[ : , 3*idx3:3*(idx3+1)] += da0dy3
        da0du                         += da0dm0@dm0du
        # d2a0d2u
        d2a0d2u = np.zeros((3,3*Ne,3*Ne))
        d2a0d2y0   = np.array([[[(m0y*(y3z-y0z)+m0z*(y0y-y3y))*(3*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))**2-4*(m0y**2+m0z**2)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)),3*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))+4*m0x*m0y*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)-2*m0z*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),3*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))+2*m0y*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))+4*m0x*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)],
                                [3*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+4*m0x*m0y*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*m0z*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))**2-4*m0z*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-4*(m0x**2+m0z**2)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+2*m0y*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*m0z*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+4*m0y*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))+2*m0y*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))+4*m0x*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+2*m0y*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*m0z*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+4*m0y*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))**2+4*m0y*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))-4*(m0x**2+m0y**2)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)]],
                               [[3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))**2+4*m0z*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))-4*(m0y**2+m0z**2)*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+2*m0z*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+4*m0x*m0y*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))-2*m0x*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))+2*m0z*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+4*m0x*m0z*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [3*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))+2*m0z*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))+4*m0x*m0y*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(3*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))**2-4*(m0x**2+m0z**2)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)),3*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*m0x*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))+4*m0y*m0z*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)],
                                [3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))-2*m0x*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))+2*m0z*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+4*m0x*m0z*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*m0x*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+4*m0y*m0z*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))**2-4*m0x*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))-4*(m0x**2+m0y**2)*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)]],
                               [[3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))**2-4*m0y*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))-4*(m0y**2+m0z**2)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*m0y*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+4*m0x*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+2*m0x*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))+4*m0x*m0z*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*m0y*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*m0y*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+4*m0x*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+2*m0x*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))**2+4*m0x*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-4*(m0x**2+m0z**2)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+4*m0y*m0z*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+2*m0x*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [3*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))+4*m0x*m0z*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)-2*m0y*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),3*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))+4*m0y*m0z*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)+2*m0x*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(3*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))**2-4*(m0x**2+m0y**2)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2))]]])/(4*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)**(5/2))
        d2a0dy0dy3 = -d2a0d2y0
        d2a0dy0dm0 = np.array([[[(m0y*(y3z-y0z)+m0z*(y0y-y3y))*(3*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))-2*(-2*m0y*y0y+2*m0y*y3y-2*m0z*y0z+2*m0z*y3z)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)),3*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))-2*(y3z-y0z)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))-2*(2*m0x*(y3y-y0y)+4*m0y*(y0x-y3x))*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),3*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(2*m0x*(y3z-y0z)+4*m0z*(y0x-y3x))*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)-2*(y0y-y3y)*(2*m0y*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*m0z*(m0x*(y3z-y0z)+m0z*(y0x-y3x)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)],
                                [3*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*m0z*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(-2*m0y*y0x+4*m0x*y0y+2*m0y*y3x-4*m0x*y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y3z-y0z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*m0z*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(-2*m0x*y0x-2*m0z*y0z+2*m0x*y3x+2*m0z*y3z)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),4*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)**2-2*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(4*m0z*(y0y-y3y)+2*m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(y0y-y3y)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*m0z*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+3*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))],
                                [3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))-2*(-2*m0z*y0x+4*m0x*y0z+2*m0z*y3x-4*m0x*y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))+2*m0y*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),-4*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)**2+2*m0y*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(y3z-y0z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(-2*m0z*y0y+4*m0y*y0z+2*m0z*y3y-4*m0y*y3z)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(m0z*(y0y-y3y)+m0y*(y3z-y0z)),3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+2*m0y*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y0y-y3y)*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(-2*m0x*y0x-2*m0y*y0y+2*m0x*y3x+2*m0y*y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)]],
                               [[3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))-2*(y0z-y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))+2*m0z*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(-2*m0y*y0y-2*m0z*y0z+2*m0y*y3y+2*m0z*y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))-2*(4*m0y*(y0x-y3x)+2*m0x*(y3y-y0y))*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+2*m0z*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),-4*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)**2-2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(4*m0z*(y0x-y3x)+2*m0x*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(y3x-y0x)*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+2*m0z*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))],
                                [3*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(y0z-y3z)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(4*m0x*y0y-4*m0x*y3y-2*m0y*y0x+2*m0y*y3x)*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(3*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(-2*m0x*y0x+2*m0x*y3x-2*m0z*y0z+2*m0z*y3z)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)),3*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(2*m0y*(y3z-y0z)+4*m0z*(y0y-y3y))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)-2*(y3x-y0x)*(2*m0x*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*m0z*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)],
                                [4*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)**2-2*m0x*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(y0z-y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(-2*m0z*y0x+4*m0x*y0z+2*m0z*y3x-4*m0x*y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z)),3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-2*m0x*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(-2*m0z*y0y+4*m0y*y0z+2*m0z*y3y-4*m0y*y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*m0x*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y3x-y0x)*(2*m0x*(m0z*(y3x-y0x)+m0x*(y0z-y3z))+2*m0y*(m0z*(y3y-y0y)+m0y*(y0z-y3z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(-2*m0x*y0x-2*m0y*y0y+2*m0x*y3x+2*m0y*y3y)*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)]],
                               [[3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))-2*(y3y-y0y)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))-2*m0y*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(-2*m0y*y0y-2*m0z*y0z+2*m0y*y3y+2*m0z*y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),4*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)**2-2*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(4*m0y*(y0x-y3x)+2*m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*m0y*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(y0x-y3x)*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z))),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*m0y*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*m0z*(m0z*(y0x-y3x)+m0x*(y3z-y0z)))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*m0y*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(4*m0z*(y0x-y3x)+2*m0x*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [-4*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)**2-2*(-2*m0y*y0x+4*m0x*y0y+2*m0y*y3x-4*m0x*y3y)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+2*m0x*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(y3y-y0y)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)+3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z))),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y0x-y3x)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+2*m0x*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(-2*m0x*y0x-2*m0z*y0z+2*m0x*y3x+2*m0z*y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*m0x*(m0y*(y3x-y0x)+m0x*(y0y-y3y))+2*m0z*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))+2*m0x*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(4*m0z*(y0y-y3y)+2*m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [3*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))-2*(y3y-y0y)*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)-2*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(4*m0x*y0z-4*m0x*y3z-2*m0z*y0x+2*m0z*y3x)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),3*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))-2*(y0x-y3x)*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)-2*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(4*m0y*y0z-4*m0y*y3z-2*m0z*y0y+2*m0z*y3y)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(3*(2*m0x*(m0x*(y0z-y3z)+m0z*(y3x-y0x))+2*m0y*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(-2*m0x*y0x+2*m0x*y3x-2*m0y*y0y+2*m0y*y3y)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2))]]])/(4*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)**(5/2))
        d2a0dy3dy0 = d2a0dy0dy3.swapaxes(1,2)
        d2a0d2y3   = d2a0d2y0
        d2a0dy3dm0 = -d2a0dy0dm0
        d2a0dm0dy0 = d2a0dy0dm0.swapaxes(1,2)
        d2a0dm0dy3 = -d2a0dm0dy0
        d2a0d2m0   = np.array([[[(m0y*(y3z-y0z)+m0z*(y0y-y3y))*(3*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))**2-4*((y0y-y3y)**2+(y0z-y3z)**2)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)),3*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))-4*(y0x-y3x)*(y3y-y0y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)-2*(y3z-y0z)*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),3*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(y0y-y3y)*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)-4*(y3x-y0x)*(y0z-y3z)*(m0y*(y3z-y0z)+m0z*(y0y-y3y))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)],
                                [3*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))-4*(y0x-y3x)*(y3y-y0y)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))-2*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(y3z-y0z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))**2-4*(y3z-y0z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-4*((y0x-y3x)**2+(y0z-y3z)**2)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y3z-y0z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y0y-y3y)*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-4*(y0y-y3y)*(y3z-y0z)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [3*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y0y-y3y)*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-4*(y3x-y0x)*(y0z-y3z)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y3z-y0z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y0y-y3y)*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-4*(y0y-y3y)*(y3z-y0z)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))**2-4*(y0y-y3y)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-4*((y0x-y3x)**2+(y0y-y3y)**2)*(m0z*(y0y-y3y)+m0y*(y3z-y0z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)]],
                               [[3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))**2-4*(y0z-y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))-4*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((y0y-y3y)**2+(y0z-y3z)**2)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-2*(y0z-y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-4*(y0x-y3x)*(y3y-y0y)*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y0z-y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y3x-y0x)*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-4*(y3x-y0x)*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [3*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))-2*(y0z-y3z)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))-4*(y0x-y3x)*(y3y-y0y)*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(3*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))**2-4*((y0x-y3x)**2+(y0z-y3z)**2)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)),3*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(y3x-y0x)*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)-4*(y0y-y3y)*(y3z-y0z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)],
                                [3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y0z-y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y3x-y0x)*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-4*(y3x-y0x)*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y3x-y0x)*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-4*(y0y-y3y)*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y3z-y0z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))**2-4*(y3x-y0x)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-4*((y0x-y3x)**2+(y0y-y3y)**2)*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)]],
                               [[3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))**2-4*(y3y-y0y)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))-4*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((y0y-y3y)**2+(y0z-y3z)**2)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-2*(y3y-y0y)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-4*(y0x-y3x)*(y3y-y0y)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(y0x-y3x)*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y3y-y0y)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-4*(y3x-y0x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(y0z-y3z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-2*(y3y-y0y)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-4*(y0x-y3x)*(y3y-y0y)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)-2*(y0x-y3x)*(2*(m0y*(y3x-y0x)+m0x*(y0y-y3y))*(y0y-y3y)+2*(m0z*(y3x-y0x)+m0x*(y0z-y3z))*(y0z-y3z))*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))**2-4*(y0x-y3x)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))-4*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*((y0x-y3x)**2+(y0z-y3z)**2)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2),3*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(2*(y0x-y3x)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))+2*(m0z*(y3y-y0y)+m0y*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-2*(y0x-y3x)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m0z*(y0x-y3x)+m0x*(y3z-y0z))+2*(y0y-y3y)*(m0z*(y0y-y3y)+m0y*(y3z-y0z)))-4*(y0y-y3y)*(m0y*(y0x-y3x)+m0x*(y3y-y0y))*(y3z-y0z)*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)],
                                [3*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(2*(y0y-y3y)*(m0x*(y0y-y3y)+m0y*(y3x-y0x))+2*(y0z-y3z)*(m0x*(y0z-y3z)+m0z*(y3x-y0x)))*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(y3y-y0y)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-4*(y3x-y0x)*(y0z-y3z)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),3*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(2*(y0x-y3x)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))+2*(y0z-y3z)*(m0y*(y0z-y3z)+m0z*(y3y-y0y)))*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-2*(y0x-y3x)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2)*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))-4*(y0y-y3y)*(y3z-y0z)*(m0x*(y3y-y0y)+m0y*(y0x-y3x))*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2),(m0x*(y3y-y0y)+m0y*(y0x-y3x))*(3*(2*(y0x-y3x)*(m0x*(y3z-y0z)+m0z*(y0x-y3x))+2*(y0y-y3y)*(m0y*(y3z-y0z)+m0z*(y0y-y3y)))**2-4*((y0x-y3x)**2+(y0y-y3y)**2)*((m0x*(y3y-y0y)+m0y*(y0x-y3x))**2+(m0x*(y3z-y0z)+m0z*(y0x-y3x))**2+(m0y*(y3z-y0z)+m0z*(y0y-y3y))**2))]]])/(4*((m0y*(y0x-y3x)+m0x*(y3y-y0y))**2+(m0z*(y0x-y3x)+m0x*(y3z-y0z))**2+(m0z*(y0y-y3y)+m0y*(y3z-y0z))**2)**(5/2))
        d2a0d2u[ : , 3*idx0:3*(idx0+1) , 3*idx0:3*(idx0+1)] +=  d2a0d2y0
        d2a0d2u[ : , 3*idx0:3*(idx0+1) , 3*idx3:3*(idx3+1)] +=  d2a0dy0dy3
        d2a0d2u[ : , 3*idx0:3*(idx0+1) , : ] +=  np.tensordot(d2a0dy0dm0,dm0du,axes=[2,0])
        d2a0d2u[ : , 3*idx3:3*(idx3+1) , 3*idx0:3*(idx0+1)] +=  d2a0dy3dy0
        d2a0d2u[ : , 3*idx3:3*(idx3+1) , 3*idx3:3*(idx3+1)] +=  d2a0d2y3
        d2a0d2u[ : , 3*idx3:3*(idx3+1) , : ] +=  np.tensordot(d2a0dy3dm0,dm0du,axes=[2,0])
        d2a0d2u[ : , : , 3*idx0:3*(idx0+1)] +=  np.tensordot(d2a0dm0dy0,dm0du,axes=[1,0]).swapaxes(1,2)
        d2a0d2u[ : , : , 3*idx3:3*(idx3+1)] +=  np.tensordot(d2a0dm0dy3,dm0du,axes=[1,0]).swapaxes(1,2)
        d2a0d2u += np.tensordot(np.tensordot(d2a0d2m0,dm0du,axes=1),dm0du,axes=[1,0])
        d2a0d2u += np.tensordot(da0dm0,d2m0d2u,axes=[1,0])
        # da3du
        da3du = np.zeros((3,3*Ne))
        da3dm3 = np.array([[((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(-m3x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m3y*(y0x-y3x)*(y0y-y3y)+m3z*(y0x-y3x)*(y0z-y3z)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(-m3x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m3y*(y0x-y3x)*(y0y-y3y)+m3z*(y0x-y3x)*(y0z-y3z)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(-m3x*(y0y**2-2*y0y*y3y+y0z**2-2*y0z*y3z+y3y**2+y3z**2)+m3y*(y0x-y3x)*(y0y-y3y)+m3z*(y0x-y3x)*(y0z-y3z)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)],
                           [((m3y*(y0z-y3z)+m3z*(y3y-y0y))*(m3y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)-(y0y-y3y)*(m3x*(y0x-y3x)+m3z*(y0z-y3z))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*((y0y-y3y)*(m3x*(y0x-y3x)+m3z*(y0z-y3z))-m3y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),-(((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3y*(y0x**2-2*y0x*y3x+y0z**2-2*y0z*y3z+y3x**2+y3z**2)-(y0y-y3y)*(m3x*(y0x-y3x)+m3z*(y0z-y3z))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2))],
                           [-(((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)-(y0z-y3z)*(m3x*(y0x-y3x)+m3y*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)-(y0z-y3z)*(m3x*(y0x-y3x)+m3y*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*((y0z-y3z)*(m3x*(y0x-y3x)+m3y*(y0y-y3y))-m3z*(y0x**2-2*y0x*y3x+y0y**2-2*y0y*y3y+y3x**2+y3y**2)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)]])
        da3dy0 = np.array([[-(((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2))],
                           [((m3y*(y0z-y3z)+m3z*(y3y-y0y))*(m3x**2*(y0y-y3y)+m3x*m3y*(y3x-y0x)+m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x**2*(y0y-y3y)+m3x*m3y*(y3x-y0x)+m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x**2*(y3y-y0y)+m3x*m3y*(y0x-y3x)+m3z*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)],
                           [((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x**2*(y3z-y0z)+m3x*m3z*(y0x-y3x)+m3y*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x**2*(y3z-y0z)+m3x*m3z*(y0x-y3x)+m3y*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),-(((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x**2*(y0z-y3z)+m3x*m3z*(y3x-y0x)+m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2))]])
        da3dy3 = np.array([[((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x*m3y*(y3y-y0y)+m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+m3y**2*(y0x-y3x)))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)],
                           [((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x**2*(y0y-y3y)+m3x*m3y*(y3x-y0x)+m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),-(((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x**2*(y0y-y3y)+m3x*m3y*(y3x-y0x)+m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)),-(((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x**2*(y3y-y0y)+m3x*m3y*(y0x-y3x)+m3z*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2))],
                           [((m3y*(y3z-y0z)+m3z*(y0y-y3y))*(m3x**2*(y0z-y3z)+m3x*m3z*(y3x-y0x)+m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3z-y0z)+m3z*(y0x-y3x))*(m3x**2*(y3z-y0z)+m3x*m3z*(y0x-y3x)+m3y*(m3y*(y3z-y0z)+m3z*(y0y-y3y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2),((m3x*(y3y-y0y)+m3y*(y0x-y3x))*(m3x**2*(y0z-y3z)+m3x*m3z*(y3x-y0x)+m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y))))/((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)**(3/2)]])
        da3du[ : , 3*idx0:3*(idx0+1)] += da3dy0
        da3du[ : , 3*idx3:3*(idx3+1)] += da3dy3
        da3du                         += da3dm3@dm3du
        # d2a3d2u
        d2a3d2u = np.zeros((3,3*Ne,3*Ne))
        d2a3d2y0   = np.array([[[(m3y*(y3z-y0z)+m3z*(y0y-y3y))*(3*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))**2-4*(m3y**2+m3z**2)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)),3*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))+4*m3x*m3y*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)-2*m3z*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),3*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))+2*m3y*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))+4*m3x*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)],
                                [3*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+4*m3x*m3y*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*m3z*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))**2-4*m3z*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-4*(m3x**2+m3z**2)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+2*m3y*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*m3z*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+4*m3y*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))+2*m3y*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))+4*m3x*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+2*m3y*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*m3z*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+4*m3y*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))**2+4*m3y*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))-4*(m3x**2+m3y**2)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)]],
                               [[3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))**2+4*m3z*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))-4*(m3y**2+m3z**2)*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+2*m3z*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+4*m3x*m3y*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))-2*m3x*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))+2*m3z*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+4*m3x*m3z*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [3*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))+2*m3z*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))+4*m3x*m3y*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(3*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))**2-4*(m3x**2+m3z**2)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)),3*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*m3x*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))+4*m3y*m3z*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)],
                                [3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))-2*m3x*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))+2*m3z*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+4*m3x*m3z*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*m3x*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+4*m3y*m3z*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))**2-4*m3x*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))-4*(m3x**2+m3y**2)*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)]],
                               [[3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))**2-4*m3y*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))-4*(m3y**2+m3z**2)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*m3y*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+4*m3x*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+2*m3x*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))+4*m3x*m3z*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*m3y*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*m3y*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+4*m3x*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+2*m3x*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))**2+4*m3x*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-4*(m3x**2+m3z**2)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+4*m3y*m3z*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+2*m3x*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [3*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))+4*m3x*m3z*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)-2*m3y*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),3*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))+4*m3y*m3z*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)+2*m3x*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(3*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))**2-4*(m3x**2+m3y**2)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2))]]])/(4*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)**(5/2))
        d2a3dy0dy3 = -d2a3d2y0
        d2a3dy0dm3 = np.array([[[(m3y*(y3z-y0z)+m3z*(y0y-y3y))*(3*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))-2*(-2*m3y*y0y+2*m3y*y3y-2*m3z*y0z+2*m3z*y3z)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)),3*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))-2*(y3z-y0z)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))-2*(2*m3x*(y3y-y0y)+4*m3y*(y0x-y3x))*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),3*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(2*m3x*(y3z-y0z)+4*m3z*(y0x-y3x))*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)-2*(y0y-y3y)*(2*m3y*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*m3z*(m3x*(y3z-y0z)+m3z*(y0x-y3x)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)],
                                [3*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*m3z*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(-2*m3y*y0x+4*m3x*y0y+2*m3y*y3x-4*m3x*y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y3z-y0z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*m3z*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(-2*m3x*y0x-2*m3z*y0z+2*m3x*y3x+2*m3z*y3z)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),4*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)**2-2*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(4*m3z*(y0y-y3y)+2*m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(y0y-y3y)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*m3z*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+3*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))],
                                [3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))-2*(-2*m3z*y0x+4*m3x*y0z+2*m3z*y3x-4*m3x*y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))+2*m3y*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),-4*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)**2+2*m3y*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(y3z-y0z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(-2*m3z*y0y+4*m3y*y0z+2*m3z*y3y-4*m3y*y3z)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(m3z*(y0y-y3y)+m3y*(y3z-y0z)),3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+2*m3y*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y0y-y3y)*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(-2*m3x*y0x-2*m3y*y0y+2*m3x*y3x+2*m3y*y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)]],
                               [[3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))-2*(y0z-y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))+2*m3z*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(-2*m3y*y0y-2*m3z*y0z+2*m3y*y3y+2*m3z*y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))-2*(4*m3y*(y0x-y3x)+2*m3x*(y3y-y0y))*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+2*m3z*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),-4*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)**2-2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(4*m3z*(y0x-y3x)+2*m3x*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(y3x-y0x)*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+2*m3z*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))],
                                [3*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(y0z-y3z)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(4*m3x*y0y-4*m3x*y3y-2*m3y*y0x+2*m3y*y3x)*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(3*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(-2*m3x*y0x+2*m3x*y3x-2*m3z*y0z+2*m3z*y3z)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)),3*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(2*m3y*(y3z-y0z)+4*m3z*(y0y-y3y))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)-2*(y3x-y0x)*(2*m3x*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*m3z*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)],
                                [4*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)**2-2*m3x*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(y0z-y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(-2*m3z*y0x+4*m3x*y0z+2*m3z*y3x-4*m3x*y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z)),3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-2*m3x*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(-2*m3z*y0y+4*m3y*y0z+2*m3z*y3y-4*m3y*y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*m3x*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y3x-y0x)*(2*m3x*(m3z*(y3x-y0x)+m3x*(y0z-y3z))+2*m3y*(m3z*(y3y-y0y)+m3y*(y0z-y3z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(-2*m3x*y0x-2*m3y*y0y+2*m3x*y3x+2*m3y*y3y)*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)]],
                               [[3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))-2*(y3y-y0y)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))-2*m3y*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(-2*m3y*y0y-2*m3z*y0z+2*m3y*y3y+2*m3z*y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),4*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)**2-2*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(4*m3y*(y0x-y3x)+2*m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*m3y*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(y0x-y3x)*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z))),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*m3y*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*m3z*(m3z*(y0x-y3x)+m3x*(y3z-y0z)))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*m3y*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(4*m3z*(y0x-y3x)+2*m3x*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [-4*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)**2-2*(-2*m3y*y0x+4*m3x*y0y+2*m3y*y3x-4*m3x*y3y)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+2*m3x*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(y3y-y0y)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)+3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z))),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y0x-y3x)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+2*m3x*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(-2*m3x*y0x-2*m3z*y0z+2*m3x*y3x+2*m3z*y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*m3x*(m3y*(y3x-y0x)+m3x*(y0y-y3y))+2*m3z*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))+2*m3x*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(4*m3z*(y0y-y3y)+2*m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [3*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))-2*(y3y-y0y)*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)-2*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(4*m3x*y0z-4*m3x*y3z-2*m3z*y0x+2*m3z*y3x)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),3*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))-2*(y0x-y3x)*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)-2*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(4*m3y*y0z-4*m3y*y3z-2*m3z*y0y+2*m3z*y3y)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(3*(2*m3x*(m3x*(y0z-y3z)+m3z*(y3x-y0x))+2*m3y*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(-2*m3x*y0x+2*m3x*y3x-2*m3y*y0y+2*m3y*y3y)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2))]]])/(4*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)**(5/2))
        d2a3dy3dy0 = d2a3dy0dy3.swapaxes(1,2)
        d2a3d2y3   = d2a3d2y0
        d2a3dy3dm3 = -d2a3dy0dm3
        d2a3dm3dy0 = d2a3dy0dm3.swapaxes(1,2)
        d2a3dm3dy3 = -d2a3dm3dy0
        d2a3d2m3   = np.array([[[(m3y*(y3z-y0z)+m3z*(y0y-y3y))*(3*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))**2-4*((y0y-y3y)**2+(y0z-y3z)**2)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)),3*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))-4*(y0x-y3x)*(y3y-y0y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)-2*(y3z-y0z)*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),3*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(y0y-y3y)*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)-4*(y3x-y0x)*(y0z-y3z)*(m3y*(y3z-y0z)+m3z*(y0y-y3y))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)],
                                [3*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))-4*(y0x-y3x)*(y3y-y0y)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))-2*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(y3z-y0z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))**2-4*(y3z-y0z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-4*((y0x-y3x)**2+(y0z-y3z)**2)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y3z-y0z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y0y-y3y)*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-4*(y0y-y3y)*(y3z-y0z)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [3*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y0y-y3y)*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-4*(y3x-y0x)*(y0z-y3z)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y3z-y0z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y0y-y3y)*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-4*(y0y-y3y)*(y3z-y0z)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))**2-4*(y0y-y3y)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-4*((y0x-y3x)**2+(y0y-y3y)**2)*(m3z*(y0y-y3y)+m3y*(y3z-y0z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)]],
                               [[3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))**2-4*(y0z-y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))-4*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((y0y-y3y)**2+(y0z-y3z)**2)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-2*(y0z-y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-4*(y0x-y3x)*(y3y-y0y)*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y0z-y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y3x-y0x)*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-4*(y3x-y0x)*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [3*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))-2*(y0z-y3z)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))-4*(y0x-y3x)*(y3y-y0y)*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(3*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))**2-4*((y0x-y3x)**2+(y0z-y3z)**2)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)),3*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(y3x-y0x)*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)-4*(y0y-y3y)*(y3z-y0z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)],
                                [3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y0z-y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y3x-y0x)*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-4*(y3x-y0x)*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y3x-y0x)*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-4*(y0y-y3y)*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y3z-y0z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))**2-4*(y3x-y0x)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-4*((y0x-y3x)**2+(y0y-y3y)**2)*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)]],
                               [[3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))**2-4*(y3y-y0y)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))-4*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((y0y-y3y)**2+(y0z-y3z)**2)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-2*(y3y-y0y)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-4*(y0x-y3x)*(y3y-y0y)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(y0x-y3x)*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y3y-y0y)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-4*(y3x-y0x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(y0z-y3z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-2*(y3y-y0y)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-4*(y0x-y3x)*(y3y-y0y)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)-2*(y0x-y3x)*(2*(m3y*(y3x-y0x)+m3x*(y0y-y3y))*(y0y-y3y)+2*(m3z*(y3x-y0x)+m3x*(y0z-y3z))*(y0z-y3z))*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))**2-4*(y0x-y3x)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))-4*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*((y0x-y3x)**2+(y0z-y3z)**2)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2),3*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(2*(y0x-y3x)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))+2*(m3z*(y3y-y0y)+m3y*(y0z-y3z))*(y0z-y3z))*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-2*(y0x-y3x)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)*(2*(y0x-y3x)*(m3z*(y0x-y3x)+m3x*(y3z-y0z))+2*(y0y-y3y)*(m3z*(y0y-y3y)+m3y*(y3z-y0z)))-4*(y0y-y3y)*(m3y*(y0x-y3x)+m3x*(y3y-y0y))*(y3z-y0z)*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)],
                                [3*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(2*(y0y-y3y)*(m3x*(y0y-y3y)+m3y*(y3x-y0x))+2*(y0z-y3z)*(m3x*(y0z-y3z)+m3z*(y3x-y0x)))*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(y3y-y0y)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-4*(y3x-y0x)*(y0z-y3z)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),3*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(2*(y0x-y3x)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))+2*(y0z-y3z)*(m3y*(y0z-y3z)+m3z*(y3y-y0y)))*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-2*(y0x-y3x)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2)*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))-4*(y0y-y3y)*(y3z-y0z)*(m3x*(y3y-y0y)+m3y*(y0x-y3x))*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2),(m3x*(y3y-y0y)+m3y*(y0x-y3x))*(3*(2*(y0x-y3x)*(m3x*(y3z-y0z)+m3z*(y0x-y3x))+2*(y0y-y3y)*(m3y*(y3z-y0z)+m3z*(y0y-y3y)))**2-4*((y0x-y3x)**2+(y0y-y3y)**2)*((m3x*(y3y-y0y)+m3y*(y0x-y3x))**2+(m3x*(y3z-y0z)+m3z*(y0x-y3x))**2+(m3y*(y3z-y0z)+m3z*(y0y-y3y))**2))]]])/(4*((m3y*(y0x-y3x)+m3x*(y3y-y0y))**2+(m3z*(y0x-y3x)+m3x*(y3z-y0z))**2+(m3z*(y0y-y3y)+m3y*(y3z-y0z))**2)**(5/2))
        d2a3d2u[ : , 3*idx0:3*(idx0+1) , 3*idx0:3*(idx0+1)] +=  d2a3d2y0
        d2a3d2u[ : , 3*idx0:3*(idx0+1) , 3*idx3:3*(idx3+1)] +=  d2a3dy0dy3
        d2a3d2u[ : , 3*idx0:3*(idx0+1) , : ] +=  np.tensordot(d2a3dy0dm3,dm3du,axes=[2,0])
        d2a3d2u[ : , 3*idx3:3*(idx3+1) , 3*idx0:3*(idx0+1)] +=  d2a3dy3dy0
        d2a3d2u[ : , 3*idx3:3*(idx3+1) , 3*idx3:3*(idx3+1)] +=  d2a3d2y3
        d2a3d2u[ : , 3*idx3:3*(idx3+1) , : ] +=  np.tensordot(d2a3dy3dm3,dm3du,axes=[2,0])
        d2a3d2u[ : , : , 3*idx0:3*(idx0+1)] +=  np.tensordot(d2a3dm3dy0,dm3du,axes=[1,0]).swapaxes(1,2)
        d2a3d2u[ : , : , 3*idx3:3*(idx3+1)] +=  np.tensordot(d2a3dm3dy3,dm3du,axes=[1,0]).swapaxes(1,2)
        d2a3d2u += np.tensordot(np.tensordot(d2a3d2m3,dm3du,axes=1),dm3du,axes=[1,0])
        d2a3d2u += np.tensordot(da3dm3,d2m3d2u,axes=[1,0])
        # db0du
        db0du                       = np.array(dy00du)
        db0du[:,3*idx0:3*(idx0+1)] -= np.eye(3)
        # d2b0du
        d2b0d2u = np.array(d2y00d2u)
        # db3du
        db3du                       = np.array(dy03du)
        db3du[:,3*idx3:3*(idx3+1)] -= np.eye(3)
        # d2b0du
        d2b3d2u = np.array(d2y03d2u)
        # dc0du
        dc0du                       = np.array(dy1du)       # <-- COPY of dy1du (not pointer)
        dc0du[:,3*idx0:3*(idx0+1)] -= np.eye(3)
        # d2c0d2u
        d2c0d2u = np.array(d2y1d2u)
        # dc1du
        dc1du  = np.array(dy2du)
        dc1du -= dy1du
        # d2c1d2u
        d2c1d2u  = np.array(d2y2d2u)
        d2c1d2u -= d2y1d2u
        # dc2du
        dc2du = -np.array(dy2du)
        dc2du[:,3*idx3:3*(idx3+1)] += np.eye(3)
        # d2c2d2u
        d2c2d2u = -np.array(d2y2d2u)


        ###################
        # Contributions d2yp1d2u  <----  y1,a0,a3,b0,b3,c0,c1,c2
        ###################
        # y1
        d2yp1d2u += np.tensordot(dyp1dy1,d2y1d2u,axes=[1,0])
        # a0
        d2yp1d2u += np.tensordot(( np.tensordot(d2yp1d2a0  ,da0du,axes=[2,0]) 
                                  +np.tensordot(d2yp1da0da3,da3du,axes=[2,0]) 
                                  +np.tensordot(d2yp1da0db0,db0du,axes=[2,0])
                                  +np.tensordot(d2yp1da0db3,db3du,axes=[2,0]) ) , da0du,axes=[1,0]) + np.tensordot(dyp1da0,d2a0d2u,axes=[1,0])
        # a3
        d2yp1d2u += np.tensordot(( np.tensordot(d2yp1da3da0,da0du,axes=[2,0]) 
                                  +np.tensordot(d2yp1da3db0,db0du,axes=[2,0])
                                  +np.tensordot(d2yp1da3db3,db3du,axes=[2,0]) ) , da3du,axes=[1,0]) + np.tensordot(dyp1da3,d2a3d2u,axes=[1,0])
        # b0
        d2yp1d2u += np.tensordot(( np.tensordot(d2yp1db0da0,da0du,axes=[2,0]) 
                                  +np.tensordot(d2yp1db0da3,da3du,axes=[2,0]) 
                                  +np.tensordot(d2yp1db0dc0,dc0du,axes=[2,0])
                                  +np.tensordot(d2yp1db0dc1,dc1du,axes=[2,0]) ) , db0du,axes=[1,0]) + np.tensordot(dyp1db0,d2b0d2u,axes=[1,0])
        # b3
        d2yp1d2u += np.tensordot(( np.tensordot(d2yp1db3da0,da0du,axes=[2,0]) 
                                  +np.tensordot(d2yp1db3da3,da3du,axes=[2,0]) 
                                  +np.tensordot(d2yp1db3dc0,dc0du,axes=[2,0])
                                  +np.tensordot(d2yp1db3dc2,dc2du,axes=[2,0]) ) , db3du,axes=[1,0]) + np.tensordot(dyp1db3,d2b3d2u,axes=[1,0])
        # c0
        d2yp1d2u += np.tensordot(( np.tensordot(d2yp1dc0db0,db0du,axes=[2,0]) 
                                  +np.tensordot(d2yp1dc0db3,db3du,axes=[2,0]) 
                                  +np.tensordot(d2yp1d2c0  ,dc0du,axes=[2,0]) 
                                  +np.tensordot(d2yp1dc0dc1,dc1du,axes=[2,0])
                                  +np.tensordot(d2yp1dc0dc2,dc2du,axes=[2,0]) ) , dc0du,axes=[1,0]) + np.tensordot(dyp1dc0,d2c0d2u,axes=[1,0])
        # c1
        d2yp1d2u += np.tensordot(( np.tensordot(d2yp1dc1db0,db0du,axes=[2,0]) 
                                  +np.tensordot(d2yp1dc1dc0,dc0du,axes=[2,0]) ) , dc1du,axes=[1,0]) + np.tensordot(dyp1dc1,d2c1d2u,axes=[1,0])
        # c2
        d2yp1d2u += np.tensordot(( np.tensordot(d2yp1dc2db3,db3du,axes=[2,0]) 
                                  +np.tensordot(d2yp1dc2dc0,dc0du,axes=[2,0])
                                  +np.tensordot(d2yp1d2c2  ,dc2du,axes=[2,0]) ) , dc2du,axes=[1,0]) + np.tensordot(dyp1dc2,d2c2d2u,axes=[1,0])

        return d2yp1d2u

    def dxidu(self,idx, order=1):
        Ne = len(self.squad)
        dxidu = np.zeros((3,3*(Ne+1)))
        if idx < 5:
            dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
            d2xid2u = 0
        elif idx < 13:
            idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
            inverse = [True,False][idx%2]   # CW / CCW
            dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
            if order==2:
                d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
                d2xid2u[:,3:,3:] = self.d2y1d2u(idx0,inverse=inverse)
        else:
            idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
            inverse = [False, True, True, False, False, True, True, False][idx-13]
            dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)
            if order==2:
                d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
                d2xid2u[:,3:,3:] = self.d2yp1d2u(idx0,inverse=inverse)

        return dxidu if order==1 else (dxidu,d2xid2u)



    # Contact Force & Stiffness
    def mC(self, xs, kn, cubicT=None, OPA=1e-8):
        """Frictionless contact potential"""

        t = self.findProjection(xs)
        xc = self.Grg(t)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal

        if  cubicT is None:
            return 1/2*kn*gn**2
        # TODO: define following cubic cases
        elif gn<cubicT:
            print("Not yet defined")
            set_trace()
        else:
            print("Not yet defined")
            set_trace()


    def fintC_fless_rigidMaster(self, xs, kn, seeding=10, cubicT=None, OPA=1e-8, xs_check=None):       #Chain rule by hand
        """Frictionless contact force"""
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc = self.Grg(t)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal

        # print("id_patch: ",self.iquad,"gn: ",gn)
        # print("         t: ",t)

        opa = 0
        if not (0-opa<=t[0]<=1+opa and 0-opa<=t[1]<=1+opa):
            return np.zeros(  3*(Ne+1) ), gn, t


        dgndu    = np.zeros(  3*(Ne+1) )
        dgndu[:3] = normal         # only the terms related to the slave node. Also dgndxc@dxcdxs = dgndn@dndxs = 0

        if  cubicT is None:
            fintCN = kn*gn*dgndu
            # print("norm_force: ",(fintCN[:3]))

            return fintCN, gn, t
        elif gn<cubicT:
            return kn/2*(2*gn*dgndu - cubicT*dgndu), gn, t
        else:
            return kn/(2*cubicT)*gn**2*dgndu, gn, t

    def fintC(self, xs, kn, seeding=10, cubicT=None, OPA=1e-8, xs_check=None):       #Chain rule by hand
        """Frictionless contact force"""
        Ne = len(self.squad)

        t = self.findProjection(xs)
        # xc, dxcdt, d2xcd2t= self.Grg(t, deriv = 2)
        xc = self.Grg(t)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal

        # print("id_patch: ",self.iquad,"gn: ",gn)
        # print("         t: ",t)

        opa = 0
        if not (0-opa<=t[0]<=1+opa and 0-opa<=t[1]<=1+opa):
            return np.zeros(  3*(Ne+1) ), gn, t

        # SAVE DATA for Leonint (if within projection   0<t<1)
        with open("ContactData", 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([xs[0],xs[1],xs[2],gn,normal[0],normal[1],normal[2]])

        # dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxc = -normal # this is more consistent with the definition of gn and allows attraction even when slave is outside
        dgndxs = -dgndxc
        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        # dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        # dfdxs = -2*dxcdt.T
        # dtdxs = np.linalg.solve(-dfdt,dfdxs)

        # dxcdxs = dxcdt @ dtdxs

        # # partial derivatives
        # DxcDxi      = self.dxcdxi(t)                # shape (20,)
        # D2xcDxiDt   = self.d2xcdxidt(t)             # shape (20,2)

        #########################
        ## fintC & KC assembly ##
        #########################
        dgndu    = np.zeros(  3*(Ne+1) )

        # fintC Slave :
        # dgndu[:3] += dgndxs + dgndxc @ dxcdxs
        dgndu[:3] += dgndxs

        # # fintC CtrlPts :
        # for idx in range(1,21):
        #     dxcdxi_part = DxcDxi[idx-1]*np.eye(3)
        #     d2xcdxidt_part = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)     # shape = (3,3,2)
        #     dfdxi = 2*(dxcdxi_part.T@dxcdt).T -2*((xs-xc)@d2xcdxidt_part).T
        #     dtdxi = np.linalg.solve(-dfdt,dfdxi)
        #     dxcdxi = dxcdxi_part + dxcdt@dtdxi

        #     dxidu = np.zeros((3,3*(Ne+1)))
        #     if idx < 5:
        #         dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
        #     elif idx < 13:
        #         idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
        #         inverse = [True,False][idx%2]   # CW / CCW
        #         dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
        #     else:
        #         idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
        #         inverse = [False, True, True, False, False, True, True, False][idx-13]
        #         dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)

        #     dgndu += dgndxc @ dxcdxi @ dxidu

        #     # set_trace()

        if  cubicT is None:
            fintCN = kn*gn*dgndu
            # print("norm_force: ",(fintCN[:3]))

            return fintCN, gn, t
        elif gn<cubicT:
            return kn/2*(2*gn*dgndu - cubicT*dgndu), gn, t
        else:
            return kn/(2*cubicT)*gn**2*dgndu, gn, t

    def fintC1(self, xs, kn, seeding=10, cubicT=None, OPA=1e-8, t0=None):       #Chain rule by hand
        """gt is distance between xc and xc0"""
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc, dxcdt, d2xcd2t= self.Grg(t, deriv = 2)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal

        print("id_patch: ",self.iquad,"gn: ",gn)
        print("         t: ",t)

        dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxs = -dgndxc
        f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)

        dxcdxs = dxcdt @ dtdxs

        # partial derivatives
        DxcDxi  = self.dxcdxi(t )            # shape (20,)
        D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)

        if t0 is not None:
            kt = 0.001*kn
            dgtdu    = np.zeros(  3*(Ne+1) )
            xc0= self.Grg(t0)
            gt = norm(xc-xc0)
            dgtdxc  = (xc-xc0)/norm(xc-xc0)
            dgtdxc0 = -dgtdxc
            Dxc0Dxi = self.dxcdxi(t0)


        #########################
        ## fintC & KC assembly ##
        #########################
        dgndu    = np.zeros(  3*(Ne+1) )

        # fintC Slave :
        dgndu[:3] += dgndxs + dgndxc @ dxcdxs
        if t0 is not None:
            dgtdu[:3] += dgtdxc @ dxcdxs

        # fintC CtrlPts :
        for idx in range(1,21):
            dxcdxi_part = DxcDxi[idx-1]*np.eye(3)
            d2xcdxidt_part = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)     # shape = (3,3,2)
            dfdxi = 2*(dxcdxi_part.T@dxcdt).T -2*((xs-xc)@d2xcdxidt_part).T
            dtdxi = np.linalg.solve(-dfdt,dfdxi)
            dxcdxi_tot = dxcdxi_part + dxcdt@dtdxi

            dxidu = np.zeros((3,3*(Ne+1)))
            if idx < 5:
                dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
            elif idx < 13:
                idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
                inverse = [True,False][idx%2]   # CW / CCW
                dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
            else:
                idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
                inverse = [False, True, True, False, False, True, True, False][idx-13]
                dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)

            dgndu += dgndxc @ dxcdxi_tot @ dxidu

            if t0 is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dgtdu += (dgtdxc0@dxc0dxi + dgtdxc@dxcdxi_tot) @ dxidu


        if  cubicT ==None:
            if t0 is not None:
                return kn*gn*dgndu + kt*gt*dgtdu, gn
            else:
                return kn*gn*dgndu, gn
        elif gn<cubicT:
            return kn/2*(2*gn*dgndu - cubicT*dgndu), gn
        else:
            return kn/(2*cubicT)*gn**2*dgndu, gn

    def fintC2(self, xs, kn, kt, f0, seeding=10, cubicT=None, OPA=1e-8, t0=None, gn0=None, Sticking=True):       #Chain rule by hand
        """gt is distance between xs0 and xs"""
        
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc, dxcdt, d2xcd2t= self.Grg(t, deriv = 2)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal
        dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxs = -dgndxc
        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)

        dxcdxs = dxcdt @ dtdxs

        # partial derivatives
        DxcDxi  = self.dxcdxi(t )            # shape (20,)
        D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)

        if t0 is not None:
            # kt = 0.001*kn
            kt = 1000
            dgtdu    = np.zeros(  3*(Ne+1) )
            xc0= self.Grg(t0)
            n0 = self.D3Grg(t0)
            Dn0Dxi = self.dndxi(t0)
            xs0 = xc0 - gn0*n0
            dxs0dxc0 = np.eye(3)
            dxs0dn0 = -gn0*np.eye(3)
            Dxc0Dxi = self.dxcdxi(t0)
            gt = norm(xs-xs0)
            dgtdxs  = (xs-xs0)/norm(xs-xs0)
            dgtdxs0 = -dgtdxs

        # set_trace()

        print("id_patch: ",self.iquad,"gn: ",gn, "  gt: ",gt if t0 is not None else "-")
        print("         t: ",t)
            # set_trace()

        #########################
        ## fintC & KC assembly ##
        #########################
        dgndu    = np.zeros(  3*(Ne+1) )

        # fintC Slave :
        dgndu[:3] += dgndxs + dgndxc @ dxcdxs
        if t0 is not None:
            dgtdu[:3] += dgtdxs

        # fintC CtrlPts :
        for idx in range(1,21):
            dxcdxi_part = DxcDxi[idx-1]*np.eye(3)
            d2xcdxidt_part = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)     # shape = (3,3,2)
            dfdxi = 2*(dxcdxi_part.T@dxcdt).T -2*((xs-xc)@d2xcdxidt_part).T
            dtdxi = np.linalg.solve(-dfdt,dfdxi)
            dxcdxi_tot = dxcdxi_part + dxcdt@dtdxi

            dxidu = np.zeros((3,3*(Ne+1)))
            if idx < 5:
                dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
            elif idx < 13:
                idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
                inverse = [True,False][idx%2]   # CW / CCW
                dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
            else:
                idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
                inverse = [False, True, True, False, False, True, True, False][idx-13]
                dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)

            dgndu += dgndxc @ dxcdxi_tot @ dxidu

            if t0 is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi  = Dn0Dxi[idx-1]

                dgtdu += dgtdxs0@(dxs0dxc0@dxc0dxi+dxs0dn0@dn0dxi) @ dxidu


        if cubicT is None:
            if t0 is not None:
                # set_trace()

                print("")
                print("kt*gt = ",kt*gt, "\twhile mu*N = ",0.1*gn*kn)
                print("")

                print("norm_force: ",(kn*gn*dgndu[:3]))
                if Sticking:
                    print("tang_force: ",(kt*gt*dgtdu[:3]))
                    return kn*gn*dgndu +kt*gt*dgtdu, gn, gt, t
                else:
                    # set_trace()
                    print("tang_force: ",(f0*dgtdu[:3]))
                    return kn*gn*dgndu +f0*dgtdu, gn, gt, t
            else:
                return kn*gn*dgndu, gn, 0.0, t
        elif gn<cubicT:
            return kn/2*(2*gn*dgndu - cubicT*dgndu), gn
        else:
            return kn/(2*cubicT)*gn**2*dgndu, gn

    def fintC2_tang(self, xs, kt, f0, t0=None, gn0=None, Sticking=True):       #Chain rule by hand
        """gt is distance between xs0 and xs"""
        
        Ne = len(self.squad)

        dgtdu    = np.zeros(  3*(Ne+1) )
        xc0= self.Grg(t0)
        n0 = self.D3Grg(t0)
        Dn0Dxi = self.dndxi(t0)
        xs0 = xc0 - gn0*n0
        dxs0dxc0 = np.eye(3)
        dxs0dn0 = -gn0*np.eye(3)
        Dxc0Dxi = self.dxcdxi(t0)
        gt = norm(xs-xs0)
        dgtdxs  = (xs-xs0)/norm(xs-xs0)
        dgtdxs0 = -dgtdxs


        ####################
        ## fintC assembly ##
        ####################

        # fintC Slave :
        dgtdu[:3] += dgtdxs

        # fintC CtrlPts :
        for idx in range(1,21):
            dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
            dn0dxi  = Dn0Dxi[idx-1]
            dxidu = np.zeros((3,3*(Ne+1)))

            if idx < 5:
                dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
            elif idx < 13:
                idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
                inverse = [True,False][idx%2]   # CW / CCW
                dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
            else:
                idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
                inverse = [False, True, True, False, False, True, True, False][idx-13]
                dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)

            dgtdu += dgtdxs0@(dxs0dxc0@dxc0dxi+dxs0dn0@dn0dxi) @ dxidu

        print("")
        print("kt*gt = ",kt*gt)
        print("")

        if Sticking:
            return kt*gt*dgtdu, gt
        else:
            return f0*dgtdu, gt

    def fintC_tot(self,xs,kn,kt,mu,hook=None,Stick=True, cubicT=None,disp=False, output=None, niter=3, new=False):
        """gt is distance between xs0 and xs"""
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc, dxcdt, d2xcd2t= self.Grg(t, deriv = 2)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal
        dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxs = -dgndxc
        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)

        dxcdxs = dxcdt @ dtdxs

        # partial derivatives
        DxcDxi  = self.dxcdxi(t )            # shape (20,)
        D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)

        if hook is not None:
            ihp = int(hook[0])
            hpatch = self.surf.patches[ihp] if ihp!=self.iquad else self
            hNe = len(hpatch.squad)
            dgtdu = np.zeros(  3*(hNe+1) )
            t0 = hook[1:3]
            gn0 = hook[3]
            xc0 = hpatch.Grg(t0)
            n0  = hpatch.D3Grg(t0)
            xs0 = xc0 - gn0*n0
            gt = norm(xs-xs0)
            dxs0dxc0= np.eye(3)
            dxs0dn0 = -gn0*np.eye(3)
            dgtdxs  = (xs-xs0)/norm(xs-xs0)
            dgtdxs0 = -dgtdxs
            Dxc0Dxi = hpatch.dxcdxi(t0)
            Dn0Dxi  = hpatch.dndxi(t0)
            dgtdu[:3] += dgtdxs

        printif(disp,"id_patch: ",self.iquad,"gn: ",gn, "  gt: ",gt if hook is not None else "-")
        printif(disp,"         t: ",t)

        #########################
        ## fintC & KC assembly ##
        #########################
        dgndu    = np.zeros(  3*(Ne+1) )

        # fintC Slave :
        dgndu[:3] += dgndxs + dgndxc @ dxcdxs

        # fintC CtrlPts :
        for idx in range(1,21):
            dxcdxi_part = DxcDxi[idx-1]*np.eye(3)
            d2xcdxidt_part = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)     # shape = (3,3,2)
            dfdxi = 2*(dxcdxi_part.T@dxcdt).T -2*((xs-xc)@d2xcdxidt_part).T
            dtdxi = np.linalg.solve(-dfdt,dfdxi)
            dxcdxi_tot = dxcdxi_part + dxcdt@dtdxi


            dxidu = self.dxidu(idx)
            dgndu += dgndxc @ dxcdxi_tot @ dxidu

            if hook is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi  = Dn0Dxi[idx-1]
                dxiduh = hpatch.dxidu(idx) if ihp != self.iquad else dxidu
                dgtdu += dgtdxs0@(dxs0dxc0@dxc0dxi+dxs0dn0@dn0dxi) @ dxiduh


        # if not Stick:
        #     set_trace()
        if type(output) is str:
            return locals()[output]


        if cubicT is None:
            fintCN = kn*gn*dgndu
            fn = fintCN[:3]
            printif(disp,"norm_force: ",(fn))
        elif gn<cubicT:
            fintCN = kn/2*(2*gn*dgndu - cubicT*dgndu)
        else:
            fintCN = kn/(2*cubicT)*gn**2*dgndu, gn

        if hook is None:
            return fintCN, gn, t
        else:
            printif(disp,"kt*gt = ",kt*gt, "\twhile mu*N = ",-mu*gn*kn)
            # f0 = min(kt*gt,-mu*kn*gn)
            # fintCT= kt*gt*dgtdu if Stick else -mu*kn*gn*dgtdu
            # fintCT= kt*gt*dgtdu if kt*gt<-mu*kn*gn else -mu*kn*gn*dgtdu
            cond = niter<1 and new

            printif(cond, "ELASTIC IMPOSED")
            fintCT= kt*gt*dgtdu if (kt*gt<-mu*kn*gn or cond) else -mu*kn*gn*dgtdu
            if (kt*gt<-mu*kn*gn or cond):
                print("elastic fintC")
            else:
                print("plastic fintC")

            ft = fintCT[:3]
            printif(disp,"tang_force: ",ft)
            printif(disp,"angle fn-ft: ",np.arccos((fn@ft)/(norm(fn)*norm(ft)))*180/np.pi, "degrees" )
            return fintCN, fintCT, gn,gt,t


    def fintC_tot2(self,xs,kn,kt,mu,hook=None,Stick=True, cubicT=None,disp=False, output=None, niter=3, new=False, tracing=False):
        """gt is orthogonal distance between xs and line passing through xc0 with direction n0"""
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc, dxcdt, d2xcd2t= self.Grg(t, deriv = 2)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal
        dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxs = -dgndxc
        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)

        dxcdxs = dxcdt @ dtdxs

        # partial derivatives
        DxcDxi  = self.dxcdxi(t )            # shape (20,)
        D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)

        if hook is not None:
            ihp = int(hook[0])
            hpatch = self.surf.patches[ihp] if ihp!=self.iquad else self
            hNe = len(hpatch.squad)
            dgtdu = np.zeros(  3*(hNe+1) )
            t0 = hook[1:3]
            if np.allclose(t-t0,0.0): return self.fintC_tot2(xs,kn,kt,mu,hook=None,Stick=True, cubicT=cubicT,disp=disp, output=None, niter=3, new=new, tracing=False)[0], np.zeros(3*(Ne+1)), gn,0.0,t
            gn0 = hook[3]
            xc0 = hpatch.Grg(t0)
            n0  = hpatch.D3Grg(t0)
            xsp = xc0 + np.dot(xs-xc0,n0)*n0
            gt = norm(xs-xsp)
            # print("CHECK HERE. HOW CAN gt>gn???")
            # set_trace()
            dgtdxs  = (xs-xsp)/norm(xs-xsp)
            dgtdxsp = -dgtdxs
            dxspdxs = np.outer(n0,n0)
            dxspdxc0= np.eye(3)-dxspdxs
            dxspdn0 = np.outer(n0,xs-xc0)+np.dot(xs-xc0,n0)*np.eye(3)   #TODO: check order of terms for outer product
            if tracing: set_trace()
            # dxspdn0 = np.outer(xs-xc0,n0)+np.dot(xs-xc0,n0)*np.eye(3)   #TODO: check order of terms for outer product
            # set_trace()
            Dxc0Dxi = hpatch.dxcdxi(t0)
            Dn0Dxi  = hpatch.dndxi(t0)
            dgtdu[:3] += dgtdxs + dgtdxsp@dxspdxs

        printif(disp,"id_patch: ",self.iquad,"gn: ",gn, "  gt: ",gt if hook is not None else "-")
        printif(disp,"         t: ",t)

        #########################
        ## fintC & KC assembly ##
        #########################
        dgndu    = np.zeros(  3*(Ne+1) )

        # fintC Slave :
        dgndu[:3] += dgndxs + dgndxc @ dxcdxs

        # fintC CtrlPts :
        for idx in range(1,21):
            dxcdxi_part = DxcDxi[idx-1]*np.eye(3)
            d2xcdxidt_part = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)     # shape = (3,3,2)
            dfdxi = 2*(dxcdxi_part.T@dxcdt).T -2*((xs-xc)@d2xcdxidt_part).T
            dtdxi = np.linalg.solve(-dfdt,dfdxi)
            dxcdxi_tot = dxcdxi_part + dxcdt@dtdxi


            dxidu = self.dxidu(idx)
            dgndu += dgndxc @ dxcdxi_tot @ dxidu

            if hook is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi  = Dn0Dxi[idx-1]
                dxiduh = hpatch.dxidu(idx) if ihp != self.iquad else dxidu
                dgtdu += dgtdxsp@(dxspdxc0@dxc0dxi+dxspdn0@dn0dxi) @ dxiduh


        if type(output) is str:
            return locals()[output]


        if cubicT is None:
            fintCN = kn*gn*dgndu
            fn = fintCN[:3]
            printif(disp,"norm_force: ",(fn))
        elif gn<cubicT:
            fintCN = kn/2*(2*gn*dgndu - cubicT*dgndu)
        else:
            fintCN = kn/(2*cubicT)*gn**2*dgndu, gn

        if hook is None:
            return fintCN
        else:
            printif(disp,"kt*gt = ",kt*gt, "\twhile mu*N = ",-mu*gn*kn)
            # f0 = min(kt*gt,-mu*kn*gn)
            # fintCT= kt*gt*dgtdu if Stick else -mu*kn*gn*dgtdu
            # fintCT= kt*gt*dgtdu if kt*gt<-mu*kn*gn else -mu*kn*gn*dgtdu
            # new=True
            cond = niter<1 and new
            cond=False

            printif(cond, "ELASTIC IMPOSED")
            fintCT= kt*gt*dgtdu if (kt*gt<-mu*kn*gn or cond) else -mu*kn*gn*dgtdu
            Stick =(kt*gt<-mu*kn*gn or cond)
            # Stick=True

            ft = fintCT[:3]
            printif(disp,"tang_force: ",ft)
            printif(disp,"angle fn-ft: ",np.arccos((fn@ft)/(norm(fn)*norm(ft)))*180/np.pi, "degrees" )
            return fintCN, fintCT, gt, Stick


    def fintC_tot3(self,xs,kn,kt,mu,hook=None,Stick=True, cubicT=None,disp=False, output=None, niter=3, new=False, tracing=False):
        """gt is orthogonal distance between xs and line passing through xc0 with direction n0"""
        Ne = len(self.squad)

        t = self.findProjection(xs)
        if hook is None:
            xc, dxcdt, d2xcd2t= self.Grg(t, deriv = 2)
            normal = self.D3Grg(t)
            gn = (xs-xc)@normal
            dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
            dgndxs = -dgndxc
            # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
            dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
            dfdxs = -2*dxcdt.T
            dtdxs = np.linalg.solve(-dfdt,dfdxs)

            dxcdxs = dxcdt @ dtdxs

            # partial derivatives
            DxcDxi  = self.dxcdxi(t )            # shape (20,)
            D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)

        else:
            ihp = int(hook[0])
            hpatch = self.surf.patches[ihp] if ihp!=self.iquad else self
            hNe = len(hpatch.squad)
            dgtdu = np.zeros(  3*(hNe+1) )
            t0 = hook[1:3]
            if np.allclose(t-t0,0.0): return self.fintC_tot3(xs,kn,kt,mu,hook=None,Stick=True, cubicT=cubicT,disp=disp, output=None, niter=3, new=new, tracing=False)[0], np.zeros(3*(Ne+1)), gn,0.0,t
            gn0 = hook[3]
            xc0 = hpatch.Grg(t0)
            n0  = hpatch.D3Grg(t0)
            gn = (xs-xc0)@n0
            dgndxs = n0
            dgndxc0 = -n0
            dgndn0 = xs-xc0
            xsp = xc0 + gn*n0
            gt = norm(xs-xsp)
            # print("CHECK HERE. HOW CAN gt>gn???")
            # set_trace()
            dgtdxs  = (xs-xsp)/norm(xs-xsp)
            dgtdxsp = -dgtdxs
            dxspdxs = np.outer(n0,n0)
            dxspdxc0= np.eye(3)-dxspdxs
            dxspdn0 = np.outer(n0,xs-xc0)+np.dot(xs-xc0,n0)*np.eye(3)   #TODO: check order of terms for outer product
            if tracing: set_trace()
            # dxspdn0 = np.outer(xs-xc0,n0)+np.dot(xs-xc0,n0)*np.eye(3)   #TODO: check order of terms for outer product
            # set_trace()
            Dxc0Dxi = hpatch.dxcdxi(t0)
            Dn0Dxi  = hpatch.dndxi(t0)
            dgtdu[:3] += dgtdxs + dgtdxsp@dxspdxs

        printif(disp,"id_patch: ",self.iquad,"gn: ",gn, "  gt: ",gt if hook is not None else "-")
        printif(disp,"         t: ",t)

        #########################
        ## fintC & KC assembly ##
        #########################
        dgndu    = np.zeros(  3*(Ne+1) if hook is None else 3*(Ne+1) )

        # fintC Slave :
        dgndu[:3] += dgndxs

        # fintC CtrlPts :
        for idx in range(1,21):
            
            dxidu = self.dxidu(idx)
            if hook is None:
                dxcdxi_part = DxcDxi[idx-1]*np.eye(3)
                d2xcdxidt_part = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)     # shape = (3,3,2)
                dfdxi = 2*(dxcdxi_part.T@dxcdt).T -2*((xs-xc)@d2xcdxidt_part).T
                dtdxi = np.linalg.solve(-dfdt,dfdxi)
                dxcdxi_tot = dxcdxi_part + dxcdt@dtdxi
                dgndu += dgndxc @ dxcdxi_tot @ dxidu



            else:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi  = Dn0Dxi[idx-1]
                dxiduh = hpatch.dxidu(idx) if ihp != self.iquad else dxidu
                dgndu += (dgndxc0@dxc0dxi + dgndn0@dn0dxi )@ dxiduh
                dgtdu += dgtdxsp@(dxspdxc0@dxc0dxi+dxspdn0@dn0dxi) @ dxiduh


        if type(output) is str:
            return locals()[output]


        if cubicT is None:
            fintCN = kn*gn*dgndu
            fn = fintCN[:3]
            printif(disp,"norm_force: ",(fn))
        elif gn<cubicT:
            fintCN = kn/2*(2*gn*dgndu - cubicT*dgndu)
        else:
            fintCN = kn/(2*cubicT)*gn**2*dgndu, gn

        if hook is None:
            return fintCN, gn, t
        else:
            printif(disp,"kt*gt = ",kt*gt, "\twhile mu*N = ",-mu*gn*kn)
            # f0 = min(kt*gt,-mu*kn*gn)
            # fintCT= kt*gt*dgtdu if Stick else -mu*kn*gn*dgtdu
            # fintCT= kt*gt*dgtdu if kt*gt<-mu*kn*gn else -mu*kn*gn*dgtdu
            cond = niter<1 and new

            printif(cond, "ELASTIC IMPOSED")
            fintCT= kt*gt*dgtdu if (kt*gt<-mu*kn*gn or cond) else -mu*kn*gn*dgtdu
            if (kt*gt<-mu*kn*gn or cond):
                print("elastic fintC")
            else:
                print("plastic fintC")

            ft = fintCT[:3]
            printif(disp,"tang_force: ",ft)
            printif(disp,"angle fn-ft: ",np.arccos((fn@ft)/(norm(fn)*norm(ft)))*180/np.pi, "degrees" )
            return fintCN, fintCT, gn,gt,t




    def fintC3(self, xs, kn, seeding=10, cubicT=None, OPA=1e-8, t0=None):       #Chain rule by hand
        """"Projected distance from xc0 to xs tangent to xc"""
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc, dxcdt, d2xcd2t= self.Grg(t, deriv = 2)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal

        print("id_patch: ",self.iquad,"gn: ",gn)
        print("         t: ",t)

        dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxs = -dgndxc
        f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)

        dxcdxs = dxcdt @ dtdxs

        # partial derivatives
        DxcDxi      = self.dxcdxi(t)                # shape (20,)
        D2xcDxiDt   = self.d2xcdxidt(t)             # shape (20,2)

        if t0 is not None:
            kt=0.001*kn
            dgtdu    = np.zeros(  3*(Ne+1) )
            xc0 = self.Grg(t0)
            tau1, tau2 = dxcdt.T
            dtau1dt, dtau2dt = d2xcd2t.swapaxes(0,1)
            N = np.cross(tau1,tau2)
            ns = N/norm(N)          # I dont use self.D3Grg cause I need tau_i and N anyway
            xcp = xc0-(ns@(xc0-xs))*ns
            gt = norm(xcp-xs)

            dgtdxcp = (xcp-xs)/gt
            dgtdxs  =-dgtdxcp
            dxcpdxs = np.outer(ns,ns)
            dxcpdxc0= np.eye(3) - dxcpdxs
            dxcpdns =-np.outer(ns,xc0-xs) -(ns@(xc0-xs))*np.eye(3)
            dnsdN   = np.eye(3)/norm(N) - np.outer(N,N)/(norm(N)**3)
            dNdtau1 =-skew(tau2)        # signs verified with Autograd
            dNdtau2 = skew(tau1)        # signs verified with Autograd
            
            Dtau1Dxi, Dtau2Dxi  = self.d2xcdxidt(t0).T
            dxcpdtau1 = dxcpdns@dnsdN@dNdtau1
            dxcpdtau2 = dxcpdns@dnsdN@dNdtau2
            Dxc0Dxi = self.dxcdxi(t0)




        #########################
        ## fintC & KC assembly ##
        #########################
        dgndu    = np.zeros(  3*(Ne+1) )

        # fintC Slave :
        dgndu[:3] += dgndxs + dgndxc @ dxcdxs
        if t0 is not None:
            dgtdu[:3] += dgtdxs + dgtdxcp@(dxcpdxs + (dxcpdtau1@dtau1dt+dxcpdtau2@dtau2dt)@dtdxs)

        # fintC CtrlPts :
        for idx in range(1,21):
            dxcdxi_part = DxcDxi[idx-1]*np.eye(3)
            d2xcdxidt_part = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)     # shape = (3,3,2)
            dfdxi = 2*(dxcdxi_part.T@dxcdt).T -2*((xs-xc)@d2xcdxidt_part).T
            dtdxi = np.linalg.solve(-dfdt,dfdxi)
            dxcdxi = dxcdxi_part + dxcdt@dtdxi

            dxidu = np.zeros((3,3*(Ne+1)))
            if idx < 5:
                dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
            elif idx < 13:
                idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
                inverse = [True,False][idx%2]   # CW / CCW
                dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
            else:
                idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
                inverse = [False, True, True, False, False, True, True, False][idx-13]
                dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)

            dgndu += dgndxc @ dxcdxi @ dxidu

            if t0 is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dtau1dxi_tot = Dtau1Dxi[idx-1]*np.eye(3) + dtau1dt@dtdxi
                dtau2dxi_tot = Dtau2Dxi[idx-1]*np.eye(3) + dtau2dt@dtdxi
                dgtdu += dgtdxcp@(dxcpdxc0@dxc0dxi + dxcpdtau1@dtau1dxi_tot + dxcpdtau2@dtau2dxi_tot)@dxidu


        if  cubicT is None:
            if t0 is not None:
                return kn*gn*dgndu + kt*gt*dgtdu, gn
            else:
                return kn*gn*dgndu, gn
        elif gn<cubicT:
            return kn/2*(2*gn*dgndu - cubicT*dgndu), gn
        else:
            return kn/(2*cubicT)*gn**2*dgndu, gn

    def KC_fless_rigidMaster(self, xs, kn, seeding=10, cubicT=None, OPA=1e-8):       #Chain rule by hand
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc, dxcdt, d2xcd2t= self.Grg(t, deriv = 2)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal

        opa = 1e-3
        if not (0-opa<=t[0]<=1+opa and 0-opa<=t[1]<=1+opa):
            return np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        dgdxs =  normal

        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)

        # Vars for K---------------------------------------------------
        dndt = self.dndt(t)
        dndxs = dndt@dtdxs

        #########################
        ## fintC & KC assembly ##
        #########################
        dgdu    = np.zeros(  3*(Ne+1) )
        d2gd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        # fintC Slave :
        dgdu[:3] = dgdxs

        # K     Slave-Slave :
        d2gd2u[:3,:3] =  dndxs

                
        if cubicT==None:
            return kn*(np.outer(dgdu,dgdu)+gn*d2gd2u)
        elif gn<cubicT:
            return kn*(np.outer(dgdu,dgdu)+(gn-cubicT/2)*d2gd2u)
        else:
            return kn*((gn/cubicT)*np.outer(dgdu,dgdu)+(gn**2/(2*cubicT))*d2gd2u)

    def KC_fless_rigidMaster_redundant(self, xs, kn, seeding=10, cubicT=None, OPA=1e-8):       #Chain rule by hand
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc, dxcdt, d2xcd2t, d3xcd3t= self.Grg(t, deriv = 3)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal

        opa = 1e-3
        if not (0-opa<=t[0]<=1+opa and 0-opa<=t[1]<=1+opa):
            return np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        # dgdxc = -normal            # norm(xs-xc) != gn   (opposite sign)
        dgdxs =  normal
        dgdn  =  xs - xc

        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)
        dxcdxs = dxcdt @ dtdxs


        # Vars for K---------------------------------------------------

        # d/du(dxcdxs)
        d2fdt2   = 2*(np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]) + np.tensordot(d2xcd2t,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcd3t,axes=[[0],[0]]) )
        invdfdt = np.linalg.inv(dfdt)

        d2fdxsdt = -2*d2xcd2t.swapaxes(0,1)
        d2td2xs = np.tensordot(-invdfdt,(np.tensordot(np.tensordot(dtdxs,d2fdt2,axes=[0,1]),dtdxs,axes=[2,0]).swapaxes(0,1)  + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]) + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]).swapaxes(1,2)),axes = [[1],[0]])
        d2xcd2xs = np.tensordot( np.tensordot(dtdxs,d2xcd2t,axes=[0,2]),dtdxs,axes=[2,0]).swapaxes(0,1) + np.tensordot(dxcdt,d2td2xs,axes=[1,0])

        dndt, d2nd2t = self.dndt(t,degree=2)
        dndxs = dndt@dtdxs
        d2nd2xs = np.tensordot(np.tensordot(d2nd2t,dtdxs,axes=[1,0]),dtdxs,axes=[1,0]).swapaxes(1,2) + np.tensordot(dndt,d2td2xs,axes=[1,0])


        #########################
        ## fintC & KC assembly ##
        #########################
        dgdu    = np.zeros(  3*(Ne+1) )
        d2gd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        # fintC Slave :
        dgdu[:3] = dgdxs

        # K     Slave-Slave :
        set_trace()
        d2gd2u[:3,:3] =  dndxs - dndxs@dxcdxs - np.tensordot(normal,d2xcd2xs, axes=[0,0]) + (np.eye(3)-dxcdxs)@dndxs + np.tensordot(dgdn,d2nd2xs,axes=[0,0])

                
        if cubicT==None:
            return kn*(np.outer(dgdu,dgdu)+gn*d2gd2u)
        elif gn<cubicT:
            return kn*(np.outer(dgdu,dgdu)+(gn-cubicT/2)*d2gd2u)
        else:
            return kn*((gn/cubicT)*np.outer(dgdu,dgdu)+(gn**2/(2*cubicT))*d2gd2u)

    def KC(self, xs, kn, seeding=10, cubicT=None, OPA=1e-8):       #Chain rule by hand
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc, dxcdt, d2xcd2t, d3xcd3t= self.Grg(t, deriv = 3)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal

        opa = 1e-3
        if not (0-opa<=t[0]<=1+opa and 0-opa<=t[1]<=1+opa):
            return np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        #### EXPERIMENT #######
        ## REPLACING THIS ###
        # dgdxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        # dgdxs = -dgdxc
        ## BY THIS 333
        dgdxc = -normal            # norm(xs-xc) != gn   (opposite sign)
        dgdxs =  normal
        ########################


        f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)
        dxcdxs = dxcdt @ dtdxs


        # Vars for K---------------------------------------------------
        d2gd2xs = np.eye(3)/gn - np.outer((xs-xc),(xs-xc))/gn**3         
        d2gd2xc = d2gd2xs
        d2gdxsdxc = -d2gd2xs

        # d/du(dxcdxs)
        d2fdt2   = 2*(np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]) + np.tensordot(d2xcd2t,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcd3t,axes=[[0],[0]]) )
        invdfdt = np.linalg.inv(dfdt)

        d2fdxsdt = -2*d2xcd2t.swapaxes(0,1)
        d2td2xs = np.tensordot(-invdfdt,(np.tensordot(np.tensordot(dtdxs,d2fdt2,axes=[0,1]),dtdxs,axes=[2,0]).swapaxes(0,1)  + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]) + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]).swapaxes(1,2)),axes = [[1],[0]])
        d2xcd2xs = np.tensordot( np.tensordot(dtdxs,d2xcd2t,axes=[0,2]),dtdxs,axes=[2,0]).swapaxes(0,1) + np.tensordot(dxcdt,d2td2xs,axes=[1,0])



        # DxcDxi = self.dxcdxi(t)             # shape (20,)
        # D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)
        # D3xcDxiD2t = self.d3xcdxid2t(t)     # shape (20,2,2)
        # # DxcDxi_tot = []
        # # DtDxi = []
        # # D2fDxiDt = []
        # DtDxi      = np.empty((20,2,3)  ,dtype=float)
        # D2fDxiDt   = np.empty((20,2,3,2),dtype=float)
        # DxcDxi_tot = np.empty((20,3,3)  ,dtype=float)

        #########################
        ## fintC & KC assembly ##
        #########################
        dgdu    = np.zeros(  3*(Ne+1) )
        d2gd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        # fintC Slave :
        # dgdu[:3] += dgdxs + dgdxc @ dxcdxs
        dgdu[:3] += dgdxs

        # K     Slave-Slave :
        # d2gd2u[:3,:3] +=  d2gd2xs + 2*d2gdxsdxc@dxcdxs + d2gd2xc@dxcdxs@dxcdxs + np.tensordot(dgdxc,d2xcd2xs, axes=[0,0])
        d2gd2u[:3,:3] +=  d2gd2xs + 2*d2gdxsdxc@dxcdxs + (d2gd2xc@dxcdxs).T@dxcdxs + np.tensordot(dgdxc,d2xcd2xs, axes=[0,0])

        # DxiDu   = []        # will be used in nested loop below
        # DxiDu   = np.empty((20,3,3*(Ne+1))  ,dtype=float)        # will be used in nested loop below
        # for idx in range(1,21):
        #     dxcdxi = DxcDxi[idx-1]*np.eye(3)
        #     d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
        #     d3xcdxid2t = np.array([[ D3xcDxiD2t[idx-1,0,0]*np.eye(3) , D3xcDxiD2t[idx-1,0,1]*np.eye(3) ],
        #                            [ D3xcDxiD2t[idx-1,1,0]*np.eye(3) , D3xcDxiD2t[idx-1,1,1]*np.eye(3) ]]).swapaxes(0,2).swapaxes(1,3)

        #     dfdxi = 2*((dxcdxi.T@dxcdt) - np.tensordot(xs-xc,d2xcdxidt,axes=[0,0])).T
        #     dtdxi = -invdfdt@dfdxi          # torch-verified

        #     # DtDxi.append(dtdxi)

        #     DtDxi[idx-1] = dtdxi

        #     d2fdxidt  = 2*(np.tensordot(dxcdxi,d2xcd2t,axes=[0,0]) + np.tensordot(d2xcdxidt,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcdxidt,axes=[0,0]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcdxid2t,axes=[0,0])).swapaxes(0,1)
        #     # D2fDxiDt.append(d2fdxidt)
        #     D2fDxiDt[idx-1] = d2fdxidt
        #     d2fdxidxs = -2*d2xcdxidt.swapaxes(0,2)
        #     d2tdxidxs = np.tensordot(-invdfdt,  ((d2fdt2@dtdxi).swapaxes(1,2)+d2fdxidt)@dtdxs + (d2fdxsdt@dtdxi).swapaxes(1,2) + d2fdxidxs , axes = [1,0])
        #     d2xcdxidxs = np.tensordot(dxcdt,d2tdxidxs,axes=[1,0]) + ((d2xcd2t@dtdxi).swapaxes(1,2)+d2xcdxidt)@dtdxs

        #     dxcdxi += dxcdt@dtdxi # checking here <====
        #     # DxcDxi_tot.append(dxcdxi)
        #     DxcDxi_tot[idx-1] = dxcdxi

        #     if idx < 5:
        #         dxidu = np.zeros((3,3*(Ne+1)))
        #         dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
        #         d2xid2u = 0
        #     elif idx < 13:
        #         idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
        #         inverse = [True,False][idx%2]   # CW / CCW
        #         # dxidu = np.hstack((np.zeros((3,3)),self.dy1du(idx0,inverse=inverse)))
        #         dxidu = np.zeros((3,3*(Ne+1)))
        #         dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
        #         d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
        #         d2xid2u[:,3:,3:] = self.d2y1d2u(idx0,inverse=inverse)
        #     else:
        #         idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
        #         inverse = [False, True, True, False, False, True, True, False][idx-13]
        #         dxidu = np.zeros((3,3*(Ne+1)))
        #         dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)
        #         d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
        #         d2xid2u[:,3:,3:] = self.d2yp1d2u(idx0,inverse=inverse)

        #     # DxiDu.append(dxidu)
        #     DxiDu[idx-1] = dxidu


        #     # set_trace()


        #     # fintC CtrlPts :
        #     dgdu += dgdxc @ dxcdxi @ dxidu
            
        #     # K     Slave-CtrlPts :
        #     d2gd2u[:3,:] += ((d2gdxsdxc + (d2gd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgdxc,d2xcdxidxs,axes=[0,0]).T)@dxidu

        #     # K     CtrlPts-Slave:
        #     d2gd2u[ : , :3] += (((d2gdxsdxc+(d2gd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgdxc,d2xcdxidxs,axes=[0,0]).T) @ dxidu).T

        #     # K     CtrlPts-CtrlPts:
        #     d2gd2u += np.tensordot(dgdxc@dxcdxi,d2xid2u,axes=[0,0]) if type(d2xid2u)!=int else 0

        # # K     CtrlPts-CtrlPts (2):
        # for idx in range(1,21):
        #     dxidu = DxiDu[idx-1]
        #     dxcdxi = DxcDxi_tot[idx-1]
        #     d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
        #     dtdxi = DtDxi[idx-1]
        #     d2fdxidt = D2fDxiDt[idx-1]
        #     for jdx in range(1,21):
        #         dxjdu = DxiDu[jdx-1]
        #         dxcdxj = DxcDxi_tot[jdx-1]
        #         d2xcdxjdt = np.array([D2xcDxiDt[jdx-1,0]*np.eye(3),D2xcDxiDt[jdx-1,1]*np.eye(3)]).swapaxes(0,2)
        #         dtdxj = DtDxi[jdx-1]
        #         d2fdxjdt = D2fDxiDt[jdx-1]

        #         d2fdxidxj = 2*(np.tensordot(dxcdxi,d2xcdxjdt, axes = [0,0]).swapaxes(0,1) + np.tensordot(dxcdxj,d2xcdxidt,axes=[0,0])).swapaxes(0,2)
        #         d2tdxidxj = np.tensordot(-invdfdt , (((d2fdt2@dtdxj).swapaxes(1,2)+d2fdxjdt)@dtdxi).swapaxes(1,2) + d2fdxidxj + d2fdxidt@dtdxj , axes=[1,0])
                
        #         d2xcdxidxj = d2xcdxidt@dtdxj + ((d2xcdxjdt+(d2xcd2t@dtdxj).swapaxes(1,2))@dtdxi).swapaxes(1,2) + np.tensordot(dxcdt,d2tdxidxj,axes=[1,0])
        #         d2gdxidxj = (d2gd2xc@dxcdxi).T@dxcdxj + np.tensordot(dgdxc,d2xcdxidxj,axes=[0,0])

        #         d2gd2u += (d2gdxidxj@dxjdu).T@dxidu
                
        if cubicT==None:
            return kn*(np.outer(dgdu,dgdu)+gn*d2gd2u)
        elif gn<cubicT:
            return kn*(np.outer(dgdu,dgdu)+(gn-cubicT/2)*d2gd2u)
        else:
            return kn*((gn/cubicT)*np.outer(dgdu,dgdu)+(gn**2/(2*cubicT))*d2gd2u)

    def KC1(self, xs, kn, seeding = 10, cubicT = None, OPA = 1e-8, t0=None):       #Chain rule by hand
        Ne = len(self.squad)

        t = self.findProjection(xs)
        xc, dxcdt, d2xcd2t, d3xcd3t= self.Grg(t, deriv = 3)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal
        dgndu    = np.zeros(  3*(Ne+1) )
        d2gnd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxs = -dgndxc
        f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)
        dxcdxs = dxcdt @ dtdxs


        # Vars for K---------------------------------------------------
        d2gnd2xs = np.eye(3)/gn - np.outer((xs-xc),(xs-xc))/gn**3         
        d2gnd2xc = d2gnd2xs
        d2gndxsdxc = -d2gnd2xs
        if t0 is not None:
            # set_trace()
            kt = 0.001*kn
            dgtdu    = np.zeros(  3*(Ne+1) )
            d2gtd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )
            xc0= self.Grg(t0)
            gt = norm(xc-xc0)
            dgtdxc  = (xc-xc0)/norm(xc-xc0)
            dgtdxc0 = -dgtdxc
            Dxc0Dxi = self.dxcdxi(t0)
            d2gtd2xc = np.eye(3)/gt - np.outer((xc-xc0),(xc-xc0))/gt**3         
            d2gtd2xc0= d2gtd2xc
            d2gtdxc0dxc = -d2gtd2xc0



        # d/du(dxcdxs)
        d2fdt2   = 2*(np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]) + np.tensordot(d2xcd2t,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcd3t,axes=[[0],[0]]) )
        invdfdt = np.linalg.inv(dfdt)

        d2fdxsdt = -2*d2xcd2t.swapaxes(0,1)
        d2td2xs = np.tensordot(-invdfdt,(np.tensordot(np.tensordot(dtdxs,d2fdt2,axes=[0,1]),dtdxs,axes=[2,0]).swapaxes(0,1)  + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]) + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]).swapaxes(1,2)),axes = [[1],[0]])
        d2xcd2xs = np.tensordot( np.tensordot(dtdxs,d2xcd2t,axes=[0,2]),dtdxs,axes=[2,0]).swapaxes(0,1) + np.tensordot(dxcdt,d2td2xs,axes=[1,0])



        DxcDxi = self.dxcdxi(t)             # shape (20,)
        D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)
        D3xcDxiD2t = self.d3xcdxid2t(t)     # shape (20,2,2)
        DtDxi      = np.empty((20,2,3)  ,dtype=float)
        D2fDxiDt   = np.empty((20,2,3,2),dtype=float)
        DxcDxi_tot = np.empty((20,3,3)  ,dtype=float)

        #########################
        ## fintC & KC assembly ##
        #########################

        # fintC,KC     Slave-Slave :
        dgndu[:3] += dgndxs + dgndxc @ dxcdxs
        d2gnd2u[:3,:3] +=  d2gnd2xs + 2*d2gndxsdxc@dxcdxs + d2gnd2xc@dxcdxs@dxcdxs + np.tensordot(dgndxc,d2xcd2xs, axes=[0,0])
        if t0 is not None:    
            dgtdu[:3] += dgtdxc @ dxcdxs
            d2gtd2u[:3,:3] +=  np.tensordot(d2gtd2xc@dxcdxs,dxcdxs, axes=[0,0]) + dgtdxc@d2xcd2xs

        DxiDu   = np.empty((20,3,3*(Ne+1))  ,dtype=float)        # will be used in nested loop below
        for idx in range(1,21):
            dxcdxi = DxcDxi[idx-1]*np.eye(3)
            if t0 is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)

            d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
            d3xcdxid2t = np.array([[ D3xcDxiD2t[idx-1,0,0]*np.eye(3) , D3xcDxiD2t[idx-1,0,1]*np.eye(3) ],
                                   [ D3xcDxiD2t[idx-1,1,0]*np.eye(3) , D3xcDxiD2t[idx-1,1,1]*np.eye(3) ]]).swapaxes(0,2).swapaxes(1,3)

            dfdxi = 2*((dxcdxi.T@dxcdt) - np.tensordot(xs-xc,d2xcdxidt,axes=[0,0])).T
            dtdxi = -invdfdt@dfdxi          # torch-verified

            DtDxi[idx-1] = dtdxi

            d2fdxidt  = 2*(np.tensordot(dxcdxi,d2xcd2t,axes=[0,0]) + np.tensordot(d2xcdxidt,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcdxidt,axes=[0,0]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcdxid2t,axes=[0,0])).swapaxes(0,1)
            D2fDxiDt[idx-1] = d2fdxidt
            d2fdxidxs = -2*d2xcdxidt.swapaxes(0,2)
            d2tdxidxs = np.tensordot(-invdfdt,  ((d2fdt2@dtdxi).swapaxes(1,2)+d2fdxidt)@dtdxs + (d2fdxsdt@dtdxi).swapaxes(1,2) + d2fdxidxs , axes = [1,0])
            d2xcdxidxs = np.tensordot(dxcdt,d2tdxidxs,axes=[1,0]) + ((d2xcd2t@dtdxi).swapaxes(1,2)+d2xcdxidt)@dtdxs

            dxcdxi += dxcdt@dtdxi # checking here <====
            DxcDxi_tot[idx-1] = dxcdxi

            if idx < 5:
                dxidu = np.zeros((3,3*(Ne+1)))
                dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
                d2xid2u = 0
            elif idx < 13:
                idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
                inverse = [True,False][idx%2]   # CW / CCW
                dxidu = np.zeros((3,3*(Ne+1)))
                dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
                d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
                d2xid2u[:,3:,3:] = self.d2y1d2u(idx0,inverse=inverse)
            else:
                idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
                inverse = [False, True, True, False, False, True, True, False][idx-13]
                dxidu = np.zeros((3,3*(Ne+1)))
                dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)
                d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
                d2xid2u[:,3:,3:] = self.d2yp1d2u(idx0,inverse=inverse)

            DxiDu[idx-1] = dxidu            # has to be stored for ALL CtrlPts before starting nested loop!

            # fintC:i   ,   KC: s-i  i-s  i-i
            dgndu += dgndxc @ dxcdxi @ dxidu
            d2gnd2u[:3,:] += ((d2gndxsdxc + (d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T)@dxidu
            d2gnd2u[ : , :3] += (((d2gndxsdxc+(d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T) @ dxidu).T
            d2gnd2u += np.tensordot(dgndxc@dxcdxi,d2xid2u,axes=[0,0]) if type(d2xid2u)!=int else 0
            if t0 is not None:
                dgtdu += (dgtdxc@dxcdxi + dgtdxc0@dxc0dxi) @ dxidu

                si = (np.tensordot(d2gtd2xc@dxcdxs,dxcdxi,axes=[0,0]) + np.tensordot(dgtdxc,d2xcdxidxs,axes=[0,0]).T + np.tensordot(d2gtdxc0dxc@dxcdxs,dxc0dxi,axes=[0,0]))@dxidu
                # wer = ((np.tensordot(d2gtd2xc@dxcdxi,dxcdxs,axes=[0,0]) + np.tensordot(d2gtdxc0dxc.T@dxc0dxi,dxcdxs,axes=[0,0]) + np.tensordot(dgtdxc,d2xcdxidxs,axes=[0,0])).T@dxidu).T
                d2gtd2u[ :3, : ] += si
                d2gtd2u[ : , :3] += si.T
                ii= np.tensordot( dgtdxc@dxcdxi+dgtdxc0@dxc0dxi , d2xid2u , axes=[0,0]) if type(d2xid2u)!=int else 0
                d2gtd2u += ii

            # K     CtrlPts-CtrlPts (2):



        for idx in range(1,21): #Necessary! it goes over the (previously) stored DxiDu to used them as dxjdu, for example
            dxidu = DxiDu[idx-1]
            dxcdxi = DxcDxi_tot[idx-1]
            d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
            dtdxi = DtDxi[idx-1]
            d2fdxidt = D2fDxiDt[idx-1]
            for jdx in range(1,21):
                dxjdu = DxiDu[jdx-1]
                dxcdxj = DxcDxi_tot[jdx-1]
                d2xcdxjdt = np.array([D2xcDxiDt[jdx-1,0]*np.eye(3),D2xcDxiDt[jdx-1,1]*np.eye(3)]).swapaxes(0,2)
                dtdxj = DtDxi[jdx-1]
                d2fdxjdt = D2fDxiDt[jdx-1]

                d2fdxidxj = 2*(np.tensordot(dxcdxi,d2xcdxjdt, axes = [0,0]).swapaxes(0,1) + np.tensordot(dxcdxj,d2xcdxidt,axes=[0,0])).swapaxes(0,2)
                d2tdxidxj = np.tensordot(-invdfdt , (((d2fdt2@dtdxj).swapaxes(1,2)+d2fdxjdt)@dtdxi).swapaxes(1,2) + d2fdxidxj + d2fdxidt@dtdxj , axes=[1,0])
                
                d2xcdxidxj = d2xcdxidt@dtdxj + ((d2xcdxjdt+(d2xcd2t@dtdxj).swapaxes(1,2))@dtdxi).swapaxes(1,2) + np.tensordot(dxcdt,d2tdxidxj,axes=[1,0])
                d2gndxidxj = (d2gnd2xc@dxcdxi).T@dxcdxj + np.tensordot(dgndxc,d2xcdxidxj,axes=[0,0])

                # set_trace()

                d2gnd2u += (d2gndxidxj@dxjdu).T@dxidu

                if t0 is not None:
                    dxc0dxj = Dxc0Dxi[jdx-1]*np.eye(3)
                    d2gtd2u += ((((d2gtd2xc@dxcdxj+d2gtdxc0dxc.T@dxc0dxj).T@dxcdxi).T + ((d2gtdxc0dxc@dxcdxj + d2gtd2xc0@dxc0dxj).T@dxc0dxi).T + dgtdxc@d2xcdxidxj)@dxjdu).T@dxidu

                
        if cubicT==None:
            if t0 is not None:
                return kn*(np.outer(dgndu,dgndu)+gn*d2gnd2u) + kt*(np.outer(dgtdu,dgtdu)+gt*d2gtd2u)
            else:
                return kn*(np.outer(dgndu,dgndu)+gn*d2gnd2u)
        elif gn<cubicT:
            return kn*(np.outer(dgndu,dgndu)+(gn-cubicT/2)*d2gnd2u)
        else:
            return kn*((gn/cubicT)*np.outer(dgndu,dgndu)+(gn**2/(2*cubicT))*d2gnd2u)

    def KC2(self, xs, kn,kt, f0, seeding = 10, cubicT = None, OPA = 1e-8, t0=None, gn0=None, Sticking=True):       #Chain rule by hand
        """gt is distance between xs0 and xs"""
        # rho = 1e-4
        rho = 1

        Ne = len(self.squad)

        t = self.findProjection(xs)

        if np.allclose(t0-t,0.0): return self.KC(xs, kn, seeding=seeding, cubicT=cubicT, OPA=OPA)

        xc, dxcdt, d2xcd2t, d3xcd3t= self.Grg(t, deriv = 3)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal
        dgndu    = np.zeros(  3*(Ne+1) )
        d2gnd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxs = -dgndxc
        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)
        dxcdxs = dxcdt @ dtdxs


        # Vars for K---------------------------------------------------
        d2gnd2xs = np.eye(3)/gn - np.outer((xs-xc),(xs-xc))/gn**3         
        d2gnd2xc = d2gnd2xs
        d2gndxsdxc = -d2gnd2xs
        if t0 is not None:
            dgtdu    = np.zeros(  3*(Ne+1) )
            d2gtd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )
            xc0= self.Grg(t0)
            n0 = self.D3Grg(t0)
            Dn0Dxi,D2n0DxiDxj = self.dndxi(t0, degree=2)
            xs0 = xc0 - gn0*n0
            dxs0dxc0 = np.eye(3)
            dxs0dn0 = -gn0*np.eye(3)
            Dxc0Dxi = self.dxcdxi(t0)
            gt = norm(xs-xs0)
            # set_trace()
            dgtdxs  = (xs-xs0)/norm(xs-xs0)
            dgtdxs0 = -dgtdxs
            d2gtd2xs = np.eye(3)/gt - np.outer((xs-xs0),(xs-xs0))/gt**3
            d2gtd2xs0 = d2gtd2xs
            d2gtdxsdxs0 = -d2gtd2xs
            s = rho*kt


        # d/du(dxcdxs)
        d2fdt2  = 2*(np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]) + np.tensordot(d2xcd2t,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcd3t,axes=[[0],[0]]) )
        invdfdt = np.linalg.inv(dfdt)

        d2fdxsdt = -2*d2xcd2t.swapaxes(0,1)
        d2td2xs = np.tensordot(-invdfdt,(np.tensordot(np.tensordot(dtdxs,d2fdt2,axes=[0,1]),dtdxs,axes=[2,0]).swapaxes(0,1)  + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]) + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]).swapaxes(1,2)),axes = [[1],[0]])
        d2xcd2xs = np.tensordot( np.tensordot(dtdxs,d2xcd2t,axes=[0,2]),dtdxs,axes=[2,0]).swapaxes(0,1) + np.tensordot(dxcdt,d2td2xs,axes=[1,0])

        DxcDxi = self.dxcdxi(t)             # shape (20,)
        D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)
        D3xcDxiD2t = self.d3xcdxid2t(t)     # shape (20,2,2)
        DtDxi      = np.empty((20,2,3)  ,dtype=float)
        D2fDxiDt   = np.empty((20,2,3,2),dtype=float)
        DxcDxi_tot = np.empty((20,3,3)  ,dtype=float)

        #########################
        ## fintC & KC assembly ##
        #########################

        # fintC,KC     Slave-Slave :
        dgndu[:3] += dgndxs + dgndxc @ dxcdxs
        d2gnd2u[:3,:3] +=  d2gnd2xs + 2*d2gndxsdxc@dxcdxs + d2gnd2xc@dxcdxs@dxcdxs + np.tensordot(dgndxc,d2xcd2xs, axes=[0,0])
        if t0 is not None:    
            dgtdu[:3] += dgtdxs
            d2gtd2u[:3,:3] += d2gtd2xs


        DxiDu   = np.empty((20,3,3*(Ne+1))  ,dtype=float)        # will be used in nested loop below
        for idx in range(1,21):
            dxcdxi = DxcDxi[idx-1]*np.eye(3)
            if t0 is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi = Dn0Dxi[idx-1]

            d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
            d3xcdxid2t = np.array([[ D3xcDxiD2t[idx-1,0,0]*np.eye(3) , D3xcDxiD2t[idx-1,0,1]*np.eye(3) ],
                                   [ D3xcDxiD2t[idx-1,1,0]*np.eye(3) , D3xcDxiD2t[idx-1,1,1]*np.eye(3) ]]).swapaxes(0,2).swapaxes(1,3)

            dfdxi = 2*((dxcdxi.T@dxcdt) - np.tensordot(xs-xc,d2xcdxidt,axes=[0,0])).T
            dtdxi = -invdfdt@dfdxi          # torch-verified

            DtDxi[idx-1] = dtdxi

            d2fdxidt  = 2*(np.tensordot(dxcdxi,d2xcd2t,axes=[0,0]) + np.tensordot(d2xcdxidt,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcdxidt,axes=[0,0]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcdxid2t,axes=[0,0])).swapaxes(0,1)
            D2fDxiDt[idx-1] = d2fdxidt
            d2fdxidxs = -2*d2xcdxidt.swapaxes(0,2)
            d2tdxidxs = np.tensordot(-invdfdt,  ((d2fdt2@dtdxi).swapaxes(1,2)+d2fdxidt)@dtdxs + (d2fdxsdt@dtdxi).swapaxes(1,2) + d2fdxidxs , axes = [1,0])
            d2xcdxidxs = np.tensordot(dxcdt,d2tdxidxs,axes=[1,0]) + ((d2xcd2t@dtdxi).swapaxes(1,2)+d2xcdxidt)@dtdxs

            dxcdxi += dxcdt@dtdxi # checking here <====
            DxcDxi_tot[idx-1] = dxcdxi

            if idx < 5:
                dxidu = np.zeros((3,3*(Ne+1)))
                dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
                d2xid2u = 0
            elif idx < 13:
                idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
                inverse = [True,False][idx%2]   # CW / CCW
                dxidu = np.zeros((3,3*(Ne+1)))
                dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
                d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
                d2xid2u[:,3:,3:] = self.d2y1d2u(idx0,inverse=inverse)
            else:
                idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
                inverse = [False, True, True, False, False, True, True, False][idx-13]
                dxidu = np.zeros((3,3*(Ne+1)))
                dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)
                d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
                d2xid2u[:,3:,3:] = self.d2yp1d2u(idx0,inverse=inverse)

            DxiDu[idx-1] = dxidu

            # set_trace()

            # fintC:i   ,   KC: s-i  i-s  i-i
            dgndu += dgndxc @ dxcdxi @ dxidu
            d2gnd2u[:3,:] += ((d2gndxsdxc + (d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T)@dxidu
            d2gnd2u[ : , :3] += (((d2gndxsdxc+(d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T) @ dxidu).T
            d2gnd2u += np.tensordot(dgndxc@dxcdxi,d2xid2u,axes=[0,0]) if type(d2xid2u)!=int else 0
            if t0 is not None:
                dxs0dxi = dxs0dxc0@dxc0dxi+dxs0dn0@dn0dxi
                dgtdu += dgtdxs0@dxs0dxi@ dxidu

                i_s = np.tensordot(d2gtdxsdxs0,dxs0dxi,axes=[1,0])@dxidu
                # s_i = np.tensordot(d2gtdxsdxs0,dxs0dxc0@dxc0dxi,axes=[1,0])     # d2gtdxsdxs0 == d2gtdxs0dxs

                # wer = ((np.tensordot(d2gtd2xc@dxcdxi,dxcdxs,axes=[0,0]) + np.tensordot(d2gtdxc0dxc.T@dxc0dxi,dxcdxs,axes=[0,0]) + np.tensordot(dgtdxc,d2xcdxidxs,axes=[0,0])).T@dxidu).T
                d2gtd2u[ :3, : ] += i_s
                d2gtd2u[ : , :3] += i_s.T
                ii= np.tensordot(dgtdxs0@dxs0dxi,d2xid2u,axes=[0,0]) if type(d2xid2u)!=int else 0
                d2gtd2u += ii

            # K     CtrlPts-CtrlPts (2):

        for idx in range(1,21):
            dxidu = DxiDu[idx-1]
            dxcdxi = DxcDxi_tot[idx-1]
            d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
            dtdxi = DtDxi[idx-1]
            d2fdxidt = D2fDxiDt[idx-1]
            if t0 is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi = Dn0Dxi[idx-1]
                dxs0dxi = dxs0dxc0@dxc0dxi + dxs0dn0@dn0dxi

            for jdx in range(1,21):
                dxjdu = DxiDu[jdx-1]
                dxcdxj = DxcDxi_tot[jdx-1]
                d2xcdxjdt = np.array([D2xcDxiDt[jdx-1,0]*np.eye(3),D2xcDxiDt[jdx-1,1]*np.eye(3)]).swapaxes(0,2)
                dtdxj = DtDxi[jdx-1]
                d2fdxjdt = D2fDxiDt[jdx-1]

                d2fdxidxj = 2*(np.tensordot(dxcdxi,d2xcdxjdt, axes = [0,0]).swapaxes(0,1) + np.tensordot(dxcdxj,d2xcdxidt,axes=[0,0])).swapaxes(0,2)
                d2tdxidxj = np.tensordot(-invdfdt , (((d2fdt2@dtdxj).swapaxes(1,2)+d2fdxjdt)@dtdxi).swapaxes(1,2) + d2fdxidxj + d2fdxidt@dtdxj , axes=[1,0])

                d2xcdxidxj = d2xcdxidt@dtdxj + ((d2xcdxjdt+(d2xcd2t@dtdxj).swapaxes(1,2))@dtdxi).swapaxes(1,2) + np.tensordot(dxcdt,d2tdxidxj,axes=[1,0])
                d2gndxidxj = (d2gnd2xc@dxcdxi).T@dxcdxj + np.tensordot(dgndxc,d2xcdxidxj,axes=[0,0])

                d2gnd2u += (d2gndxidxj@dxjdu).T@dxidu

                if t0 is not None:
                    dxc0dxj = Dxc0Dxi[jdx-1]*np.eye(3)
                    dn0dxj = Dn0Dxi[jdx-1]
                    dxs0dxj = dxs0dxc0@dxc0dxj + dxs0dn0@dn0dxj
                    d2n0dxidxj = D2n0DxiDxj[:,:,:,idx-1,jdx-1]
                    d2gtdxidxj = np.tensordot(d2gtd2xs0@dxs0dxi,dxs0dxj,axes=[0,0]) + np.tensordot(dgtdxs0@dxs0dn0,d2n0dxidxj,axes=[0,0])
                    d2gtd2u += (d2gtdxidxj.T@dxidu).T@dxjdu
                    
        if cubicT is None:
            if t0 is not None:
                if Sticking:
                    return kn*(np.outer(dgndu,dgndu)+gn*d2gnd2u) + kt*(np.outer(dgtdu,dgtdu)+gt*d2gtd2u)
                else:
                    # set_trace()
                    f=f0
                    # f=0
                    s=1e-5
                    print("")
                    print("Being plastic right now...")
                    print("")
                    return kn*(np.outer(dgndu,dgndu)+gn*d2gnd2u) + s*np.outer(dgtdu,dgtdu)+f*d2gtd2u
            else:
                return kn*(np.outer(dgndu,dgndu)+gn*d2gnd2u)
        elif gn<cubicT:
            return kn*(np.outer(dgndu,dgndu)+(gn-cubicT/2)*d2gnd2u)
        else:
            return kn*((gn/cubicT)*np.outer(dgndu,dgndu)+(gn**2/(2*cubicT))*d2gnd2u)

    def KC2_tang(self, xs, kt, f0, seeding = 10, cubicT = None, OPA = 1e-8, t0=None, gn0=None, Sticking=True):       #Chain rule by hand
        """gt is distance between xs0 and xs"""
        Ne = len(self.squad)

        # Vars for K---------------------------------------------------
        dgtdu    = np.zeros(  3*(Ne+1) )
        d2gtd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )
        xc0= self.Grg(t0)
        n0 = self.D3Grg(t0)
        Dn0Dxi,D2n0DxiDxj = self.dndxi(t0, degree=2)
        xs0 = xc0 - gn0*n0
        dxs0dxc0 = np.eye(3)
        dxs0dn0 = -gn0*np.eye(3)
        Dxc0Dxi = self.dxcdxi(t0)
        gt = norm(xs-xs0)
        # set_trace()

        dgtdxs  = (xs-xs0)/norm(xs-xs0)
        dgtdxs0 = -dgtdxs
        d2gtd2xs = np.eye(3)/gt - np.outer((xs-xs0),(xs-xs0))/gt**3
        d2gtd2xs0 = d2gtd2xs
        d2gtdxsdxs0 = -d2gtd2xs

        #########################
        ## fintC & KC assembly ##
        #########################
        # fintC,KC     Slave-Slave :
        dgtdu[:3] += dgtdxs
        d2gtd2u[:3,:3] += d2gtd2xs

        DxiDu   = np.empty((20,3,3*(Ne+1))  ,dtype=float)        # will be used in nested loop below
        for idx in range(1,21):
            dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
            dn0dxi = Dn0Dxi[idx-1]

            if idx < 5:
                dxidu = np.zeros((3,3*(Ne+1)))
                dxidu[:,3*idx:3*(idx+1)] = np.eye(3)
                d2xid2u = 0
            elif idx < 13:
                idx0    = int((idx-5)/2)        # edge in which the CtrlPt belongs
                inverse = [True,False][idx%2]   # CW / CCW
                dxidu = np.zeros((3,3*(Ne+1)))
                dxidu[:,3:] = self.dy1du(idx0,inverse=inverse)
                d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
                d2xid2u[:,3:,3:] = self.d2y1d2u(idx0,inverse=inverse)
            else:
                idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the CtrlPt belongs
                inverse = [False, True, True, False, False, True, True, False][idx-13]
                dxidu = np.zeros((3,3*(Ne+1)))
                dxidu[:,3:] = self.dyp1du(idx0,inverse=inverse)
                d2xid2u = np.zeros((3,3*(Ne+1),3*(Ne+1)))
                d2xid2u[:,3:,3:] = self.d2yp1d2u(idx0,inverse=inverse)

            DxiDu[idx-1] = dxidu

            # fintC:i   ,   KC: s-i  i-s  i-i
            dxs0dxi = dxs0dxc0@dxc0dxi+dxs0dn0@dn0dxi
            dgtdu += dgtdxs0@dxs0dxi@ dxidu

            i_s = np.tensordot(d2gtdxsdxs0,dxs0dxi,axes=[1,0])@dxidu
            d2gtd2u[ :3, : ] += i_s
            d2gtd2u[ : , :3] += i_s.T
            ii= np.tensordot(dgtdxs0@dxs0dxi,d2xid2u,axes=[0,0]) if type(d2xid2u)!=int else 0
            d2gtd2u += ii

        # K     CtrlPts-CtrlPts (2):
        for idx in range(1,21):
            dxidu = DxiDu[idx-1]
            dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
            dn0dxi = Dn0Dxi[idx-1]
            dxs0dxi = dxs0dxc0@dxc0dxi + dxs0dn0@dn0dxi

            for jdx in range(1,21):
                dxjdu = DxiDu[jdx-1]
                dxc0dxj = Dxc0Dxi[jdx-1]*np.eye(3)
                dn0dxj = Dn0Dxi[jdx-1]
                dxs0dxj = dxs0dxc0@dxc0dxj + dxs0dn0@dn0dxj

                d2n0dxidxj = D2n0DxiDxj[:,:,:,idx-1,jdx-1]
                d2gtdxidxj = np.tensordot(d2gtd2xs0@dxs0dxi,dxs0dxj,axes=[0,0]) + np.tensordot(dgtdxs0@dxs0dn0,d2n0dxidxj,axes=[0,0])
                d2gtd2u += (d2gtdxidxj.T@dxidu).T@dxjdu
                    
                
        if Sticking:
            return kt*(np.outer(dgtdu,dgtdu)+gt*d2gtd2u)
        else:
            f=f0
            s=0
            print("")
            print("Being plastic right now... (tangential part)")
            print("")
            return s*np.outer(dgtdu,dgtdu)+f*d2gtd2u



    def KC_tot(self,xs,kn,kt,mu,hook=None,Stick=True, cubicT=None, output=None, niter=3, new=False):
        """gt is distance between xs0 and xs"""
        Ne = len(self.squad)

        t = self.findProjection(xs)

        xc, dxcdt, d2xcd2t, d3xcd3t= self.Grg(t, deriv = 3)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal
        dgndu    = np.zeros(  3*(Ne+1) )
        d2gnd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxs = -dgndxc
        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)
        dxcdxs = dxcdt @ dtdxs

        # Vars for K---------------------------------------------------
        d2gnd2xs = np.eye(3)/gn - np.outer((xs-xc),(xs-xc))/gn**3         
        d2gnd2xc = d2gnd2xs
        d2gndxsdxc = -d2gnd2xs
        if hook is not None:
            ihp = int(hook[0])
            hpatch = self.surf.patches[ihp] if ihp!=self.iquad else self
            hNe = len(hpatch.squad)
            dgtdu    = np.zeros(  3*(hNe+1) )
            d2gtd2u  = np.zeros( (3*(hNe+1) , 3*(hNe+1)) )
            t0 = hook[1:3]
            # if np.allclose(t0-t,0.0): return self.KC_tot(xs,kn,kt,mu,hook=None, cubicT=cubicT), None, None
            gn0 = hook[3]
            xc0= hpatch.Grg(t0)
            n0 = hpatch.D3Grg(t0)
            xs0 = xc0 - gn0*n0
            gt = norm(xs-xs0)
            if gt == 0 : return self.KC_tot(xs,kn,kt,mu,hook=None,Stick=True, cubicT=cubicT, output=None, niter=3, new=new), None,None
            dxs0dxc0 = np.eye(3)
            dxs0dn0 = -gn0*np.eye(3)
            dgtdxs  = (xs-xs0)/norm(xs-xs0)
            dgtdxs0 = -dgtdxs
            d2gtd2xs = np.eye(3)/gt - np.outer((xs-xs0),(xs-xs0))/gt**3
            d2gtd2xs0 = d2gtd2xs
            Dxc0Dxi = hpatch.dxcdxi(t0)
            Dn0Dxi,D2n0DxiDxj = hpatch.dndxi(t0, degree=2)
            d2gtdxsdxs0 = -d2gtd2xs
            dgtdu[:3] += dgtdxs
            d2gtd2u[:3,:3] += d2gtd2xs


        # d/du(dxcdxs)
        d2fdt2  = 2*(np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]) + np.tensordot(d2xcd2t,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcd3t,axes=[[0],[0]]) )
        invdfdt = np.linalg.inv(dfdt)

        d2fdxsdt = -2*d2xcd2t.swapaxes(0,1)
        d2td2xs = np.tensordot(-invdfdt,(np.tensordot(np.tensordot(dtdxs,d2fdt2,axes=[0,1]),dtdxs,axes=[2,0]).swapaxes(0,1)  + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]) + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]).swapaxes(1,2)),axes = [[1],[0]])
        d2xcd2xs = np.tensordot( np.tensordot(dtdxs,d2xcd2t,axes=[0,2]),dtdxs,axes=[2,0]).swapaxes(0,1) + np.tensordot(dxcdt,d2td2xs,axes=[1,0])

        DxcDxi = self.dxcdxi(t)             # shape (20,)
        D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)
        D3xcDxiD2t = self.d3xcdxid2t(t)     # shape (20,2,2)
        DtDxi      = np.empty((20,2,3)  ,dtype=float)
        D2fDxiDt   = np.empty((20,2,3,2),dtype=float)
        DxcDxi_tot = np.empty((20,3,3)  ,dtype=float)

        #########################
        ## fintC & KC assembly ##
        #########################

        # fintC,KC     Slave-Slave :
        dgndu[:3] += dgndxs + dgndxc @ dxcdxs
        d2gnd2u[:3,:3] +=  d2gnd2xs + 2*d2gndxsdxc@dxcdxs + d2gnd2xc@dxcdxs@dxcdxs + np.tensordot(dgndxc,d2xcd2xs, axes=[0,0])

        DxiDu   = np.empty((20,3,3*(Ne+1))  ,dtype=float)        # will be used in nested loop below
        if hook is not None: DxiDuh  = np.empty((20,3,3*(hNe+1))  ,dtype=float)        # will be used in nested loop below
        for idx in range(1,21):
            dxcdxi = DxcDxi[idx-1]*np.eye(3)
            if hook is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi = Dn0Dxi[idx-1]

            d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
            d3xcdxid2t = np.array([[ D3xcDxiD2t[idx-1,0,0]*np.eye(3) , D3xcDxiD2t[idx-1,0,1]*np.eye(3) ],
                                   [ D3xcDxiD2t[idx-1,1,0]*np.eye(3) , D3xcDxiD2t[idx-1,1,1]*np.eye(3) ]]).swapaxes(0,2).swapaxes(1,3)

            dfdxi = 2*((dxcdxi.T@dxcdt) - np.tensordot(xs-xc,d2xcdxidt,axes=[0,0])).T
            dtdxi = -invdfdt@dfdxi          # torch-verified

            DtDxi[idx-1] = dtdxi

            d2fdxidt  = 2*(np.tensordot(dxcdxi,d2xcd2t,axes=[0,0]) + np.tensordot(d2xcdxidt,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcdxidt,axes=[0,0]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcdxid2t,axes=[0,0])).swapaxes(0,1)
            D2fDxiDt[idx-1] = d2fdxidt
            d2fdxidxs = -2*d2xcdxidt.swapaxes(0,2)
            d2tdxidxs = np.tensordot(-invdfdt,  ((d2fdt2@dtdxi).swapaxes(1,2)+d2fdxidt)@dtdxs + (d2fdxsdt@dtdxi).swapaxes(1,2) + d2fdxidxs , axes = [1,0])
            d2xcdxidxs = np.tensordot(dxcdt,d2tdxidxs,axes=[1,0]) + ((d2xcd2t@dtdxi).swapaxes(1,2)+d2xcdxidt)@dtdxs

            dxcdxi += dxcdt@dtdxi # checking here <====
            DxcDxi_tot[idx-1] = dxcdxi

            dxidu, d2xid2u = self.dxidu(idx,order=2)
            DxiDu[idx-1] = dxidu

            # set_trace()

            # fintC:i   ,   KC: s-i  i-s  i-i
            dgndu += dgndxc @ dxcdxi @ dxidu
            d2gnd2u[:3,:] += ((d2gndxsdxc + (d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T)@dxidu
            d2gnd2u[ : , :3] += (((d2gndxsdxc+(d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T) @ dxidu).T
            d2gnd2u += np.tensordot(dgndxc@dxcdxi,d2xid2u,axes=[0,0]) if type(d2xid2u)!=int else 0
            if hook is not None:
                dxiduh, d2xid2uh = (dxidu, d2xid2u) if ihp==self.iquad else hpatch.dxidu(idx,order=2)
                DxiDuh[idx-1] = dxiduh

                dxs0dxi = dxs0dxc0@dxc0dxi+dxs0dn0@dn0dxi
                dgtdu += dgtdxs0@dxs0dxi@ dxiduh

                i_s = np.tensordot(d2gtdxsdxs0,dxs0dxi,axes=[1,0])@dxiduh

                d2gtd2u[ :3, : ] += i_s
                d2gtd2u[ : , :3] += i_s.T
                ii= np.tensordot(dgtdxs0@dxs0dxi,d2xid2uh,axes=[0,0]) if type(d2xid2uh)!=int else 0
                d2gtd2u += ii

            # K     CtrlPts-CtrlPts (2):

        for idx in range(1,21):
            dxidu = DxiDu[idx-1]
            dxcdxi = DxcDxi_tot[idx-1]
            d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
            dtdxi = DtDxi[idx-1]
            d2fdxidt = D2fDxiDt[idx-1]
            if hook is not None:
                dxiduh = DxiDuh[idx-1]
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi = Dn0Dxi[idx-1]
                dxs0dxi = dxs0dxc0@dxc0dxi + dxs0dn0@dn0dxi

            for jdx in range(1,21):
                dxjdu = DxiDu[jdx-1]
                dxcdxj = DxcDxi_tot[jdx-1]
                d2xcdxjdt = np.array([D2xcDxiDt[jdx-1,0]*np.eye(3),D2xcDxiDt[jdx-1,1]*np.eye(3)]).swapaxes(0,2)
                dtdxj = DtDxi[jdx-1]
                d2fdxjdt = D2fDxiDt[jdx-1]

                d2fdxidxj = 2*(np.tensordot(dxcdxi,d2xcdxjdt, axes = [0,0]).swapaxes(0,1) + np.tensordot(dxcdxj,d2xcdxidt,axes=[0,0])).swapaxes(0,2)
                d2tdxidxj = np.tensordot(-invdfdt , (((d2fdt2@dtdxj).swapaxes(1,2)+d2fdxjdt)@dtdxi).swapaxes(1,2) + d2fdxidxj + d2fdxidt@dtdxj , axes=[1,0])

                d2xcdxidxj = d2xcdxidt@dtdxj + ((d2xcdxjdt+(d2xcd2t@dtdxj).swapaxes(1,2))@dtdxi).swapaxes(1,2) + np.tensordot(dxcdt,d2tdxidxj,axes=[1,0])
                d2gndxidxj = (d2gnd2xc@dxcdxi).T@dxcdxj + np.tensordot(dgndxc,d2xcdxidxj,axes=[0,0])

                d2gnd2u += (d2gndxidxj@dxjdu).T@dxidu

                if hook is not None:
                    dxjduh = DxiDuh[jdx-1]
                    dxc0dxj = Dxc0Dxi[jdx-1]*np.eye(3)
                    dn0dxj = Dn0Dxi[jdx-1]
                    dxs0dxj = dxs0dxc0@dxc0dxj + dxs0dn0@dn0dxj
                    d2n0dxidxj = D2n0DxiDxj[:,:,:,idx-1,jdx-1]
                    d2gtdxidxj = np.tensordot(d2gtd2xs0@dxs0dxi,dxs0dxj,axes=[0,0]) + np.tensordot(dgtdxs0@dxs0dn0,d2n0dxidxj,axes=[0,0])
                    d2gtd2u += (d2gtdxidxj.T@dxiduh).T@dxjduh

        # if (niter>=3) and (hook is not None): set_trace()
        if type(output) is str:
            return locals()[output]

        if cubicT is None:
            KCN = kn*(np.outer(dgndu,dgndu)+gn*d2gnd2u)
        elif gn<cubicT:
            KCN = kn*(np.outer(dgndu,dgndu)+(gn-cubicT/2)*d2gnd2u)
        else:
            KCN = kn*((gn/cubicT)*np.outer(dgndu,dgndu)+(gn**2/(2*cubicT))*d2gnd2u)
        
        cond = (niter<2 and new) or (niter==0)
        printif(cond, "ELASTIC IMPOSED")
        
        if hook is None:
            return KCN
        # elif Stick:
        # elif kt*gt<-mu*kn*gn:
        elif (kt*gt<-mu*kn*gn or cond):
            print("elastic KC")
            # print("NITER: ", niter)
            KCT = kt*(np.outer(dgtdu,dgtdu)+gt*d2gtd2u)
            KCNT = None
        else:
            print("plastic KC")
            KCT = -mu*kn*gn*d2gtd2u
            # dgNdu = np.zeros(3*(Ne+1))
            # dgNdu[:3]=dgndxs + dgndxc@dxcdxs
            # KCNT =-mu*kn*np.outer(dgtdu,dgNdu) 
            KCNT =-mu*kn*np.outer(dgtdu,dgndu) 

        return KCN,KCT,KCNT

    def KC_tot2(self,xs,kn,kt,mu,hook=None,Stick=True, cubicT=None, output=None, niter=3, new=False):
        """gt is orthogonal distance between xs and line passing through xc0 with direction n0"""
        Ne = len(self.squad)

        t = self.findProjection(xs)

        xc, dxcdt, d2xcd2t, d3xcd3t= self.Grg(t, deriv = 3)
        normal = self.D3Grg(t)
        gn = (xs-xc)@normal
        dgndu    = np.zeros(  3*(Ne+1) )
        d2gnd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgndxs = -dgndxc
        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)
        dxcdxs = dxcdt @ dtdxs

        # Vars for K---------------------------------------------------
        d2gnd2xs = np.eye(3)/gn - np.outer((xs-xc),(xs-xc))/gn**3         
        d2gnd2xc = d2gnd2xs
        d2gndxsdxc = -d2gnd2xs
        if hook is not None:
            ihp = int(hook[0])
            hpatch = self.surf.patches[ihp] if ihp!=self.iquad else self
            hNe = len(hpatch.squad)
            dgtdu    = np.zeros(  3*(hNe+1) )
            d2gtd2u  = np.zeros( (3*(hNe+1) , 3*(hNe+1)) )
            t0 = hook[1:3]
            # if np.allclose(t0-t,0.0): return self.KC_tot2(xs,kn,kt,mu,hook=None, cubicT=cubicT), None, None
            gn0 = hook[3]
            xc0= hpatch.Grg(t0)
            n0 = hpatch.D3Grg(t0)
            xsp = xc0 + np.dot(xs-xc0,n0)*n0
            gt = norm(xs-xsp)
            if gt == 0 : return self.KC_tot2(xs,kn,kt,mu,hook=None,Stick=True, cubicT=cubicT, output=None, niter=3, new=new), None,None
            dgtdxs  = (xs-xsp)/norm(xs-xsp)
            dgtdxsp = -dgtdxs
            d2gtd2xs = np.eye(3)/gt - np.outer((xs-xsp),(xs-xsp))/gt**3
            d2gtd2xsp = d2gtd2xs
            d2gtdxsdxsp = -d2gtd2xs
            dxspdxs = np.outer(n0,n0)
            dxspdxc0= np.eye(3)-dxspdxs
            dxspdn0 = np.outer(n0,xs-xc0)+np.dot(xs-xc0,n0)*np.eye(3)   #TODO: check order of terms for outer product
            # to check ...
            d2xspd2n0   = (np.multiply.outer(np.eye(3),xs-xc0)+np.multiply.outer(xs-xc0,np.eye(3))).swapaxes(0,1)
            d2xspdn0dxs = np.multiply.outer(np.eye(3),n0)+np.multiply.outer(n0,np.eye(3))
            d2xspdxsdn0  = d2xspdn0dxs.swapaxes(1,2)            
            d2xspdxc0dn0= -d2xspdxsdn0
            d2xspdn0dxc0= -d2xspdn0dxs
            
            Dxc0Dxi = hpatch.dxcdxi(t0)
            Dn0Dxi,D2n0DxiDxj = hpatch.dndxi(t0, degree=2)
            dgtdu[:3] += dgtdxs + dgtdxsp@dxspdxs
            d2gtd2u[:3,:3] += d2gtd2xs + 2*d2gtdxsdxsp@dxspdxs + (d2gtd2xsp@dxspdxs).T@dxspdxs

        # d/du(dxcdxs)
        d2fdt2  = 2*(np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]) + np.tensordot(d2xcd2t,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcd3t,axes=[[0],[0]]) )
        invdfdt = np.linalg.inv(dfdt)

        d2fdxsdt = -2*d2xcd2t.swapaxes(0,1)
        d2td2xs = np.tensordot(-invdfdt,(np.tensordot(np.tensordot(dtdxs,d2fdt2,axes=[0,1]),dtdxs,axes=[2,0]).swapaxes(0,1)  + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]) + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]).swapaxes(1,2)),axes = [[1],[0]])
        d2xcd2xs = np.tensordot( np.tensordot(dtdxs,d2xcd2t,axes=[0,2]),dtdxs,axes=[2,0]).swapaxes(0,1) + np.tensordot(dxcdt,d2td2xs,axes=[1,0])

        DxcDxi = self.dxcdxi(t)             # shape (20,)
        D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)
        D3xcDxiD2t = self.d3xcdxid2t(t)     # shape (20,2,2)
        DtDxi      = np.empty((20,2,3)  ,dtype=float)
        D2fDxiDt   = np.empty((20,2,3,2),dtype=float)
        DxcDxi_tot = np.empty((20,3,3)  ,dtype=float)

        #########################
        ## fintC & KC assembly ##
        #########################

        # fintC,KC     Slave-Slave :
        dgndu[:3] += dgndxs + dgndxc @ dxcdxs
        d2gnd2u[:3,:3] +=  d2gnd2xs + 2*d2gndxsdxc@dxcdxs + d2gnd2xc@dxcdxs@dxcdxs + np.tensordot(dgndxc,d2xcd2xs, axes=[0,0])

        DxiDu   = np.empty((20,3,3*(Ne+1))  ,dtype=float)        # will be used in nested loop below
        if hook is not None: DxiDuh  = np.empty((20,3,3*(hNe+1))  ,dtype=float)        # will be used in nested loop below
        for idx in range(1,21):
            dxcdxi = DxcDxi[idx-1]*np.eye(3)
            if hook is not None:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi = Dn0Dxi[idx-1]

            d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
            d3xcdxid2t = np.array([[ D3xcDxiD2t[idx-1,0,0]*np.eye(3) , D3xcDxiD2t[idx-1,0,1]*np.eye(3) ],
                                   [ D3xcDxiD2t[idx-1,1,0]*np.eye(3) , D3xcDxiD2t[idx-1,1,1]*np.eye(3) ]]).swapaxes(0,2).swapaxes(1,3)

            dfdxi = 2*((dxcdxi.T@dxcdt) - np.tensordot(xs-xc,d2xcdxidt,axes=[0,0])).T
            dtdxi = -invdfdt@dfdxi          # torch-verified

            DtDxi[idx-1] = dtdxi

            d2fdxidt  = 2*(np.tensordot(dxcdxi,d2xcd2t,axes=[0,0]) + np.tensordot(d2xcdxidt,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcdxidt,axes=[0,0]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcdxid2t,axes=[0,0])).swapaxes(0,1)
            D2fDxiDt[idx-1] = d2fdxidt
            d2fdxidxs = -2*d2xcdxidt.swapaxes(0,2)
            d2tdxidxs = np.tensordot(-invdfdt,  ((d2fdt2@dtdxi).swapaxes(1,2)+d2fdxidt)@dtdxs + (d2fdxsdt@dtdxi).swapaxes(1,2) + d2fdxidxs , axes = [1,0])
            d2xcdxidxs = np.tensordot(dxcdt,d2tdxidxs,axes=[1,0]) + ((d2xcd2t@dtdxi).swapaxes(1,2)+d2xcdxidt)@dtdxs

            dxcdxi += dxcdt@dtdxi # checking here <====
            DxcDxi_tot[idx-1] = dxcdxi

            dxidu, d2xid2u = self.dxidu(idx,order=2)
            DxiDu[idx-1] = dxidu

            # set_trace()

            # fintC:i   ,   KC: s-i  i-s  i-i
            dgndu += dgndxc @ dxcdxi @ dxidu
            d2gnd2u[:3,:] += ((d2gndxsdxc + (d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T)@dxidu
            d2gnd2u[ : , :3] += (((d2gndxsdxc+(d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T) @ dxidu).T
            d2gnd2u += np.tensordot(dgndxc@dxcdxi,d2xid2u,axes=[0,0]) if type(d2xid2u)!=int else 0
            if hook is not None:
                dxiduh, d2xid2uh = (dxidu, d2xid2u) if ihp==self.iquad else hpatch.dxidu(idx,order=2)
                DxiDuh[idx-1] = dxiduh

                dxspdxi = dxspdxc0@dxc0dxi+dxspdn0@dn0dxi
                dgtdu += dgtdxsp@dxspdxi@dxiduh

                s_i = ((d2gtdxsdxsp+(d2gtd2xsp@dxspdxs).T)@dxspdxi + dgtdxsp@np.tensordot(d2xspdxsdn0,dn0dxi,axes=[2,0]))@dxiduh

                d2gtd2u[ :3, : ] += s_i
                d2gtd2u[ : , :3] += s_i.T
                i_i= np.tensordot(dgtdxsp@dxspdxi,d2xid2uh,axes=[0,0]) if type(d2xid2uh)!=int else 0
                d2gtd2u += i_i


        # K     CtrlPts-CtrlPts (2):
        for idx in range(1,21):
            dxidu = DxiDu[idx-1]
            dxcdxi = DxcDxi_tot[idx-1]
            d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
            dtdxi = DtDxi[idx-1]
            d2fdxidt = D2fDxiDt[idx-1]
            if hook is not None:
                dxiduh = DxiDuh[idx-1]
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi = Dn0Dxi[idx-1]
                dxspdxi = dxspdxc0@dxc0dxi + dxspdn0@dn0dxi

            for jdx in range(1,21):
                dxjdu = DxiDu[jdx-1]
                dxcdxj = DxcDxi_tot[jdx-1]
                d2xcdxjdt = np.array([D2xcDxiDt[jdx-1,0]*np.eye(3),D2xcDxiDt[jdx-1,1]*np.eye(3)]).swapaxes(0,2)
                dtdxj = DtDxi[jdx-1]
                d2fdxjdt = D2fDxiDt[jdx-1]

                d2fdxidxj = 2*(np.tensordot(dxcdxi,d2xcdxjdt, axes = [0,0]).swapaxes(0,1) + np.tensordot(dxcdxj,d2xcdxidt,axes=[0,0])).swapaxes(0,2)
                d2tdxidxj = np.tensordot(-invdfdt , (((d2fdt2@dtdxj).swapaxes(1,2)+d2fdxjdt)@dtdxi).swapaxes(1,2) + d2fdxidxj + d2fdxidt@dtdxj , axes=[1,0])

                d2xcdxidxj = d2xcdxidt@dtdxj + ((d2xcdxjdt+(d2xcd2t@dtdxj).swapaxes(1,2))@dtdxi).swapaxes(1,2) + np.tensordot(dxcdt,d2tdxidxj,axes=[1,0])
                d2gndxidxj = (d2gnd2xc@dxcdxi).T@dxcdxj + np.tensordot(dgndxc,d2xcdxidxj,axes=[0,0])

                d2gnd2u += (d2gndxidxj@dxjdu).T@dxidu

                if hook is not None:
                    dxjduh = DxiDuh[jdx-1]
                    dxc0dxj = Dxc0Dxi[jdx-1]*np.eye(3)
                    dn0dxj = Dn0Dxi[jdx-1]
                    dxspdxj = dxspdxc0@dxc0dxj + dxspdn0@dn0dxj
                    d2n0dxidxj = D2n0DxiDxj[:,:,:,idx-1,jdx-1]
                    d2xspdxidxj = np.tensordot(np.tensordot(d2xspdxc0dn0,dxc0dxi,axes=[1,0]),dn0dxj,axes=[1,0]) + np.tensordot(d2xspdn0dxc0@dxc0dxj+d2xspd2n0@dn0dxj,dn0dxi,axes=[1,0]).swapaxes(0,1) + np.tensordot(dxspdn0,d2n0dxidxj,axes=[1,0])
                    d2gtdxidxj = np.tensordot(d2gtd2xsp@dxspdxi,dxspdxj,axes=[0,0]) + np.tensordot(dgtdxsp,d2xspdxidxj,axes=[0,0])
                    d2gtd2u += (d2gtdxidxj.T@dxiduh).T@dxjduh

        # if (niter>=3) and (hook is not None): set_trace()
        if type(output) is str:
            return locals()[output]

        if cubicT is None:
            KCN = kn*(np.outer(dgndu,dgndu)+gn*d2gnd2u)
        elif gn<cubicT:
            KCN = kn*(np.outer(dgndu,dgndu)+(gn-cubicT/2)*d2gnd2u)
        else:
            KCN = kn*((gn/cubicT)*np.outer(dgndu,dgndu)+(gn**2/(2*cubicT))*d2gnd2u)
        # new=True
        cond = (niter<2 and new) or (niter==0)
        cond=False
        printif(cond, "ELASTIC IMPOSED")
        # cond = True
        if hook is None:
            return KCN
        # elif Stick:
        # elif kt*gt<-mu*kn*gn:
        elif (kt*gt<-mu*kn*gn or cond):
            KCT = kt*(np.outer(dgtdu,dgtdu)+gt*d2gtd2u)
            KCNT = None
        else:
            KCT = -mu*kn*gn*d2gtd2u
            # dgNdu = np.zeros(3*(Ne+1))
            # # dgNdu[:3]=dgndxs + dgndxc@dxcdxs
            # dgNdu[:3]=dgndxs
            # KCNT =-mu*kn*np.outer(dgtdu,dgNdu) 
            KCNT =-mu*kn*np.outer(dgtdu,dgndu) 

        return KCN,KCT,KCNT


    def KC_tot3(self,xs,kn,kt,mu,hook=None,Stick=True, cubicT=None, output=None, niter=3, new=False):
        """gt is orthogonal distance between xs and line passing through xc0 with direction n0.
        gn is in the direcion of n0 at hooking surface point"""
        Ne = len(self.squad)

        t = self.findProjection(xs)

        if hook is None:
            xc, dxcdt, d2xcd2t, d3xcd3t= self.Grg(t, deriv = 3)
            normal = self.D3Grg(t)
            gn = (xs-xc)@normal
            dgndu    = np.zeros(  3*(Ne+1) )
            d2gnd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

            dgndxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
            dgndxs = -dgndxc
            # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
            dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
            dfdxs = -2*dxcdt.T
            dtdxs = np.linalg.solve(-dfdt,dfdxs)
            dxcdxs = dxcdt @ dtdxs

            # Vars for K---------------------------------------------------
            d2gnd2xs = np.eye(3)/gn - np.outer((xs-xc),(xs-xc))/gn**3         
            d2gnd2xc = d2gnd2xs
            d2gndxsdxc = -d2gnd2xs

            # d/du(dxcdxs)
            d2fdt2  = 2*(np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]) + np.tensordot(d2xcd2t,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcd3t,axes=[[0],[0]]) )
            invdfdt = np.linalg.inv(dfdt)

            d2fdxsdt = -2*d2xcd2t.swapaxes(0,1)
            d2td2xs = np.tensordot(-invdfdt,(np.tensordot(np.tensordot(dtdxs,d2fdt2,axes=[0,1]),dtdxs,axes=[2,0]).swapaxes(0,1)  + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]) + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]).swapaxes(1,2)),axes = [[1],[0]])
            d2xcd2xs = np.tensordot( np.tensordot(dtdxs,d2xcd2t,axes=[0,2]),dtdxs,axes=[2,0]).swapaxes(0,1) + np.tensordot(dxcdt,d2td2xs,axes=[1,0])

            DxcDxi = self.dxcdxi(t)             # shape (20,)
            D2xcDxiDt = self.d2xcdxidt(t)       # shape (20,2)
            D3xcDxiD2t = self.d3xcdxid2t(t)     # shape (20,2,2)
            DtDxi      = np.empty((20,2,3)  ,dtype=float)
            D2fDxiDt   = np.empty((20,2,3,2),dtype=float)
            DxcDxi_tot = np.empty((20,3,3)  ,dtype=float)
            dgndu[:3] += dgndxs + dgndxc @ dxcdxs
            d2gnd2u[:3,:3] +=  d2gnd2xs + 2*d2gndxsdxc@dxcdxs + d2gnd2xc@dxcdxs@dxcdxs + np.tensordot(dgndxc,d2xcd2xs, axes=[0,0])



        else:
            ihp = int(hook[0])
            hpatch = self.surf.patches[ihp] if ihp!=self.iquad else self
            Ne = len(hpatch.squad)
            dgndu    = np.zeros(  3*(Ne+1) )
            d2gnd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

            dgtdu    = np.zeros(  3*(Ne+1) )
            d2gtd2u  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )
            t0 = hook[1:3]
            # if np.allclose(t0-t,0.0): return self.KC_tot2(xs,kn,kt,mu,hook=None, cubicT=cubicT), None, None
            gn0 = hook[3]
            xc0= hpatch.Grg(t0)
            n0 = hpatch.D3Grg(t0)
            gn = (xs-xc0)@n0
            dgndxs = n0
            dgndxc0 = -n0
            dgndn0 = xs-xc0
            d2gndn0dxs = np.eye(3)
            d2gndn0dxc0=-np.eye(3)
            xsp = xc0 + np.dot(xs-xc0,n0)*n0
            gt = norm(xs-xsp)
            if gt == 0 : return self.KC_tot3(xs,kn,kt,mu,hook=None,Stick=True, cubicT=cubicT, output=None, niter=3, new=new), None,None
            dgtdxs  = (xs-xsp)/norm(xs-xsp)
            dgtdxsp = -dgtdxs
            d2gtd2xs = np.eye(3)/gt - np.outer((xs-xsp),(xs-xsp))/gt**3
            d2gtd2xsp = d2gtd2xs
            d2gtdxsdxsp = -d2gtd2xs
            dxspdxs = np.outer(n0,n0)
            dxspdxc0= np.eye(3)-dxspdxs
            dxspdn0 = np.outer(n0,xs-xc0)+np.dot(xs-xc0,n0)*np.eye(3)   #TODO: check order of terms for outer product
            # to check ...
            d2xspd2n0   = (np.multiply.outer(np.eye(3),xs-xc0)+np.multiply.outer(xs-xc0,np.eye(3))).swapaxes(0,1)
            d2xspdn0dxs = np.multiply.outer(np.eye(3),n0)+np.multiply.outer(n0,np.eye(3))
            d2xspdxsdn0  = d2xspdn0dxs.swapaxes(1,2)            
            d2xspdxc0dn0= -d2xspdxsdn0
            d2xspdn0dxc0= -d2xspdn0dxs
            
            Dxc0Dxi = hpatch.dxcdxi(t0)
            Dn0Dxi,D2n0DxiDxj = hpatch.dndxi(t0, degree=2)
            dgtdu[:3] += dgtdxs + dgtdxsp@dxspdxs
            d2gtd2u[:3,:3] += d2gtd2xs + 2*d2gtdxsdxsp@dxspdxs + (d2gtd2xsp@dxspdxs).T@dxspdxs
            dgndu[:3] += dgndxs


        #########################
        ## fintC & KC assembly ##
        #########################

        # fintC,KC     Slave-Slave :

        DxiDu   = np.empty((20,3,3*(Ne+1))  ,dtype=float)        # will be used in nested loop below
        if hook is not None: DxiDuh  = np.empty((20,3,3*(Ne+1))  ,dtype=float)        # will be used in nested loop below
        for idx in range(1,21):
            if hook is None:
                dxcdxi = DxcDxi[idx-1]*np.eye(3)
                d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
                d3xcdxid2t = np.array([[ D3xcDxiD2t[idx-1,0,0]*np.eye(3) , D3xcDxiD2t[idx-1,0,1]*np.eye(3) ],
                                    [ D3xcDxiD2t[idx-1,1,0]*np.eye(3) , D3xcDxiD2t[idx-1,1,1]*np.eye(3) ]]).swapaxes(0,2).swapaxes(1,3)

                dfdxi = 2*((dxcdxi.T@dxcdt) - np.tensordot(xs-xc,d2xcdxidt,axes=[0,0])).T
                dtdxi = -invdfdt@dfdxi          # torch-verified

                DtDxi[idx-1] = dtdxi

                d2fdxidt  = 2*(np.tensordot(dxcdxi,d2xcd2t,axes=[0,0]) + np.tensordot(d2xcdxidt,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcdxidt,axes=[0,0]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcdxid2t,axes=[0,0])).swapaxes(0,1)
                D2fDxiDt[idx-1] = d2fdxidt
                d2fdxidxs = -2*d2xcdxidt.swapaxes(0,2)
                d2tdxidxs = np.tensordot(-invdfdt,  ((d2fdt2@dtdxi).swapaxes(1,2)+d2fdxidt)@dtdxs + (d2fdxsdt@dtdxi).swapaxes(1,2) + d2fdxidxs , axes = [1,0])
                d2xcdxidxs = np.tensordot(dxcdt,d2tdxidxs,axes=[1,0]) + ((d2xcd2t@dtdxi).swapaxes(1,2)+d2xcdxidt)@dtdxs

                dxcdxi += dxcdt@dtdxi # checking here <====
                DxcDxi_tot[idx-1] = dxcdxi
            else:
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi = Dn0Dxi[idx-1]


            dxidu, d2xid2u = self.dxidu(idx,order=2)
            DxiDu[idx-1] = dxidu

            # set_trace()

            # fintC:i   ,   KC: s-i  i-s  i-i
            if hook is None:
                dgndu += dgndxc @ dxcdxi @ dxidu
                d2gnd2u[:3,:] += ((d2gndxsdxc + (d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T)@dxidu
                d2gnd2u[ : , :3] += (((d2gndxsdxc+(d2gnd2xc@dxcdxs).T)@dxcdxi + np.tensordot(dgndxc,d2xcdxidxs,axes=[0,0]).T) @ dxidu).T
                d2gnd2u += np.tensordot(dgndxc@dxcdxi,d2xid2u,axes=[0,0]) if type(d2xid2u)!=int else 0
            else:
                dxiduh, d2xid2uh = (dxidu, d2xid2u) if ihp==self.iquad else hpatch.dxidu(idx,order=2)
                DxiDuh[idx-1] = dxiduh

                dgndu += (dgndxc0@dxc0dxi + dgndn0@dn0dxi )@ dxiduh
                s_i = ( d2gndn0dxs@dn0dxi )@dxiduh
                d2gnd2u[:3 , : ] += s_i
                d2gnd2u[ : , :3] += s_i.T
                i_i = np.tensordot(dgndxc0@dxc0dxi + dgndn0@dn0dxi,d2xid2uh,axes=[0,0]) if type(d2xid2uh)!=int else 0
                d2gnd2u += i_i


                dxspdxi = dxspdxc0@dxc0dxi+dxspdn0@dn0dxi
                dgtdu += dgtdxsp@dxspdxi@dxiduh

                s_i = ((d2gtdxsdxsp+(d2gtd2xsp@dxspdxs).T)@dxspdxi + dgtdxsp@np.tensordot(d2xspdxsdn0,dn0dxi,axes=[2,0]))@dxiduh

                d2gtd2u[ :3, : ] += s_i
                d2gtd2u[ : , :3] += s_i.T
                i_i= np.tensordot(dgtdxsp@dxspdxi,d2xid2uh,axes=[0,0]) if type(d2xid2uh)!=int else 0
                d2gtd2u += i_i


        # K     CtrlPts-CtrlPts (2):
        for idx in range(1,21):
            if hook is None:
                dxidu = DxiDu[idx-1]
                dxcdxi = DxcDxi_tot[idx-1]
                d2xcdxidt = np.array([D2xcDxiDt[idx-1,0]*np.eye(3),D2xcDxiDt[idx-1,1]*np.eye(3)]).swapaxes(0,2)
                dtdxi = DtDxi[idx-1]
                d2fdxidt = D2fDxiDt[idx-1]
            else:
                dxiduh = DxiDuh[idx-1]
                dxc0dxi = Dxc0Dxi[idx-1]*np.eye(3)
                dn0dxi = Dn0Dxi[idx-1]
                dxspdxi = dxspdxc0@dxc0dxi + dxspdn0@dn0dxi

            for jdx in range(1,21):
                if hook is None:
                    dxjdu = DxiDu[jdx-1]
                    dxcdxj = DxcDxi_tot[jdx-1]
                    d2xcdxjdt = np.array([D2xcDxiDt[jdx-1,0]*np.eye(3),D2xcDxiDt[jdx-1,1]*np.eye(3)]).swapaxes(0,2)
                    dtdxj = DtDxi[jdx-1]
                    d2fdxjdt = D2fDxiDt[jdx-1]

                    d2fdxidxj = 2*(np.tensordot(dxcdxi,d2xcdxjdt, axes = [0,0]).swapaxes(0,1) + np.tensordot(dxcdxj,d2xcdxidt,axes=[0,0])).swapaxes(0,2)
                    d2tdxidxj = np.tensordot(-invdfdt , (((d2fdt2@dtdxj).swapaxes(1,2)+d2fdxjdt)@dtdxi).swapaxes(1,2) + d2fdxidxj + d2fdxidt@dtdxj , axes=[1,0])

                    d2xcdxidxj = d2xcdxidt@dtdxj + ((d2xcdxjdt+(d2xcd2t@dtdxj).swapaxes(1,2))@dtdxi).swapaxes(1,2) + np.tensordot(dxcdt,d2tdxidxj,axes=[1,0])
                    d2gndxidxj = (d2gnd2xc@dxcdxi).T@dxcdxj + np.tensordot(dgndxc,d2xcdxidxj,axes=[0,0])

                    d2gnd2u += (d2gndxidxj@dxjdu).T@dxidu

                else:
                    # d2gnd2u += (d2gndxidxj@dxjdu).T@dxidu

                    dxjduh = DxiDuh[jdx-1]
                    dxc0dxj = Dxc0Dxi[jdx-1]*np.eye(3)
                    dn0dxj = Dn0Dxi[jdx-1]
                    dxspdxj = dxspdxc0@dxc0dxj + dxspdn0@dn0dxj
                    d2n0dxidxj = D2n0DxiDxj[:,:,:,idx-1,jdx-1]
                    d2xspdxidxj = np.tensordot(np.tensordot(d2xspdxc0dn0,dxc0dxi,axes=[1,0]),dn0dxj,axes=[1,0]) + np.tensordot(d2xspdn0dxc0@dxc0dxj+d2xspd2n0@dn0dxj,dn0dxi,axes=[1,0]).swapaxes(0,1) + np.tensordot(dxspdn0,d2n0dxidxj,axes=[1,0])
                    d2gtdxidxj = np.tensordot(d2gtd2xsp@dxspdxi,dxspdxj,axes=[0,0]) + np.tensordot(dgtdxsp,d2xspdxidxj,axes=[0,0])
                    d2gtd2u += (d2gtdxidxj.T@dxiduh).T@dxjduh
                    d2gnd2u += (((d2gndn0dxc0@dxc0dxi).T@dn0dxj + (d2gndn0dxc0.T@dn0dxi).T@dxc0dxj + dgndn0@d2n0dxidxj).T@dxiduh).T@dxjduh

        # if (niter>=3) and (hook is not None): set_trace()
        if type(output) is str:
            return locals()[output]

        if cubicT is None:
            KCN = kn*(np.outer(dgndu,dgndu)+gn*d2gnd2u)
        elif gn<cubicT:
            KCN = kn*(np.outer(dgndu,dgndu)+(gn-cubicT/2)*d2gnd2u)
        else:
            KCN = kn*((gn/cubicT)*np.outer(dgndu,dgndu)+(gn**2/(2*cubicT))*d2gnd2u)
        # new=True
        cond = (niter<2 and new) or (niter==0)
        printif(cond, "ELASTIC IMPOSED")
        
        if hook is None:
            return KCN
        # elif Stick:
        # elif kt*gt<-mu*kn*gn:
        elif (kt*gt<-mu*kn*gn or cond):
            print("elastic KC")
            # print("NITER: ", niter)
            KCT = kt*(np.outer(dgtdu,dgtdu)+gt*d2gtd2u)
            KCNT = None
        else:
            print("plastic KC")
            KCT = -mu*kn*gn*d2gtd2u
            # dgNdu = np.zeros(3*(Ne+1))
            # # dgNdu[:3]=dgndxs + dgndxc@dxcdxs
            # dgNdu[:3]=dgndxs
            # KCNT =-mu*kn*np.outer(dgtdu,dgNdu) 
            KCNT =-mu*kn*np.outer(dgtdu,dgndu) 

        return KCN,KCT,KCNT


    def get_Dgn_for_rigids(self, gn, t):       #Chain rule by hand

        xc, dxcdt, d2xcd2t, d3xcd3t= self.Grg(t, deriv = 3)
        normal = self.D3Grg(t)
        xs = xc + gn*normal

        # opa = 0
        # if not (0-opa<=t[0]<=1+opa and 0-opa<=t[1]<=1+opa):
        #     print("the point does not belong in the patch!!!!")
        #     set_trace()



        dgdxc = (xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
        dgdxs = -dgdxc
        # f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
        dfdt =  -2*np.tensordot(xs-xc,d2xcd2t,axes=1) + 2*(dxcdt.T @ dxcdt)
        dfdxs = -2*dxcdt.T
        dtdxs = np.linalg.solve(-dfdt,dfdxs)
        dxcdxs = dxcdt @ dtdxs


        # Vars for K---------------------------------------------------
        d2gd2xs = np.eye(3)/gn - np.outer((xs-xc),(xs-xc))/gn**3         
        d2gd2xc = d2gd2xs
        d2gdxsdxc = -d2gd2xs

        # d/du(dxcdxs)
        d2fdt2   = 2*(np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]) + np.tensordot(d2xcd2t,dxcdt,axes=[0,0]) + np.tensordot(dxcdt,d2xcd2t,axes=[[0],[0]]).swapaxes(0,1) - np.tensordot(xs-xc,d3xcd3t,axes=[[0],[0]]) )
        invdfdt = np.linalg.inv(dfdt)

        d2fdxsdt = -2*d2xcd2t.swapaxes(0,1)
        d2td2xs = np.tensordot(-invdfdt,(np.tensordot(np.tensordot(dtdxs,d2fdt2,axes=[0,1]),dtdxs,axes=[2,0]).swapaxes(0,1)  + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]) + np.tensordot(d2fdxsdt,dtdxs,axes=[2,0]).swapaxes(1,2)),axes = [[1],[0]])
        # d2xcd2xs = np.tensordot( np.tensordot(dtdxs,d2xcd2t,axes=[0,2]),dtdxs,axes=[2,0]).swapaxes(0,1) + np.tensordot(dxcdt,d2td2xs,axes=[1,0])


        # dgdu = dgdxs + dgdxc @ dxcdxs     # the last product is always zero
        # dgdu = dgdxs
        # d2gd2u =  d2gd2xs + 2*d2gdxsdxc@dxcdxs + (d2gd2xc@dxcdxs).T@dxcdxs + np.tensordot(dgdxc,d2xcd2xs, axes=[0,0])

        # result = np.array([xs[0],xs[1],xs[2],t[0],t[1],gn,
        #                    dgdu[0],dgdu[1],dgdu[2],
        #                    d2gd2u[0,0],d2gd2u[1,1],d2gd2u[2,2],
        #                    d2gd2u[0,1],d2gd2u[1,2],d2gd2u[0,2]])


        result = np.array([xs[0],xs[1],xs[2],self.iquad,t[0],t[1],gn])



        return result

    # Plot options
    def plot(self,axis, color = "blue", ref=10, surf=True,wire=False, label = False, Edges=True):
        x = np.zeros((ref+1,ref+1),dtype = float)
        y = np.zeros((ref+1,ref+1),dtype = float)
        z = np.zeros((ref+1,ref+1),dtype = float)

        if Edges:
            e1,e2,e3,e4 = np.zeros((4,ref+1,3),dtype=float)


        for i in range(ref+1):
            ti = i/(ref)

            for j in range(ref+1):
                tj = j/(ref)
                point = self.Grg((ti,tj))
                x[i,j], y[i,j], z[i,j] = point

                if Edges and i==0:
                    e3[j] = point
                elif Edges and i==ref:
                    e4[j] = point

                if Edges and j==0:
                    e1[i] = point
                elif Edges and j==ref:
                    e2[i] = point

        if surf:
            plotObj = axis.plot_surface(x, y, z, color=color,edgecolors=None)
        if wire:
            axis.plot_wireframe(x, y, z, color="black")

        label=True
        if label:
            # xm,ym,zm =(np.mean(x[np.ix_([0,-1])][:,np.ix_([0,-1])]),
            #            np.mean(y[np.ix_([0,-1])][:,np.ix_([0,-1])]),
            #            np.mean(z[np.ix_([0,-1])][:,np.ix_([0,-1])]))
            xm,ym,zm = self.Grg((0.5,0.5))
            text_orientation, _ = self.Grg((0.5,0.5), deriv = 1)[1].T
            axis.text(xm,ym,zm, str(self.iquad),text_orientation)

        # if self.iquad==34:
        #     self.BS.plot(axis,ref=100)


        if Edges:
            cline=list(color)
            # cline=[0.1,0.1,0.1,1.0]
            cline[-1]=1.0
            axis.plot(e1[:,0],e1[:,1],e1[:,2],color=cline,linewidth=0.2)
            axis.plot(e2[:,0],e2[:,1],e2[:,2],color=cline,linewidth=0.2)
            axis.plot(e3[:,0],e3[:,1],e3[:,2],color=cline,linewidth=0.2)
            axis.plot(e4[:,0],e4[:,1],e4[:,2],color=cline,linewidth=0.2)
        return plotObj
        
    def plotCtrlPtsLines(self, axis):
        cps = self.flatCtrlPts()
        for i in range(4):
            y0 = cps[i]
            y1 = cps[2*(i+2)]
            y2 = cps[2*(i+2)+1]
            y3 = cps[[1,2,3,0][i]]

            Xline = np.array([y0,y1,y2,y3])

            axis.plot(Xline[:,0],Xline[:,1],Xline[:,2], color="black", lw = 0.2)

            squares = [[4,12,14,5],[6,15,17,7],[8,16,18,9],[10,19,13,11]]

            for squa in squares:
                p1, p2, p3, p4 = cps[squa[0]],cps[squa[1]],cps[squa[2]],cps[squa[3]]
                Xline = np.array([p1,p2,p3,p4])
                axis.plot(Xline[:,0],Xline[:,1],Xline[:,2], color="black", lw = 0.2)

    def plotIsolate(self, xs = None, ForcesAt = None):
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim3d((0, 4))
        ax.set_ylim3d((0, 4))
        ax.set_zlim3d((1, 4))

        self.plot(ax, color =(0,0,1, 0.75))

        if type(xs) == np.ndarray:
            Ne = len(self.squad)
            fintC = self.fintC(xs,1)[0]

            if type(fintC) == float:
                print("No normal penetration found")
                return None

            ax.scatter(xs[0], xs[1], xs[2], s = 20, c="red")

            if ForcesAt is not None:
                fintC = fintC.reshape(Ne+1,3)

            if ForcesAt == "nodes":
                for vert in range(4):
                    for node in self.sorrounding_nodes[vert]:
                        Xn = self.surf.X[node]
                        ax.scatter(Xn[0], Xn[1], Xn[2], s = 3, c = 'blue')
                Xnodes = self.surf.X[self.squad]

                for fc, X in zip(fintC[1:],Xnodes):
                    ax.quiver(X[0],X[1],X[2],fc[0],fc[1],fc[2])

                ax.quiver(xs[0], xs[1], xs[2],fintC[0][0],fintC[0][1],fintC[0][2])
            
            elif ForcesAt == "CtrlPts":
                t1,t2 = self.findProjection(xs)
                xc = self.Grg(t1,t2)
                normal = self.D3Grg(t1,t2)
                N_tild = self.N_tild(t1,t2,normal)
                gn = (xs-xc) @ normal             # np.dot(a,b) <=> a @ b
                t_n = 1*gn
                r_tild = -t_n*N_tild   # forces conjugated to slave+CtrlPts

                self.plotCtrlPtsLines(ax)

                for fcp, X in zip(r_tild[1:],self.flatCtrlPts()):
                    ax.scatter(X[0],X[1],X[2], s = 2, c = "red")
                    ax.quiver(X[0],X[1],X[2],fcp[0],fcp[1],fcp[2], color = "green")




        plt.show()


    # Torch tool for Autograd
    def torchC(self,xs,u,kn,kt,mu,hook=None,Stick=True, cubicT=None, thing='f', seeding=10, case=0, tracing= False):
        import torch

        from scipy.special import comb, factorial
        def Bernstein(n,k,x):
            return comb(n,k)*(x**k)*((1-x)**(n-k))

        def dnBernstein(n,k,x,p):
            coef = factorial(n)/factorial(n-p)
            desde = max(0,k+p-n)
            hasta = min(k,p)
            dnB = coef*sum([(-1)**(i+p)*comb(p,i)*Bernstein(n-p,k-i,x) for i in range(desde,hasta+1)])
            return dnB

        def Grg(CtrlPts,u,v, eps = 1e-8):                      # with treatment for undefinition at nodes
            p = torch.tensor([0.0 , 0.0 , 0.0],dtype = torch.float64)

            if type(CtrlPts) is not list:
                cp = CtrlPts
                row0 = [ cp[0] ,      cp[11]     ,       cp[10]    , cp[3] ]            #This "matrix" HAS to be transposed (for indexing purposes)
                row1 = [ cp[4] , [cp[12],cp[13]] , [cp[18],cp[19]] , cp[9] ]
                row2 = [ cp[5] , [cp[14],cp[15]] , [cp[16],cp[17]] , cp[8] ]
                row3 = [ cp[1] ,       cp[6]     ,       cp[7]     , cp[2] ]
                
                CtrlPts = [row0,row1,row2,row3]

            n, m = len(CtrlPts)-1, len(CtrlPts[0])-1
            for i  in range(n+1):
                for j in range(m+1):
                    if i in [1,2] and j in [1,2]:
                        if i==1 and j ==1:
                            x110, x111 = CtrlPts[1][1]
                            den = max(eps,u+v)  
                            xij = (u*x110+v*x111)/(den)
                        elif i==1 and j==2:
                            x120, x121 = CtrlPts[1][2]
                            den = max(eps,u+1-v)
                            xij = (u*x120+(1-v)*x121)/(den)

                        elif i==2 and j==1:
                            x210, x211 = CtrlPts[2][1]
                            den = max(eps, v+1-u)
                            xij = ((1-u)*x210+v*x211)/(den)
                        else:
                            x220, x221 = CtrlPts[2][2]
                            den = max(eps, 2-u-v)
                            xij = ((1-u)*x220+(1-v)*x221)/(den)
                            
                    else:
                        xij = CtrlPts[i][j]

                    Bi   =  Bernstein(n, i, u)
                    Bj   =  Bernstein(m, j, v)
                    p += Bi*Bj*xij
            return p

        def DtGrg(CtrlPts,u,v, order = 1 , eps = 1e-8):                        # Normalized normal vector at (u,v) with treatment for undefinition at nodes
            D1p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D1p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D2D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D1D1p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D1D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D2D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D2D2D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            n, m = len(CtrlPts)-1, len(CtrlPts[0])-1

            for i  in range(n+1):
                for j in range(m+1):
                    # Inner nodes: values, derivatives and treatments
                    if i in [1,2] and j in [1,2]:
                        if i==1 and j ==1:
                            x110, x111 = CtrlPts[1][1]
                            den = max(eps,u+v)  
                            xij = (u*x110+v*x111)/(den)
                            D1xij = x110/(den) - (u*x110 + v*x111)/((den)**2)
                            D2xij = x111/(den) - (u*x110 + v*x111)/((den)**2)
                            if order > 1:
                                D11xij =-2*v*(x110 - x111)/(den)**3
                                D12xij = (u - v)*(x110 - x111)/(den)**3
                                D22xij = 2*u*(x110 - x111)/(den)**3
                            if order > 2:
                                D111xij = 6*v*(x110-x111)/(den)**4
                                D112xij = -(2*(u-2*v)*(x110-x111)/(den)**4)
                                D122xij = -(2*(2*u-v)*(x110-x111)/(den)**4)
                                D222xij = -(6*u*(x110-x111)/(den)**4)
                                
                        elif i==1 and j==2:
                            x120, x121 = CtrlPts[1][2]
                            den = max(eps,u+1-v)
                            xij = (u*x120+(1-v)*x121)/(den)
                            D1xij = x120/(den) - (u*x120 + (1-v)*x121)/((den)**2)
                            D2xij = -x121/(den) + (u*x120 + (1-v)*x121)/((den)**2)
                            if order > 1:
                                D11xij = (2*(-1 + v)*(x120 - x121))/(den)**3
                                D12xij = -(((-1 + u + v)*(x120 - x121))/(den)**3)
                                D22xij = (2*u*(x120 - x121))/(den)**3
                            if order > 2:
                                D111xij = -(6*(-1+v)*(x120-x121)/(den)**4)
                                D112xij = (2*(-2+u+2*v)*(x120-x121)/(den)**4)
                                D122xij = -(2*(-1+2*u+v)*(x120-x121)/(den)**4)
                                D222xij = (6*u*(x120-x121)/(den)**4)

                        elif i==2 and j==1:
                            x210, x211 = CtrlPts[2][1]
                            den = max(eps, v+1-u)
                            xij = ((1-u)*x210+v*x211)/(den)
                            D1xij = -x210/(den) + ((1-u)*x210 + v*x211)/((den)**2)
                            D2xij = x211/(den) - ((1-u)*x210 + v*x211)/((den)**2)
                            if order > 1:
                                D11xij = (2*v*(x210 - x211))/(-den)**3
                                D12xij = ((-1 + u + v)*(x210 - x211))/(den)**3
                                D22xij = (2*(-1 + u)*(x210 - x211))/(-den)*3
                            if order > 2:
                                D111xij = -(6*v*(x210-x211)/(den)**4)
                                D112xij = (2*(-1+u+2*v)*(x210-x211)/(den)**4)
                                D122xij = -(2*(-2+2*u+v)*(x210-x211)/(den)**4)
                                D222xij = (6*(-1+u)*(x210-x211)/(den)**4)

                        else:
                            x220, x221 = CtrlPts[2][2]
                            den = max(eps, 2-u-v)
                            xij = ((1-u)*x220+(1-v)*x221)/(den)
                            D1xij = -x220/(den) + ((1-u)*x220 + (1-v)*x221)/((den)**2)
                            D2xij = -x221/(den) + ((1-u)*x220 + (1-v)*x221)/((den)**2)
                            if order > 1:
                                D11xij = -((2*(-1 + v)*(x220 - x221))/(-den)**3)
                                D12xij = ((u - v)*(x220 - x221))/(-den)**3
                                D22xij = (2*(-1 + u)*(x220 - x221))/(-den)**3
                            if order > 2:
                                D111xij = 6*(-1+v)*(x220-x221)/(-den)**4
                                D112xij = -(2*(1+u-2*v)*(x220-x221)/(-den)**4)
                                D112xij = -(2*(-1+2*u-v)*(x220-x221)/(-den)**4)
                                D222xij = -(6*(-1+u)*(x220-x221)/(-den)**4)

                    else:
                        xij = CtrlPts[i][j]
                        D1xij, D2xij = 0.0 , 0.0
                        D11xij, D12xij, D22xij = 0.0 , 0.0 , 0.0
                        D111xij, D112xij, D122xij, D222xij = 0.0 , 0.0 , 0.0 , 0.0

                    # Bernstein polynomials
                    Bi     =   Bernstein(n, i, u)
                    Bj     =   Bernstein(m, j, v)
                    D1Bi     =  dnBernstein(n, i, u, 1)
                    D2Bj     =  dnBernstein(m, j, v, 1)
                    if order > 1:
                        DD1Bi = dnBernstein(n, i, u, 2)
                        DD2Bj = dnBernstein(m, j, v, 2)

                    if order > 2:
                        DDD1Bi = dnBernstein(n, i, u, 3)
                        DDD2Bj = dnBernstein(m, j, v, 3)

                    #TODO: Unify Grg and D3Grg: Just add "p" bellow to be computed as p+=...

                    # Tangent Derivatives w/r to LOCAL parameters
                    D1p += D1Bi*Bj*xij + Bi*Bj*D1xij
                    D2p += Bi*D2Bj*xij + Bi*Bj*D2xij

                    if order > 1:
                        D1D1p += (DD1Bi*xij + 2*D1Bi*D1xij + Bi*D11xij)*Bj
                        D1D2p += D1Bi*D2Bj*xij + D1Bi*Bj*D2xij + Bi*D2Bj*D1xij + Bi*Bj*D12xij
                        D2D2p += (DD2Bj*xij + 2*D2Bj*D2xij + Bj*D22xij)*Bi

                    if order > 2:
                        D1D1D1p += (DDD1Bi*xij + 3*DD1Bi*D1xij + 3*D1Bi*D11xij + Bi*D111xij)*Bj
                        D1D1D2p += (DD1Bi*D2xij + 2*D1Bi*D12xij + Bi*D112xij)*Bj + (DD1Bi*xij + 2*D1Bi*D1xij + Bi*D11xij)*D2Bj
                        D1D2D2p += (DD2Bj*D1xij + 2*D2Bj*D12xij + Bj*D122xij)*Bi + (DD2Bj*xij + 2*D2Bj*D2xij + Bj*D22xij)*D1Bi
                        D2D2D2p += (DDD2Bj*xij + 3*DD2Bj*D2xij + 3*D2Bj*D22xij + Bj*D222xij)*Bi

            if order==1:
                return torch.transpose(torch.stack([D1p, D2p]),0,1)
            elif order == 2:
                return torch.transpose(torch.stack([D1p, D2p]),0,1), torch.transpose(torch.stack([torch.stack([D1D1p, D1D2p]),
                                                    torch.stack( [D1D2p, D2D2p])    ]),0,1)
            elif order == 3:
                return torch.transpose(torch.stack([D1p, D2p]),0,1), torch.transpose(torch.stack([[D1D1p, D1D2p], [D1D2p, D2D2p]]),0,1), torch.transpose(torch.stack([[[D1D1D1p,D1D1D2p],[D1D1D2p,D1D2D2p]], [[D1D1D2p,D1D2D2p],[D1D2D2p,D2D2D2p]]]),0,1)

        def D3Grg(CtrlPts, u,v, normalize = True):                        # Normalized normal vector at (u,v) with treatment for undefinition at nodes
            D1p, D2p = DtGrg(CtrlPts,u,v).T
            D3p = torch.cross(D1p,D2p)
            if normalize:
                D3p = D3p/torch.norm(D3p)       # The NORMALIZED vectors are continuous from patch to patch (doesnt make sense otherwise)
            return D3p

        def MinDist(CtrlPts, xs, seeding = seeding):
            umin = 0
            vmin = 0
            dmin = torch.norm(xs-CtrlPts[0][0])   # Starting point

            for u in torch.linspace(0,1,seeding+1):
                for v in torch.linspace(0,1,seeding+1):
                    d = torch.norm(xs - Grg(CtrlPts,u,v))
                    if d < dmin:
                        dmin, umin, vmin = d, u, v

            return umin , vmin

        def findProjection(CtrlPts,xs, seeding=seeding):

            t = torch.tensor(MinDist(CtrlPts, xs, seeding=seeding))

            tol = 1e-15
            res = 1+tol
            count = 0
            while res>tol and (0<=t[0]<=1 and 0<=t[1]<=1) and count<100:

                xc = Grg(CtrlPts, t[0],t[1])

                dxcdt, d2xcd2t = DtGrg(CtrlPts,t[0],t[1], order = 2)


                f = -2*torch.matmul((xs-xc),dxcdt)                        # Unnecessary (?)
                K =  2*(torch.tensordot(-( xs-xc),d2xcd2t,dims = ([0],[2])) + torch.tensordot(dxcdt,dxcdt,dims= ([0],[0]) ) )


                dt = torch.linalg.solve(-K,f)

                t = t + dt

                res = torch.norm(f)

                count += 1
            return t

        def jacobian(y, x, create_graph=False):                                                            
            jac = []                                                                                       
            flat_y = y.reshape(-1)                                                                         
            grad_y = torch.zeros_like(flat_y)                                                                 
            for i in range(len(flat_y)):                                                                   
                grad_y[i] = 1.                                                                             
                grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph,allow_unused=True)
                jac.append(grad_x.reshape(x.shape))                                                        
                grad_y[i] = 0.                                                         

            return torch.stack(jac).reshape(y.shape + x.shape)        

        def hessian(y, x):                                                                                 
            return jacobian(jacobian(y, x, create_graph=True), x)                                          

        # Slave node
        Xs = torch.tensor(xs,dtype=torch.float64, requires_grad=False)
        us = torch.zeros(3)
        # Master nodes
        Xm = torch.tensor(self.X,dtype=torch.float64, requires_grad = False)
        um = torch.tensor(u[self.surf.body.DoFs[self.squad]])

        # Hook variables
        t0 = hook[1:3]
        gn0 = hook[3]

        # Displacements vector
        t0 = torch.tensor(t0,requires_grad=False)
        u =  torch.row_stack((us,um)).flatten()
        u.requires_grad = True

        x = torch.row_stack((Xs,Xm))+u.reshape(-1,3)
        xs = x[0,:]
        xm = x[1:,:]

        #Computation of normals
        facets = self.facets
        normals = torch.empty((4,3),dtype=torch.float64)
        for vert_idx, vert in enumerate(facets):
            Xi, xi = Xm[vert_idx], xm[vert_idx]
            ntmp = 0
            n0mp = 0
            ntm = 0

            for facet in vert:
                Xna, Xnb = [Xm[i] for i in facet]
                xna, xnb = [xm[i] for i in facet]

                Xa, Xb = (Xna)-(Xi) , (Xnb)-(Xi)
                xa, xb = (xna)-(xi) , (xnb)-(xi)

                ntm_e = torch.cross(xb,xa)
                n0m_e = torch.cross(Xb,Xa)
                w0m_e = torch.norm(n0m_e)

                ntmp += ntm_e / w0m_e
                n0mp += n0m_e / w0m_e

            wm = torch.norm( n0mp )
            
            ntm = ntmp / wm
            normals[vert_idx] = ntm

        # Computation of CtrlPts
        start = torch.empty( (4,3)  , dtype=torch.float64)
        edge  = torch.empty( (4,2,3), dtype=torch.float64)
        inner = torch.empty( (4,2,3), dtype=torch.float64)

        # going through every edge of the patch to find edge points
        for idx in range(4):
            y0, y3 = xm[range(4)[idx-1]], xm[idx]
            m0, m3 = normals[idx-1] , normals[idx]

            c0 = 1/3*((y3-y0)-torch.matmul((y3-y0),torch.outer(m0,m0)/torch.dot(m0,m0)))
            c2 = 1/3*((y3-y0)-torch.matmul((y3-y0),torch.outer(m3,m3)/torch.dot(m3,m3)))
            y1, y2 = y0+c0, y3-c2

            start[idx] = y0
            edge[idx,0,:] = y1
            edge[idx,1,:] = y2


        # going through every edge of the patch to find inner points
        for idx in range(4):
            y0, y3 = xm[range(4)[idx-1]] , xm[idx]
            m0, m3 = normals[idx-1] , normals[idx]
            
            y1, y2 = edge[idx][0], edge[idx][1]
            y0p , y3p = edge[idx-1][1], edge[idx-3][0]

            c0 = y1-y0
            c1 = y2-y1
            c2 = y3-y2
            b0, b3 = y0p-y0 , y3p-y3

            a0 = torch.cross(m0,(y3-y0))/torch.norm( torch.cross(m0,(y3-y0)) )
            a3 = torch.cross(m3,(y3-y0))/torch.norm( torch.cross(m3,(y3-y0)) )

            k0, k1 = torch.dot(a0,b0), torch.dot(a3,b3)
            h0, h1 = torch.dot(c0,b0)/torch.dot(c0,c0) , torch.dot(c2,b3)/torch.dot(c2,c2)

            b1 = 1/3*((k1+k0)*a0+k0*a3+2*h0*c1+h1*c0)
            b2 = 1/3*((k1+k0)*a3+k1*a0+2*h1*c1+h0*c2)

            y1p, y2p = y1+b1 , y2+b2

            inner[idx,0,:] = y1p
            inner[idx,1,:] = y2p

        # Assembly of CtrlPts matrix of the patch
        row0 = [ start[1]  ,         edge[0][1]        ,         edge[0][0]        ,  start[0]  ]     #This "matrix" HAS to be transposed (for indexing purposes)
        row1 = [edge[1][0] , [inner[1][0],inner[0][1]] , [inner[3][1],inner[0][0]] , edge[3][1] ]
        row2 = [edge[1][1] , [inner[1][1],inner[2][0]] , [inner[3][0],inner[2][1]] , edge[3][0] ]
        row3 = [ start[2]  ,         edge[2][0]        ,         edge[2][1]        ,  start[3]  ]

        # cp = [row0,row1,row2,row3]
        # CtrlPts_flat = torch.stack([
        #     cp[0][0], cp[3][0], cp[3][3], cp[0][3],
        #     cp[1][0], cp[2][0], cp[3][1], cp[3][2],cp[2][3], cp[1][3], cp[0][2], cp[0][1],
        #     cp[1][1][0], cp[1][1][1], cp[2][1][0], cp[2][1][1],cp[2][2][0], cp[2][2][1], cp[1][2][0], cp[1][2][1],
        # ])
        # cp = CtrlPts_flat
        # row0 = [ cp[0] ,      cp[11]     ,       cp[10]    , cp[3] ]            #This "matrix" HAS to be transposed (for indexing purposes)
        # row1 = [ cp[4] , [cp[12],cp[13]] , [cp[18],cp[19]] , cp[9] ]
        # row2 = [ cp[5] , [cp[14],cp[15]] , [cp[16],cp[17]] , cp[8] ]
        # row3 = [ cp[1] ,       cp[6]     ,       cp[7]     , cp[2] ]


        CtrlPts = [row0,row1,row2,row3]

        t = findProjection(CtrlPts,xs)

        xc = Grg(CtrlPts , t[0], t[1])
        normal = D3Grg(CtrlPts, t[0], t[1])
        gn = torch.dot( xs-xc , normal )

        #fritcion part
        gn0 = torch.tensor(gn0,dtype=torch.float64,requires_grad=False)
        tau1, tau2 = DtGrg(CtrlPts,t0[0],t0[1]).T
        N = torch.cross(tau1,tau2)
        normN = torch.norm(N)
        n0 = N/normN
        
        n0 = D3Grg(CtrlPts, t0[0], t0[1])
        xc0 = Grg(CtrlPts,t0[0],t0[1])
        # gn0 = torch.tensor(gn0,dtype=torch.float64,requires_grad=False)
        if case==0:
            xs0 = xc0 - gn0*n0
            gt = torch.norm(xs0-xs)
        elif case==1:
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            gt = torch.norm(xs-xsp)

        force = kt*gt*jacobian(gt,u,create_graph=True) if kt*gt<-mu*kn*gn else -mu*kn*gn*jacobian(gt,u,create_graph=True)


        def xsp0_func(xs,xc0,n0, i=0):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[i]

        def xsp0_funcxc0(xc0):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[0]
        def xsp0_funcxs(xs):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[0]
        def xsp0_funcn0(n0):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[0]
        def xsp1_funcxc0(xc0):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[1]
        def xsp1_funcxs(xs):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[1]
        def xsp1_funcn0(n0):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[1]
        def xsp2_funcxc0(xc0):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[2]
        def xsp2_funcxs(xs):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[2]
        def xsp2_funcn0(n0):
            xsp = xc0+torch.dot(xs-xc0,n0)*n0
            return xsp[2]

        def dxspdn0_00_funcxs(xs):
            return torch.autograd.functional.jacobian(xsp0_funcn0,n0,create_graph=True)[0]





        def gt_func(xs):
            xsp = xc0 + torch.dot(xs-xc0,n0)*n0
            return torch.norm(xs-xsp)
        def gt_func00(CtrlPts00):
            CtrlPts0 = list(CtrlPts)
            CtrlPts0[0][0] = CtrlPts00
            tau1, tau2 = DtGrg(CtrlPts0,t0[0],t0[1]).T
            N = torch.cross(tau1,tau2)
            normN = torch.norm(N)
            # n0 = (N/normN)[0]
            n0 = (N/normN)
            xc0 = Grg(CtrlPts0,t0[0],t0[1])
            # gn0 = torch.tensor(gn0,dtype=torch.float64,requires_grad=False)
            xsp = xc0 + torch.dot(xs-xc0,n0)*n0
            return torch.norm(xs-xsp)
        def gt_func121(CtrlPts121):
            CtrlPts0 = list(CtrlPts)
            CtrlPts0[1][2][1] = CtrlPts121
            tau1, tau2 = DtGrg(CtrlPts0,t0[0],t0[1]).T
            N = torch.cross(tau1,tau2)
            normN = torch.norm(N)
            n0 = (N/normN)
            xc0 = Grg(CtrlPts0,t0[0],t0[1])
            # gn0 = torch.tensor(gn0,dtype=torch.float64,requires_grad=False)
            xsp = xc0 + torch.dot(xs-xc0,n0)*n0
            return torch.norm(xs-xsp)

        if tracing: set_trace()


        Energy = 1/2*kn*(-gn)**2 if gn < 0 else 0

        if thing == "f":
            return np.array(force.detach())
        elif thing =="K":

            set_trace()
            K1 = jacobian(force, u)
            return np.array(K1)
        else:
            print("provide strings 'f' or 'K' as arg 'thing'")

    def KT_fd(self,u0,xs,kn,kt,mu,hook=None,Stick=True, cubicT=None, du=1e-7):
        Ne = len(self.squad)
        K  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        f0 = self.fintC_tot2(xs,kn,kt,mu,hook=hook,Stick=Stick, cubicT=cubicT)[1]

        dofs = np.append([0,0,0],self.surf.body.DoFs[self.squad])

        for ii,dof in enumerate(dofs):
            u = np.array(u0)
            xs_new = np.array(xs)
            if ii<3:
                xs_new[ii]+=du
            else:
                u[dof] += du
                self.getCtrlPts(u)

            f1 = self.fintC_tot2(xs_new,kn,kt,mu,hook=hook,Stick=Stick, cubicT=cubicT)[1]
            K[:,ii] = (f1-f0)/du
        self.getCtrlPts(u0)         #restoring the CtrlPts before finishing

        return K

    def derGT(self,u0,xs,kn,kt,mu,hook=None,Stick=True, cubicT=None, du=5e-8):
        Ne = len(self.squad)
        K  = np.zeros( (3*(Ne+1) , 3*(Ne+1)) )

        f0 = self.fintC_tot(xs,kn,kt,mu,hook=hook,Stick=Stick, cubicT=cubicT,output="dgtdu")

        dofs = np.append([0,0,0],self.surf.body.DoFs[self.squad])

        for ii,dof in enumerate(dofs):
            u = np.array(u0)
            xs_new = np.array(xs)
            if ii<3:
                xs_new[ii]+=du
            else:
                u[dof] += du
                self.getCtrlPts(u)

            f1 = self.fintC_tot(xs_new,kn,kt,mu,hook=hook,Stick=Stick, cubicT=cubicT,output="dgtdu")
            K[:,ii] = (f1-f0)/du
        self.getCtrlPts(u0)         #restoring the CtrlPts before finishing

        return K


def wij(Xi,Xj):
    w11 = Xi[0]-Xj[0]
    w22 = Xi[1]-Xj[1]
    w33 = Xi[2]-Xj[2]
    wij = np.array( [ [   0, -w33,  w22],
                      [ w33,    0, -w11],
                      [-w22,  w11,    0] ] )

    return wij








"""
    def N_tild(self,t1,t2,normal):          # DELETE: use dxcdxi to buil N_tild in self.fintC&K instead
        N = np.zeros((21,3))
        N[0] = 1.0
        c_edges = [[0,0],[3,0],[3,3],[0,3],[1,0],[2,0],[3,1],[3,2],[2,3],[1,3],[0,2],[0,1]]
        for ic, c in enumerate(c_edges):
            N[ic+1] = -Bernstein(3, c[0], t1)*Bernstein(3, c[1], t2)

        den = max(self.eps,t1+t2)
        N[13] = -Bernstein(3, 1, t1)*Bernstein(3, 1, t2)*(t1/den)
        N[14] = -Bernstein(3, 1, t1)*Bernstein(3, 1, t2)*(t2/den)
        den = max(self.eps,1-t1+t2)
        N[15] = -Bernstein(3, 2, t1)*Bernstein(3, 1, t2)*((1-t1)/den)
        N[16] = -Bernstein(3, 2, t1)*Bernstein(3, 1, t2)*(t2/den)
        den = max(self.eps,2-t1-t2)
        N[17] = -Bernstein(3, 2, t1)*Bernstein(3, 2, t2)*((1-t1)/den)
        N[18] = -Bernstein(3, 2, t1)*Bernstein(3, 2, t2)*((1-t2)/den)
        den = max(self.eps,t1+1-t2)
        N[19] = -Bernstein(3, 1, t1)*Bernstein(3, 2, t2)*(t1/den)
        N[20] = -Bernstein(3, 1, t1)*Bernstein(3, 2, t2)*((1-t2)/den)

        N *= normal

        return N


    def dy1v(self,idx, v, inverse = False, tracing=False):
        "idx0: index of the edge (0,..,3). 'inverse' option does the rest"
        Ne = len(self.squad)
        dy1v = np.zeros((Ne,3))     # Here I don't consider the DoF_slave

        if inverse:
            y3, y0 = self.Y0Y3(idx)
            idx3  = idx
            idx0  = [1,2,3,0][idx]
        else:
            y0, y3 = self.Y0Y3(idx)
            idx0 = idx
            idx3 = [1,2,3,0][idx]

        m0 = self.normals[idx0]
        XI = self.W[idx0]

        c1 = (m0 @ v) / (m0 @ m0)
        c2 = ( (y3-y0) @ m0 ) / (m0 @ m0)

        # dy1v[idx0] += v + 1/3*(2*v + c1*m0)   # Wrong formula! 
        dy1v[idx0] += 1/3*(2*v + c1*m0)
        dy1v[idx3] += 1/3*(v-c1*m0)
        dy1v       += (XI.T @ (1/3*(2*c1*c2*m0 - c1*(y3-y0) - c2*v))).reshape((Ne,3))

        return dy1v

    def dyp1v(self,idx, v, inverse = False):
        "idx0: index of the edge (0,..,3). 'inverse' option does the rest"
        Ne = len(self.squad)
        dyp1v = np.zeros((Ne,3))     # Here I don't consider the DoF_slave

        if inverse:
            y3,   y0   = self.Y0Y3(idx)
            y2,   y1   = self.Y1Y2(idx)
            yp3,  yp0  = self.Yp0Yp3(idx)
            idx3  = idx
            idx0  = [1,2,3,0][idx]

            idx00 = idx0
            idx03 = [3,0,1,2][idx]

        else:
            y0,   y3   = self.Y0Y3(idx)
            y1,   y2   = self.Y1Y2(idx)
            yp0,  yp3  = self.Yp0Yp3(idx)
            idx0 = idx
            idx3 = [1,2,3,0][idx]

            idx00 = [3,0,1,2][idx]
            idx03 = idx3

        m0, m3 = self.normals[idx0], self.normals[idx3]
        XI0, XI3 = self.W[idx0], self.W[idx3]

        # a0, a3, c0, c1, c2, b0, b3, k0, k1, h0, h1
        c0, c1, c2 = y1-y0, y2-y1, y3-y2
        b0, b3 = yp0-y0 , yp3-y3
        a0 = np.cross(m0,(y3-y0))/norm( np.cross(m0,(y3-y0)) )
        a3 = np.cross(m3,(y3-y0))/norm( np.cross(m3,(y3-y0)) )
        k0, k1 = np.dot(a0,b0), np.dot(a3,b3)
        h0, h1 = np.dot(c0,b0)/np.dot(c0,c0) , np.dot(c2,b3)/np.dot(c2,c2)
        # constants c1-c4   (not vectors)
        cc1 = (a0 @ v + a3 @ v)/(a0 @ a0)
        cc2 = (a0 @ v)/(a3 @ a3)
        cc3 = (c1 @ v)/(c0 @ c0)
        cc4 = (c0 @ v)/(c2 @ c2)

        # vectors v1-v7
        # (Puso & Laursen)
        v1 = 1/3*((k1+k0)*v + cc1*(b0-2*k0*a0))
        v2 = 1/3*(k0*v + cc2*(b3-2*k1*a3))
        v3 = 1/3*(cc1*a0 + 2*cc3*c0)
        v4 = 1/3*(cc2*a3 + cc4*c2)
        v5 = 1/3*(h1*v + 2*cc3*(b0-2*h0*c0))
        v6 = 1/3*(cc4*(b3 - 2*h1*c2))
        v7 = 2/3*(h0*v)

        # vectors v1-v7
        # (mine)
        # v1 = 1/3*((k1+k0)*v + np.outer(b0,a0+a3)@v)
        # v2 = 1/3*(( k0  )*v + np.outer(b3, a0  )@v)
        # for v3,...,v7 I get the same value but probably Puso's is faster
        # v3 = 1/3*(np.outer(a0,a0+a3)+2*np.outer(c0,c1)/np.dot(c0,c0))@v
        # v4 = 1/3*(np.outer(a3,  a0 )+ np.outer(c2,c0)/np.dot(c2,c2))@v
        # v5 = 1/3*(h1*np.eye(3) + 2*(np.outer(b0,c0)-2*np.outer(c0,b0))@np.outer(c0,c1)/np.dot(c0,c0)**2)@v
        # v6 = 1/3*(np.outer(b3,c2)-2*np.outer(c2,b3))@np.outer(c2,c0)@v/np.dot(c2,c2)**2
        # v7 = 2/3*(h0*v)


        # set_trace()

        # dy1v
        dyp1v       += self.dy1v(idx, v, inverse=inverse)

        # # da0v1       (Puso)
        # dyp1v       += (XI0 @ np.cross((y3 - y0),v1) ).reshape((Ne,3))  #Change here
        # dyp1v[idx3] += -np.cross(m0,v1)
        # dyp1v[idx0] +=  np.cross(m0,v1)
        # # da3v2       (Puso)
        # dyp1v       += (XI3 @ np.cross((y3 - y0),v2) ).reshape((Ne,3))
        # dyp1v[idx3] += -np.cross(m3,v2)
        # dyp1v[idx0] +=  np.cross(m3,v2)
        
        # da0v1 (mine)
        dyp1v       += (XI0.T @ np.cross((y3 - y0),(np.eye(3)-np.outer(a0,a0))@v1/norm(np.cross(m0,y3-y0))) ).reshape((Ne,3))
        dyp1v[idx0] +=  np.cross(m0,(np.eye(3)-np.outer(a0,a0))@v1/norm(np.cross(m0,y3-y0)))
        dyp1v[idx3] += -np.cross(m0,(np.eye(3)-np.outer(a0,a0))@v1/norm(np.cross(m0,y3-y0)))
        
        # da3v2 (mine)
        dyp1v       += (XI3.T @ np.cross((y3 - y0),(np.eye(3)-np.outer(a3,a3))@v2/norm(np.cross(m3,y3-y0))) ).reshape((Ne,3))
        dyp1v[idx0] +=  np.cross(m3,(np.eye(3)-np.outer(a3,a3))@v2/norm(np.cross(m3,y3-y0)))
        dyp1v[idx3] += -np.cross(m3,(np.eye(3)-np.outer(a3,a3))@v2/norm(np.cross(m3,y3-y0)))
        
        # db0v3
        dyp1v       += self.dy1v(idx00, v3, inverse = not inverse)
        dyp1v[idx0] += -v3                              # Algo B
        
        # db3v4
        dyp1v       += self.dy1v(idx03, v4, inverse = inverse)
        dyp1v[idx3] += -v4                              # Algo B
        
        # dc0v5
        dyp1v       += self.dy1v(idx, v5, inverse = inverse)
        dyp1v[idx0] += -v5                              # Algo B
        
        # dc2v6
        dyp1v[idx3] += v6                              # Algo B
        dyp1v       += -self.dy1v(idx, v6, inverse = not inverse)
        
        # dc1v7
        dyp1v       +=  self.dy1v(idx, v7, inverse = not inverse)
        dyp1v       += -self.dy1v(idx, v7, inverse = inverse)

        return dyp1v


    def fintC_puso(self, xs, kn, seeding = 10, tracing = False, toprint=None):       # using Puso & Laursen variables fro chain rule
        Ne = len(self.squad)

        t1,t2 = self.findProjection(xs,seeding=seeding)
        if not (0<=t1<=1 and 0<=t2<=1):
            return 0.0 , 0.0

        
        xc = self.Grg(t1,t2)
        normal = self.D3Grg(t1,t2)
        gn = (xs-xc) @ normal             # np.dot(a,b) <=> a @ b
        if gn < 0:
            
            N_tild = self.N_tild(t1,t2,normal)

            t_n = kn*gn   # < 0     because gn<0 at this point
            r_tild = -t_n*N_tild   # forces conjugated to slave+CtrlPts



            fintC = np.zeros((Ne+1,3),dtype=float)
            # N = np.zeros((Ne+1,3),dtype=float)

            # SLAVE:
            fintC[0] = r_tild[0]
            # N[0] = N_tild[0]

            # VERTICES:
            for idx in range(1,5):
                v  = r_tild[idx]

                fintC[idx] += v

                # N[idx] += N_tild[idx]

                printif(idx==toprint, v)

            # EDGES:
            for idx in range(5,13):
                v  = r_tild[idx]

                if idx%2==1:        # 5,7,9,11 (normal, CCW)
                    idx0 = int((idx-5)/2)                       # edge in which the node belongs
                    inverse = False                             # is orientation inverse (CW)?
                else:               # 6,8,10,12 (CW)
                    idx0 = int((idx-6)/2)                       # edge in which the node belong 
                    inverse = True                              # is orientation inverse (CW)?


                dy1v = self.dy1v(idx0, v, inverse=inverse)
                fintC[1:] += dy1v

                # dy1N = self.dy1v(idx0, N_tild[idx], inverse=inverse)
                # N[1:] = dy1N

                printif(idx==toprint, dy1v)


            # print(fintC)
            # INTERIOR:
            for idx in range(13,21):
                v  = r_tild[idx]

                idx0 = [ 0, 3, 0, 1, 2, 1, 2, 3 ][idx-13]       # edge in which the node belongs
                inverse = [False, True, True, False, False, True, True, False][idx-13]

                dyp1v = self.dyp1v(idx0, v, inverse=inverse)
                fintC[1:] += dyp1v

                # dyp1N = self.dyp1v(idx0, N_tild[idx], inverse=inverse)
                # N[1:] = dyp1N
                
                printif(idx==toprint, dyp1v)

                assert np.allclose(v,sum(dyp1v)), "f_cp !=  sum(f_nodes)"



            assert np.allclose(sum(fintC), np.array([0,0,0])) , "forces in CtrlPts and nodes dont match!"


            # k = -kn*np.outer(N,N)

            return fintC.ravel()

        else:
            return 0.0, 0.0

    def fintC_mine_cp(self, xs, kn, seeding = 10, toprint=None):       #Chain rule by hand

        t1,t2 = self.findProjection(xs, seeding=seeding)
        if not (0<=t1<=1 and 0<=t2<=1):
            return 0.0
        
        xc = self.Grg(t1,t2)
        normal = self.D3Grg(t1,t2)
        gn = (xs-xc) @ normal             # np.dot(a,b) <=> a @ b
        

        if gn < 0:

            dgdxi = np.zeros((20,3))
            
            dgdxc = -(xs-xc)/norm(xs-xc)            # norm(xs-xc) != gn   (opposite sign)
            DxcDxi = -self.N_tild(t1,t2, 1.0)[1:]           ### <<<=== N_tild contains -dGrgdxi(t1,t2,normal)

            #############
            ## CtrlPts ##:
            #############
            for idx in range(0,20):
                dgdxi[ idx ] += dgdxc @ np.diag(DxcDxi[idx])

                printif(idx==toprint, kn*gn*dgdxc @ np.diag(DxcDxi[idx]))

            return kn*gn*dgdxi

        else:
            return 0.0


    def fintC_fd(self, xs, kn, u0, seeding = 10, du = 1e-10):       # finite difference at nodes


        t1,t2 = self.findProjection(xs, seeding = seeding)
        if not (0<=t1<=1 and 0<=t2<=1):
            set_trace()

        xc = self.Grg(t1,t2)
        normal = self.D3Grg(t1,t2)
        gn = (xs-xc) @ normal             # np.dot(a,b) <=> a @ b

        if gn>0:
            return 0

        else:

            dofs = np.append([0,0,0],self.surf.DoFs[self.squad])
            n = 3*(len(self.squad)+1)
            E0 = 0.5*kn*gn**2

            fintC = np.zeros(n)
            for idx, dof in enumerate(dofs):
                u = np.array(u0)
                xs_now = np.array(xs)

                if idx < 3 :
                    xs_now[idx] += du
                else:
                    u[dof] += du
                    self.getCtrlPts(u)

                t1,t2 = self.findProjection(xs_now, seeding = seeding)
                xc = self.Grg(t1,t2)
                normal = self.D3Grg(t1,t2)
                gn = (xs_now-xc) @ normal             # np.dot(a,b) <=> a @ b

                fintC[idx]= (0.5*kn*gn**2 - E0)/du


            self.getCtrlPts(u0)         #restoring the CtrlPts before finishing
            return fintC

    def fintC_fd_cp(self, xs, kn, u0, seeding = 10):    # finite differente at CtrlPts
        self.getCtrlPts(u0)         #restoring the CtrlPts before finishing

        t1,t2 = self.findProjection(xs, seeding = seeding)
        if not (0<=t1<=1 and 0<=t2<=1):
            set_trace()

        xc = self.Grg(t1,t2)
        normal = self.D3Grg(t1,t2)
        gn = (xs-xc) @ normal             # np.dot(a,b) <=> a @ b

        if gn>0:
            return 0

        else:

            cp0 = np.array(self.flatCtrlPts()).ravel()
            dofs = np.append([0,0,0],cp0)
            n = len(cp0)+3
            E0 = 0.5*kn*gn**2
            fintC = np.zeros(n)

            du = 1e-9
            for idx, dof in enumerate(dofs):
                xs_now = np.array(xs)
                self.CtrlPts = np.array(cp0)

                if idx < 3 :
                    xs_now[idx] += du
                else:
                    self.CtrlPts[idx-3] += du

                self.CtrlPts = self.CtrlPts.reshape(20,3)
                self.groupCtrlPts()
                t1,t2 = self.findProjection(xs_now, seeding = seeding)
                xc = self.Grg(t1,t2)
                normal = self.D3Grg(t1,t2)
                gn = (xs_now-xc) @ normal             # np.dot(a,b) <=> a @ b

                fintC[idx]= (0.5*kn*gn**2 - E0)/du


            self.getCtrlPts(u0)         #restoring the CtrlPts before finishing
            return fintC


    def torch_fintC_KC(self, xs, kn, u, seeding = 10):
        import torch

        from scipy.special import comb, factorial
        def Bernstein(n,k,x):
            return comb(n,k)*(x**k)*((1-x)**(n-k))

        def dnBernstein(n,k,x,p):
            coef = factorial(n)/factorial(n-p)
            desde = max(0,k+p-n)
            hasta = min(k,p)
            dnB = coef*sum([(-1)**(i+p)*comb(p,i)*Bernstein(n-p,k-i,x) for i in range(desde,hasta+1)])
            return dnB

        def Grg(CtrlPts,u,v, eps = 1e-8):                      # with treatment for undefinition at nodes
            p = torch.tensor([0.0 , 0.0 , 0.0],dtype = torch.float64)
            n, m = len(CtrlPts)-1, len(CtrlPts[0])-1

            for i  in range(n+1):
                for j in range(m+1):
                    if i in [1,2] and j in [1,2]:
                        if i==1 and j ==1:
                            x110, x111 = CtrlPts[1][1]
                            den = max(eps,u+v)  
                            xij = (u*x110+v*x111)/(den)
                        elif i==1 and j==2:
                            x120, x121 = CtrlPts[1][2]
                            den = max(eps,u+1-v)
                            xij = (u*x120+(1-v)*x121)/(den)

                        elif i==2 and j==1:
                            x210, x211 = CtrlPts[2][1]
                            den = max(eps, v+1-u)
                            xij = ((1-u)*x210+v*x211)/(den)
                        else:
                            x220, x221 = CtrlPts[2][2]
                            den = max(eps, 2-u-v)
                            xij = ((1-u)*x220+(1-v)*x221)/(den)
                            
                    else:
                        xij = CtrlPts[i][j]

                    Bi   =  Bernstein(n, i, u)
                    Bj   =  Bernstein(m, j, v)
                    p += Bi*Bj*xij
            return p

        def DtGrg(CtrlPts,u,v, order = 1 , eps = 1e-8):                        # Normalized normal vector at (u,v) with treatment for undefinition at nodes
            D1p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D1p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D2D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D1D1p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D1D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D1D2D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            D2D2D2p = torch.tensor([0.0 , 0.0 , 0.0],dtype=torch.float64)
            n, m = len(CtrlPts)-1, len(CtrlPts[0])-1

            for i  in range(n+1):
                for j in range(m+1):
                    # Inner nodes: values, derivatives and treatments
                    if i in [1,2] and j in [1,2]:
                        if i==1 and j ==1:
                            x110, x111 = CtrlPts[1][1]
                            den = max(eps,u+v)  
                            xij = (u*x110+v*x111)/(den)
                            D1xij = x110/(den) - (u*x110 + v*x111)/((den)**2)
                            D2xij = x111/(den) - (u*x110 + v*x111)/((den)**2)
                            if order > 1:
                                D11xij =-2*v*(x110 - x111)/(den)**3
                                D12xij = (u - v)*(x110 - x111)/(den)**3
                                D22xij = 2*u*(x110 - x111)/(den)**3
                            if order > 2:
                                D111xij = 6*v*(x110-x111)/(den)**4
                                D112xij = -(2*(u-2*v)*(x110-x111)/(den)**4)
                                D122xij = -(2*(2*u-v)*(x110-x111)/(den)**4)
                                D222xij = -(6*u*(x110-x111)/(den)**4)
                                
                        elif i==1 and j==2:
                            x120, x121 = CtrlPts[1][2]
                            den = max(eps,u+1-v)
                            xij = (u*x120+(1-v)*x121)/(den)
                            D1xij = x120/(den) - (u*x120 + (1-v)*x121)/((den)**2)
                            D2xij = -x121/(den) + (u*x120 + (1-v)*x121)/((den)**2)
                            if order > 1:
                                D11xij = (2*(-1 + v)*(x120 - x121))/(den)**3
                                D12xij = -(((-1 + u + v)*(x120 - x121))/(den)**3)
                                D22xij = (2*u*(x120 - x121))/(den)**3
                            if order > 2:
                                D111xij = -(6*(-1+v)*(x120-x121)/(den)**4)
                                D112xij = (2*(-2+u+2*v)*(x120-x121)/(den)**4)
                                D122xij = -(2*(-1+2*u+v)*(x120-x121)/(den)**4)
                                D222xij = (6*u*(x120-x121)/(den)**4)

                        elif i==2 and j==1:
                            x210, x211 = CtrlPts[2][1]
                            den = max(eps, v+1-u)
                            xij = ((1-u)*x210+v*x211)/(den)
                            D1xij = -x210/(den) + ((1-u)*x210 + v*x211)/((den)**2)
                            D2xij = x211/(den) - ((1-u)*x210 + v*x211)/((den)**2)
                            if order > 1:
                                D11xij = (2*v*(x210 - x211))/(-den)**3
                                D12xij = ((-1 + u + v)*(x210 - x211))/(den)**3
                                D22xij = (2*(-1 + u)*(x210 - x211))/(-den)*3
                            if order > 2:
                                D111xij = -(6*v*(x210-x211)/(den)**4)
                                D112xij = (2*(-1+u+2*v)*(x210-x211)/(den)**4)
                                D122xij = -(2*(-2+2*u+v)*(x210-x211)/(den)**4)
                                D222xij = (6*(-1+u)*(x210-x211)/(den)**4)

                        else:
                            x220, x221 = CtrlPts[2][2]
                            den = max(eps, 2-u-v)
                            xij = ((1-u)*x220+(1-v)*x221)/(den)
                            D1xij = -x220/(den) + ((1-u)*x220 + (1-v)*x221)/((den)**2)
                            D2xij = -x221/(den) + ((1-u)*x220 + (1-v)*x221)/((den)**2)
                            if order > 1:
                                D11xij = -((2*(-1 + v)*(x220 - x221))/(-den)**3)
                                D12xij = ((u - v)*(x220 - x221))/(-den)**3
                                D22xij = (2*(-1 + u)*(x220 - x221))/(-den)**3
                            if order > 2:
                                D111xij = 6*(-1+v)*(x220-x221)/(-den)**4
                                D112xij = -(2*(1+u-2*v)*(x220-x221)/(-den)**4)
                                D112xij = -(2*(-1+2*u-v)*(x220-x221)/(-den)**4)
                                D222xij = -(6*(-1+u)*(x220-x221)/(-den)**4)

                    else:
                        xij = CtrlPts[i][j]
                        D1xij, D2xij = 0.0 , 0.0
                        D11xij, D12xij, D22xij = 0.0 , 0.0 , 0.0
                        D111xij, D112xij, D122xij, D222xij = 0.0 , 0.0 , 0.0 , 0.0

                    # Bernstein polynomials
                    Bi     =   Bernstein(n, i, u)
                    Bj     =   Bernstein(m, j, v)
                    D1Bi     =  dnBernstein(n, i, u, 1)
                    D2Bj     =  dnBernstein(m, j, v, 1)
                    if order > 1:
                        DD1Bi = dnBernstein(n, i, u, 2)
                        DD2Bj = dnBernstein(m, j, v, 2)

                    if order > 2:
                        DDD1Bi = dnBernstein(n, i, u, 3)
                        DDD2Bj = dnBernstein(m, j, v, 3)

                    #TODO: Unify Grg and D3Grg: Just add "p" bellow to be computed as p+=...

                    # Tangent Derivatives w/r to LOCAL parameters
                    D1p += D1Bi*Bj*xij + Bi*Bj*D1xij
                    D2p += Bi*D2Bj*xij + Bi*Bj*D2xij

                    if order > 1:
                        D1D1p += (DD1Bi*xij + 2*D1Bi*D1xij + Bi*D11xij)*Bj
                        D1D2p += D1Bi*D2Bj*xij + D1Bi*Bj*D2xij + Bi*D2Bj*D1xij + Bi*Bj*D12xij
                        D2D2p += (DD2Bj*xij + 2*D2Bj*D2xij + Bj*D22xij)*Bi

                    if order > 2:
                        D1D1D1p += (DDD1Bi*xij + 3*DD1Bi*D1xij + 3*D1Bi*D11xij + Bi*D111xij)*Bj
                        D1D1D2p += (DD1Bi*D2xij + 2*D1Bi*D12xij + Bi*D112xij)*Bj + (DD1Bi*xij + 2*D1Bi*D1xij + Bi*D11xij)*D2Bj
                        D1D2D2p += (DD2Bj*D1xij + 2*D2Bj*D12xij + Bj*D122xij)*Bi + (DD2Bj*xij + 2*D2Bj*D2xij + Bj*D22xij)*D1Bi
                        D2D2D2p += (DDD2Bj*xij + 3*DD2Bj*D2xij + 3*D2Bj*D22xij + Bj*D222xij)*Bi

            if order==1:
                return torch.transpose(torch.stack([D1p, D2p]),0,1)
            elif order == 2:
                return torch.transpose(torch.stack([D1p, D2p]),0,1), torch.transpose(torch.stack([torch.stack([D1D1p, D1D2p]),
                                                    torch.stack( [D1D2p, D2D2p])    ]),0,1)
            elif order == 3:
                return torch.transpose(torch.stack([D1p, D2p]),0,1), torch.transpose(torch.stack([[D1D1p, D1D2p], [D1D2p, D2D2p]]),0,1), torch.transpose(torch.stack([[[D1D1D1p,D1D1D2p],[D1D1D2p,D1D2D2p]], [[D1D1D2p,D1D2D2p],[D1D2D2p,D2D2D2p]]]),0,1)

        def D3Grg(CtrlPts, u,v, normalize = True):                        # Normalized normal vector at (u,v) with treatment for undefinition at nodes
            D1p, D2p = DtGrg(CtrlPts,u,v).T
            D3p = torch.cross(D1p,D2p)
            if normalize:
                D3p = D3p/torch.norm(D3p)       # The NORMALIZED vectors are continuous from patch to patch (doesnt make sense otherwise)
            return D3p

        def MinDist(CtrlPts, xs, seeding = seeding):
            umin = 0
            vmin = 0
            dmin = torch.norm(xs-CtrlPts[0][0])   # Starting point

            for u in torch.linspace(0,1,seeding+1):
                for v in torch.linspace(0,1,seeding+1):
                    d = torch.norm(xs - Grg(CtrlPts,u,v))
                    if d < dmin:
                        dmin, umin, vmin = d, u, v

            return umin , vmin

        def findProjection(CtrlPts,xs, seeding=seeding):

            t = torch.tensor(MinDist(CtrlPts, xs, seeding=seeding))

            tol = 1e-12
            res = 1+tol
            count = 0
            while res>tol and (0<=t[0]<=1 and 0<=t[1]<=1) and count<100:

                xc = Grg(CtrlPts, t[0],t[1])

                dxcdt, d2xcd2t = DtGrg(CtrlPts,t[0],t[1], order = 2)


                f = -2*torch.matmul((xs-xc),dxcdt)                        # Unnecessary (?)
                K =  2*(torch.tensordot(-( xs-xc),d2xcd2t,dims = ([0],[2])) + torch.tensordot(dxcdt,dxcdt,dims= ([0],[0]) ) )


                dt = torch.linalg.solve(-K,f)

                t = t + dt

                res = torch.norm(f)

                count += 1
            return t

        def jacobian(y, x, create_graph=False):                                                            
            jac = []                                                                                       
            flat_y = y.reshape(-1)                                                                         
            grad_y = torch.zeros_like(flat_y)                                                                 
            for i in range(len(flat_y)):                                                                   
                grad_y[i] = 1.                                                                             
                grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
                jac.append(grad_x.reshape(x.shape))                                                        
                grad_y[i] = 0.                                                                                
            return torch.stack(jac).reshape(y.shape + x.shape)        

        def hessian(y, x):                                                                                 
            return jacobian(jacobian(y, x, create_graph=True), x)                                          

        # Slave node
        Xs = torch.tensor(xs,dtype=torch.float64, requires_grad=False)
        us = torch.zeros(3)
        # Master nodes
        Xm = torch.tensor(self.X,dtype=torch.float64, requires_grad = False)
        um = torch.tensor(u[self.surf.DoFs[self.squad]])

        # Displacements vector
        u =  torch.row_stack((us,um)).flatten()
        u.requires_grad = True

        x = torch.row_stack((Xs,Xm))+u.reshape(-1,3)
        xs = x[0,:]
        xm = x[1:,:]

        #Computation of normals
        facets = self.facets
        normals = []
        for vert_idx, vert in enumerate(facets):
            Xi, xi = Xm[vert_idx], xm[vert_idx]
            ntmp = 0
            n0mp = 0
            ntm = 0

            for facet in vert:
                Xna, Xnb = [Xm[i] for i in facet]
                xna, xnb = [xm[i] for i in facet]

                Xa, Xb = (Xna)-(Xi) , (Xnb)-(Xi)
                xa, xb = (xna)-(xi) , (xnb)-(xi)

                ntm_e = torch.cross(xb,xa)
                n0m_e = torch.cross(Xb,Xa)
                w0m_e = torch.norm(n0m_e)

                ntmp += ntm_e / w0m_e
                n0mp += n0m_e / w0m_e

            wm = torch.norm( n0mp )
            
            ntm = ntmp / wm
            normals.append(ntm)

        # Computation of CtrlPts
        start = []
        edge  = []
        inner = []

        # going through every edge of the patch to find edge points
        for idx in range(4):
            y0, y3 = xm[range(4)[idx-1]], xm[idx]
            m0, m3 = normals[idx-1] , normals[idx]

            c0 = 1/3*((y3-y0)-torch.matmul((y3-y0),torch.outer(m0,m0)/torch.dot(m0,m0)))
            c2 = 1/3*((y3-y0)-torch.matmul((y3-y0),torch.outer(m3,m3)/torch.dot(m3,m3)))
            y1, y2 = y0+c0, y3-c2

            start.append(y0)
            edge.append([y1,y2])

        # going through every edge of the patch to find inner points
        for idx in range(4):
            y0, y3 = xm[range(4)[idx-1]] , xm[idx]
            m0, m3 = normals[idx-1] , normals[idx]
            
            y1, y2 = edge[idx][0], edge[idx][1]
            y0p , y3p = edge[idx-1][1], edge[idx-3][0]

            # c0, c1, c2 = y1-y0, y2-y1, y3-y2
            c0 = y1-y0
            c1 = y2-y1
            c2 = y3-y2
            b0, b3 = y0p-y0 , y3p-y3

            a0 = torch.cross(m0,(y3-y0))/torch.norm( torch.cross(m0,(y3-y0)) )
            a3 = torch.cross(m3,(y3-y0))/torch.norm( torch.cross(m3,(y3-y0)) )

            k0, k1 = torch.dot(a0,b0), torch.dot(a3,b3)
            h0, h1 = torch.dot(c0,b0)/torch.dot(c0,c0) , torch.dot(c2,b3)/torch.dot(c2,c2)

            b1 = 1/3*((k1+k0)*a0+k0*a3+2*h0*c1+h1*c0)
            b2 = 1/3*((k1+k0)*a3+k1*a0+2*h1*c1+h0*c2)

            y1p, y2p = y1+b1 , y2+b2

            inner.append([y1p,y2p])

        # Assembly of CtrlPts matrix of the patch
        row0 = [ start[1]  ,         edge[0][1]        ,         edge[0][0]        ,  start[0]  ]     #This "matrix" HAS to be transposed (for indexing purposes)
        row1 = [edge[1][0] , [inner[1][0],inner[0][1]] , [inner[3][1],inner[0][0]] , edge[3][1] ]
        row2 = [edge[1][1] , [inner[1][1],inner[2][0]] , [inner[3][0],inner[2][1]] , edge[3][0] ]
        row3 = [ start[2]  ,         edge[2][0]        ,         edge[2][1]        ,  start[3]  ]

        CtrlPts = [row0,row1,row2,row3]


        cp = CtrlPts
        CtrlPts_flat = torch.stack([
            cp[0][0], cp[3][0], cp[3][3], cp[0][3],
            cp[1][0], cp[2][0], cp[3][1], cp[3][2],cp[2][3], cp[1][3], cp[0][2], cp[0][1],
            cp[1][1][0], cp[1][1][1], cp[2][1][0], cp[2][1][1],cp[2][2][0], cp[2][2][1], cp[1][2][0], cp[1][2][1],
        ])

        # CtrlPts_flat.requires_grad=True

        cp = CtrlPts_flat
        row0 = [ cp[0] ,      cp[11]     ,       cp[10]    , cp[3] ]            #This "matrix" HAS to be transposed (for indexing purposes)
        row1 = [ cp[4] , [cp[12],cp[13]] , [cp[18],cp[19]] , cp[9] ]
        row2 = [ cp[5] , [cp[14],cp[15]] , [cp[16],cp[17]] , cp[8] ]
        row3 = [ cp[1] ,       cp[6]     ,       cp[7]     , cp[2] ]
        
        CtrlPts = [row0,row1,row2,row3]


        t = findProjection(CtrlPts,xs)

        xc = Grg(CtrlPts , t[0], t[1])
        normal = D3Grg(CtrlPts, t[0], t[1])
        gn = torch.dot( xs-xc , normal )

        Energy = 1/2*kn*(-gn)**2 if gn < 0 else 0

        f1 = jacobian(Energy, u)                                                                           
        K1 = hessian(Energy, u)

        
        return f1, K1


    # FINITE DIFFERENCES FOR (VERIFICATION PURPOSES)
    def dfdt_fd(self,xs, t0=0 ,dt = 1e-6):
        if type(t0)==int:
            t0 = self.findProjection(xs)

        xc0, dxcdt0 = self.Grg(t0[0],t0[1], deriv = 1)
        f0    = -2*(xs-xc0)@dxcdt0                  # Unnecessary (?)

        dfdt = np.zeros((len(f0),len(t0)))
        for i in range(len(t0)):
            t = np.array(t0)
            t[i] += dt
            xc, dxcdt = self.Grg(t[0],t[1], deriv = 1)
            f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
            dfdt[:,i] = (f-f0)/dt

        return dfdt
            
    def d2fd2t_fd(self,xs, dt = 1e-6, ddt=1e-6):
        t0 = self.findProjection(xs)
        dfdt0 = self.dfdt_fd(xs,dt=ddt)
        d2fd2t = np.zeros((dfdt0.shape[0],dfdt0.shape[1],len(t0)))
        for i in range(len(t0)):
            t = np.array(t0)
            t[i] += dt
            dfdt    = self.dfdt_fd(xs, t0=t ,dt=ddt)                  # Unnecessary (?)
            d2fd2t[:,:,i] = (dfdt-dfdt0)/dt

        return d2fd2t

    def d2fdtdxs_fd(self,xs0, dx = 1e-6, ddt=1e-6):
        dfdt0 = self.dfdt_fd(xs0,dt=ddt)
        d2fdtdxs = np.zeros((dfdt0.shape[0],dfdt0.shape[1],len(xs0)))
        for i in range(len(xs0)):
            xs = np.array(xs0)
            xs[i] += dx
            dfdt    = self.dfdt_fd(xs, dt=ddt)                  # Unnecessary (?)
            d2fdtdxs[:,:,i] = (dfdt-dfdt0)/dx

        return d2fdtdxs

    def dfdxs_fd(self,xs0, dx = 1e-6):
        t0 = self.findProjection(xs0)
        xc0, dxcdt0 = self.Grg(t0[0],t0[1], deriv = 1)
        f0    = -2*(xs0-xc0)@dxcdt0                  # Unnecessary (?)
        dfdxs = np.zeros((len(f0),len(xs0)))
        for i in range(len(xs0)):
            xs = np.array(xs0)
            xs[i] += dx
            t = self.findProjection(xs)
            xc, dxcdt = self.Grg(t[0],t[1], deriv = 1)
            f    = -2*(xs-xc)@dxcdt                  # Unnecessary (?)
            # f    = -2*(xs-xc0)@dxcdt0                  # Unnecessary (?)

            dfdxs[:,i] = (f-f0)/dx

        return dfdxs

    def d2fd2xs_fd(self,xs0, dx = 1e-6, ddx=1e-6):
        dfdxs0    = self.dfdxs_fd(xs0,dx=ddx)                 # Unnecessary (?)
        d2fd2xs = np.zeros((dfdxs0.shape[0],dfdxs0.shape[1],len(xs0)))
        for i in range(len(xs0)):
            xs = np.array(xs0)
            xs[i] += dx
            dfdxs    = self.dfdxs_fd(xs,dx=ddx)                 # Unnecessary (?)

            d2fd2xs[:,i] = (dfdxs-dfdxs0)/dx

        return d2fd2xs

    def dtdxs_fd(self,xs0, dx = 1e-6):
        t0 = self.findProjection(xs0)
        dtdxs = np.zeros((len(t0),len(xs0)))
        for i in range(len(xs0)):
            xs = np.array(xs0)
            xs[i] += dx
            t = self.findProjection(xs)
            dtdxs[:,i]    = (t-t0)/dx

        return dtdxs

    def d2td2xs_fd(self,xs0, dx = 1e-6, ddx=1e-6):
        dtdxs0 = self.dtdxs_fd(xs0, dx = ddx)
        d2td2xs = np.zeros((dtdxs0.shape[0],dtdxs0.shape[1],len(xs0)))
        for i in range(len(xs0)):
            xs = np.array(xs0)
            xs[i] += dx
            dtdxs = self.dtdxs_fd(xs, dx = ddx)
            d2td2xs[:,:,i]    = (dtdxs-dtdxs0)/dx

        return d2td2xs

    def dfdxi_fd(self,xs, dx=1e-6):
        t0 = self.findProjection(xs)
        xc0, dxcdt0 = self.Grg(t0[0],t0[1],deriv=1)
        f0 = -2*(xs-xc0)@dxcdt0
        cp0 = self.flatCtrlPts()

        dfdxi = np.zeros((2,20,3))
        for i in range(20):
            for j in range(3):
                cp = np.array(cp0)
                cp[i,j] += dx
                self.CtrlPts = cp
                self.groupCtrlPts()
                xc, dxcdt = self.Grg(t0[0],t0[1],deriv=1)
                f = -2*(xs-xc)@dxcdt
                dfdxi[:,i,j] = (f-f0)/dx

        return dfdxi

    def d2xcdtdxi_fd(self,xs, dx =1e-6):
        t0 = self.findProjection(xs)
        _, dxcdt0 = self.Grg(t0[0],t0[1],deriv=1)
        cp0 = self.flatCtrlPts()

        d2xcdtdxi = np.zeros((3,2,20,3))
        for i in range(20):
            for j in range(3):
                cp = np.array(cp0)
                cp[i,j] += dx
                self.CtrlPts = cp
                self.groupCtrlPts()
                _, dxcdt = self.Grg(t0[0],t0[1],deriv=1)
                d2xcdtdxi[:,:,i,j] = (dxcdt-dxcdt0)/dx

        self.CtrlPts = cp0
        self.groupCtrlPts()

        return d2xcdtdxi

    def d3xcd2tdxi_fd(self,xs, dx =1e-6):
        t0 = self.findProjection(xs)
        _, _, d2xcd2t0 = self.Grg(t0[0],t0[1],deriv=2)
        cp0 = self.flatCtrlPts()

        d2xcdtdxi = np.zeros((3,2,2,20,3))
        for i in range(20):
            for j in range(3):
                cp = np.array(cp0)
                cp[i,j] += dx
                self.CtrlPts = cp
                self.groupCtrlPts()
                _, _, d2xcd2t = self.Grg(t0[0],t0[1],deriv=2)
                d2xcdtdxi[:,:,:,i,j] = (d2xcd2t-d2xcd2t0)/dx

        self.CtrlPts = cp0
        self.groupCtrlPts()

        return d2xcdtdxi

    def KC_fd(self, xs, kn, u0, seeding = 10, du = 1e-10):
        
        # set_trace()
        dofs = np.append([0,0,0],self.surf.DoFs[self.squad])
        n = 3*(len(self.squad)+1)
        f0 = self.fintC_fd(xs, kn, u0, seeding=seeding, du=du)

        if type(f0) == int:         # equivalent to "if gn>=0"
            return 0
        else:
                
            K = np.zeros((n,n))
            for idx, dof in enumerate(dofs):
                u = np.array(u0)
                xs_now = np.array(xs)

                if idx>2:
                    u[dof] += du
                    self.getCtrlPts(u)
                else:
                    xs_now[idx] += du

                K[:,idx] = (self.fintC_fd(xs_now,kn, u0, seeding=seeding, du=du) - f0) / du

            self.getCtrlPts(u0)         #restoring the CtrlPts before finishing
            return K

"""