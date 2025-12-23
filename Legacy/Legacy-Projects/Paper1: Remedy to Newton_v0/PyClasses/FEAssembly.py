from PyClasses.Utilities import flatList
from PyClasses.Surfaces import *

import numpy as np
import pickle, time
import concurrent.futures
import multiprocessing
from scipy import sparse
from numpy.linalg import norm as norm
from pdb import set_trace
from math import sqrt, pi, cos, sin
import random

from PyClasses import eigen_backend


class FEAssembly:
    def __init__(self,X,hexahedrons , name = "unnamed", recOuters = False):
        self.X = X
        self.hexas = hexahedrons.tolist() if type(hexahedrons)!= list else hexahedrons
        self.deleteReduntants()
        self.name = name
        self.DoFs = []              # assigned by FEModel constructor
        outers = pickle.load(open("RecoveryOuters.dat","rb"))[name] if recOuters else None
        self.surf = Surface(self, outers = outers)
        self.Youngsmodulus = 1
        self.Poissonsratio = 0.3
        self.isRigid = False

    def deleteReduntants(self):
        reds = self.redundantNodes()
        self.X = np.delete(self.X,reds,0)
        self.ReduceHexaOrders(reds)

    def redundantNodes(self):
        reds = []
        nodesinhexas = list(set(flatList(self.hexas)))
        for i in range(len(self.X)):
            if i not in nodesinhexas:
                reds.append(i)
        return reds

    def ReduceHexaOrders(self,idxs):
        sortedIdxs = sorted(idxs, reverse = True)
        new_hexas = []
        for hexa in self.hexas:
            new_hexa = []
            for node in hexa:
                for idx in sortedIdxs:
                    if node>idx:
                        node -= 1
                new_hexa.append(node)
            new_hexas.append(new_hexa)
        
        self.hexas = new_hexas

    # TRANSFORMATIONS
    def Translate(self, disp):
        newX = np.zeros_like(self.X)
        disp = np.array(disp)
        for i in range(len(self.X)):
            newX[i] = self.X[i] + disp

        self.X = newX
        self.surf.X = newX

    def Rotate(self, theta,dir, unit="rad"):
        import math as m

        if unit == "deg": theta=(pi/180)*theta       
        if dir in "xX":
            M = np.matrix([[ 1, 0           , 0           ],
                           [ 0, m.cos(theta),-m.sin(theta)],
                           [ 0, m.sin(theta), m.cos(theta)]])
        
        elif dir in "yY":
            M = np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                           [ 0           , 1, 0           ],
                           [-m.sin(theta), 0, m.cos(theta)]])

        else:
            M = np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                           [ m.sin(theta), m.cos(theta) , 0 ],
                           [ 0           , 0            , 1 ]])
        newX = np.array(self.X@M)
        self.X = newX
        self.surf.X = newX

    def RandDistort(self,d , nodes = None):
        """
            Distorts the position of the nodes in the provided list (local id for the current FEAssembly) an amount r_i such that 0<r_i<d
        """
        newX = np.array(self.X)
        if nodes == None:
            nodes = range(len(self.X))
        for node in nodes:
            H_ang = random.uniform(0,2*pi)
            V_ang = random.uniform(0,pi)
            r = random.uniform(0,d)
            newX[node] += np.array([ r*sin(V_ang)*cos(H_ang) , r*sin(V_ang)*sin(H_ang) , r*cos(V_ang) ])

            self.X = newX
            self.surf.X = newX

    def Resize(self, factor, center = None, dir = None):
        xc = np.array([0,0,0]) if center is None else np.array(center)
        newX = np.zeros_like(self.X)

        if dir is None:
            for idx, x in enumerate(self.X):
                newX[idx] = xc + factor*(x-xc)
        else:
            if dir in "Xx":
                dir = 0
            elif dir in "Yy":
                dir = 1
            elif dir in "Zz":
                dir = 2
            newX = self.X
            for idx, x in enumerate(self.X):
                newX[idx][dir] = xc[dir] + factor*(x[dir]-xc[dir])


        self.X = newX
        self.surf.X = newX


    def Bulge(self, factor, center=None, dir=None):
        xc = np.array([0, 0, 0]) if center is None else np.array(center)
        newX = np.zeros_like(self.X)

        if dir is None:
            for idx, x in enumerate(self.X):
                dist = np.linalg.norm(x - xc)
                newX[idx] = x + factor * (x - xc) / (dist + 1e-8)
        else:
            if dir in "Xx":
                dir = 0
            elif dir in "Yy":
                dir = 1
            elif dir in "Zz":
                dir = 2
            newX = self.X.copy()  # Create a copy of the original coordinates
            for idx, x in enumerate(self.X):
                dist = np.abs(x[dir] - xc[dir])
                if dist > 1e-8:
                    newX[idx][dir] = x[dir] + factor * (x[dir] - xc[dir]) / dist

        self.X = newX
        self.surf.X = newX



    # SELECTION OF NODES
    def SelectNodesByBox(self,xyz_a,xyz_b, OnSurface = False):
        nodes = []
        xa, ya, za = xyz_a
        xb, yb, zb = xyz_b
        for node_id, [x,y,z] in enumerate(self.X):
            if OnSurface and node_id not in self.surf.nodes:
                continue
            if xa<=x<=xb:
                if ya<=y<=yb:
                    if za<=z<=zb:
                        nodes.append(node_id)
        return nodes

    def SelectFlatSide(self, side, tol = 1e-6, OnSurface=False):
        minX = min(self.X[:,0])
        maxX = max(self.X[:,0])
        minY = min(self.X[:,1])
        maxY = max(self.X[:,1])
        minZ = min(self.X[:,2])
        maxZ = max(self.X[:,2])
        eps = tol
        if "x" in side:
            if "-" in side:
                return self.SelectNodesByBox([minX-eps,minY-eps,minZ-eps],[minX+eps,maxY+eps,maxZ+eps],OnSurface = OnSurface)
            else:
                return self.SelectNodesByBox([maxX-eps,minY-eps,minZ-eps],[maxX+eps,maxY+eps,maxZ+eps],OnSurface = OnSurface)
        elif "y" in side:
            if "-" in side:
                return self.SelectNodesByBox([minX-eps,minY-eps,minZ-eps],[maxX+eps,minY+eps,maxZ+eps],OnSurface = OnSurface)
            else:
                return self.SelectNodesByBox([minX-eps,maxY-eps,minZ-eps],[maxX+eps,maxY+eps,maxZ+eps],OnSurface = OnSurface)
        else:
            if "-" in side:
                return self.SelectNodesByBox([minX-eps,minY-eps,minZ-eps],[maxX+eps,maxY+eps,minZ+eps],OnSurface = OnSurface)
            else:
                return self.SelectNodesByBox([minX-eps,minY-eps,maxZ-eps],[maxX+eps,maxY+eps,maxZ+eps],OnSurface = OnSurface)

    def SelectHigherThan(self, dir, val = 0, Strict = True,OnSurface = False):
        nodes = []
        if type(dir) == int:
            None
        elif dir in "Xx":
            dir = 0
        elif dir in "Yy":
            dir = 1
        elif dir in "Zz":
            dir = 2
        for node_id, xyz in enumerate(self.X):
            if OnSurface and node_id not in self.surf.nodes:
                continue
            if Strict:
                if xyz[dir] > val:
                    nodes.append(node_id)
            elif xyz[dir] >= val:
                    nodes.append(node_id)


        return nodes

    def SelectLowerThan(self, dir, val = 0, Strict = True, OnSurface = False):
        nodes = []
        if type(dir) == int:
            None
        elif dir in "Xx":
            dir = 0
        elif dir in "Yy":
            dir = 1
        elif dir in "Zz":
            dir = 2
        for node_id, [x,y,z] in enumerate(self.X):
            if OnSurface and node_id not in self.surf.nodes:
                continue
            if Strict:
                if [x,y,z][dir] < val:
                    nodes.append(node_id)
            elif [x,y,z][dir] <= val:
                    nodes.append(node_id)
        return nodes

    def SelectNodesBySphere(self,xyz_center,radius, OnSurface = False):
        nodes = []
        xc = np.array(xyz_center,dtype=float)
        r = radius
        for node_id, x_node in enumerate(self.X):
            if OnSurface and node_id not in self.surf.nodes:
                continue
            x = np.array(x_node)
            if sqrt((x-xc)@(x-xc))<=r:
                nodes.append(node_id)
        return nodes

    def SelectAll(self):
        return range(len(self.X))

    # SELECTION OF QUADS:
    def SelectQuadsByNodes(self,nodes):
        quads = []
        nodeset = set(nodes)  # no surface needed. Hexas will be taken anyway
        for qid, quad in enumerate(self.surf.quads):
            common = set(quad).intersection(nodeset)
            if len(common)>0:
                quads.append(qid)
        return quads



    # SELECTION OF HEXAS:
    def SelectHexasByBox(self,xyz_a,xyz_b):
        nodes = self.SelectNodesByBox(xyz_a,xyz_b)  # no surface needed. Hexas will be taken anyway
        return self.SelectHexasByNodes(nodes)

    def SelectHexasByNodes(self,nodes):
        hexas = []
        nodeset = set(nodes)  # no surface needed. Hexas will be taken anyway
        for hid, hexa in enumerate(self.hexas):
            common = set(hexa).intersection(nodeset)
            if len(common)>0:
                hexas.append(hid)
        return hexas



    # MECHANICAL FUNCTIONS
    # Residuals at Finite Element level
    # 1. using cpp
    def fint_el(self,hexa,u):
        X_el = self.X[hexa]
        u_el = u[self.DoFs[hexa]]
        return eigen_backend.f_element_py(X_el, u_el, self.Youngsmodulus, self.Poissonsratio).ravel()
  
    def K_el(self,hexa,u):
        X_el = self.X[hexa]
        u_el = u[self.DoFs[hexa]]
        return eigen_backend.K_element_py(X_el, u_el, self.Youngsmodulus, self.Poissonsratio)

    # 2. using python (slower)
    def m_el(self,hexa,u):
        m=0
        X = self.X[hexa]
        u = u[self.DoFs[hexa]]

        d1 = self.Youngsmodulus*self.Poissonsratio/(2*(1+self.Poissonsratio)*(1-2*self.Poissonsratio))
        c10 = self.Youngsmodulus/(4*(1+self.Poissonsratio))

        gauss_points = 1/sqrt(3)*np.array([ [-1, -1, -1],
                                            [ 1, -1, -1],
                                            [ 1,  1, -1],
                                            [-1,  1, -1],
                                            [-1, -1,  1],
                                            [ 1, -1,  1],
                                            [ 1,  1,  1],
                                            [-1,  1,  1]])

        for (g1,g2,g3) in gauss_points:
            dNd_xi = 1/8*np.array([ [-(1-g2)*(1-g3) , -(1-g1)*(1-g3) , -(1-g1)*(1-g2) ],
                                    [ (1-g2)*(1-g3) , -(1+g1)*(1-g3) , -(1+g1)*(1-g2) ],
                                    [ (1+g2)*(1-g3) ,  (1+g1)*(1-g3) , -(1+g1)*(1+g2) ],
                                    [-(1+g2)*(1-g3) ,  (1-g1)*(1-g3) , -(1-g1)*(1+g2) ],
                                    [-(1-g2)*(1+g3) , -(1-g1)*(1+g3) ,  (1-g1)*(1-g2) ],
                                    [ (1-g2)*(1+g3) , -(1+g1)*(1+g3) ,  (1+g1)*(1-g2) ],
                                    [ (1+g2)*(1+g3) ,  (1+g1)*(1+g3) ,  (1+g1)*(1+g2) ],
                                    [-(1+g2)*(1+g3) ,  (1-g1)*(1+g3) ,  (1-g1)*(1+g2) ]])

            J = np.dot(dNd_xi.T,X)
            invJ = np.linalg.inv(J)
            detJ = np.linalg.det(J)

            dNdx = np.dot( dNd_xi , invJ.T )
            F = np.eye(len(dNdx.T)) + np.dot(dNdx.T,u).T
            detF = np.linalg.det(F)


            SED=c10*(np.trace(F.T@F)-3-2*np.log(detF))+d1*(np.log(detF))**2

            m += SED*detJ

        return m
    


    def m_el_extra(self, hexa, u):
        m = np.zeros(len(self.DoFs[hexa]))  # Initialize array to store nodal SED values
        X = self.X[hexa]
        u = u[self.DoFs[hexa]]

        d1 = self.Youngsmodulus * self.Poissonsratio / (2 * (1 + self.Poissonsratio) * (1 - 2 * self.Poissonsratio))
        c10 = self.Youngsmodulus / (4 * (1 + self.Poissonsratio))

        gauss_points = 1 / np.sqrt(3) * np.array([[-1, -1, -1],
                                                [1, -1, -1],
                                                [1, 1, -1],
                                                [-1, 1, -1],
                                                [-1, -1, 1],
                                                [1, -1, 1],
                                                [1, 1, 1],
                                                [-1, 1, 1]])

        for (g1, g2, g3) in gauss_points:
            dNd_xi = 1 / 8 * np.array([[-(1 - g2) * (1 - g3), -(1 - g1) * (1 - g3), -(1 - g1) * (1 - g2)],
                                    [(1 - g2) * (1 - g3), -(1 + g1) * (1 - g3), -(1 + g1) * (1 - g2)],
                                    [(1 + g2) * (1 - g3), (1 + g1) * (1 - g3), -(1 + g1) * (1 + g2)],
                                    [-(1 + g2) * (1 - g3), (1 - g1) * (1 - g3), -(1 - g1) * (1 + g2)],
                                    [-(1 - g2) * (1 + g3), -(1 - g1) * (1 + g3), (1 - g1) * (1 - g2)],
                                    [(1 - g2) * (1 + g3), -(1 + g1) * (1 + g3), (1 + g1) * (1 - g2)],
                                    [(1 + g2) * (1 + g3), (1 + g1) * (1 + g3), (1 + g1) * (1 + g2)],
                                    [-(1 + g2) * (1 + g3), (1 - g1) * (1 + g3), (1 - g1) * (1 + g2)]])

            J = np.dot(dNd_xi.T, X)
            invJ = np.linalg.inv(J)
            detJ = np.linalg.det(J)

            dNdx = np.dot(dNd_xi, invJ.T)
            F = np.eye(len(dNdx.T)) + np.dot(dNdx.T, u).T
            detF = np.linalg.det(F)

            SED = c10 * (np.trace(F.T @ F) - 3 - 2 * np.log(detF)) + d1 * (np.log(detF))**2

            # Extrapolate the SED values to the nodes
            N = 1 / 8 * np.array([(1 - g1) * (1 - g2) * (1 - g3),
                                (1 + g1) * (1 - g2) * (1 - g3),
                                (1 + g1) * (1 + g2) * (1 - g3),
                                (1 - g1) * (1 + g2) * (1 - g3),
                                (1 - g1) * (1 - g2) * (1 + g3),
                                (1 + g1) * (1 - g2) * (1 + g3),
                                (1 + g1) * (1 + g2) * (1 + g3),
                                (1 - g1) * (1 + g2) * (1 + g3)])

            m += SED * detJ * N

        return m


    def get_nodal_SED(self,u):
        if self.isRigid: return None
        sed_tot = np.zeros((len(self.X),2))

        for hexa in self.hexas:
            sed_tot[hexa,0] *= sed_tot[hexa,1]
            sed_tot[hexa,1] += 1
            sed_tot[hexa,0] += self.m_el_extra(hexa, u)
            sed_tot[hexa,0] /= sed_tot[hexa,1]

        return sed_tot[:,0]

    def fint_el_classic(self,hexa,u):
        fint_el_tens = np.zeros((8,3),dtype=float)
        X = self.X[hexa]
        u = u[self.DoFs[hexa]]

        d1 = self.Youngsmodulus*self.Poissonsratio/(2*(1+self.Poissonsratio)*(1-2*self.Poissonsratio))
        c10 = self.Youngsmodulus/(4*(1+self.Poissonsratio))

        gauss_points = 1/sqrt(3)*np.array([ [-1, -1, -1],
                                            [ 1, -1, -1],
                                            [ 1,  1, -1],
                                            [-1,  1, -1],
                                            [-1, -1,  1],
                                            [ 1, -1,  1],
                                            [ 1,  1,  1],
                                            [-1,  1,  1]])

        for (g1,g2,g3) in gauss_points:
            dNd_xi = 1/8*np.array([ [-(1-g2)*(1-g3) , -(1-g1)*(1-g3) , -(1-g1)*(1-g2) ],
                                    [ (1-g2)*(1-g3) , -(1+g1)*(1-g3) , -(1+g1)*(1-g2) ],
                                    [ (1+g2)*(1-g3) ,  (1+g1)*(1-g3) , -(1+g1)*(1+g2) ],
                                    [-(1+g2)*(1-g3) ,  (1-g1)*(1-g3) , -(1-g1)*(1+g2) ],
                                    [-(1-g2)*(1+g3) , -(1-g1)*(1+g3) ,  (1-g1)*(1-g2) ],
                                    [ (1-g2)*(1+g3) , -(1+g1)*(1+g3) ,  (1+g1)*(1-g2) ],
                                    [ (1+g2)*(1+g3) ,  (1+g1)*(1+g3) ,  (1+g1)*(1+g2) ],
                                    [-(1+g2)*(1+g3) ,  (1-g1)*(1+g3) ,  (1-g1)*(1+g2) ]])

            J = np.dot(dNd_xi.T,X)
            invJ = np.linalg.inv(J)
            detJ = np.linalg.det(J)

            dNdx = np.dot( dNd_xi , invJ.T )
            F = np.eye(len(dNdx.T)) + np.dot(dNdx.T,u).T
            detF = np.linalg.det(F)
            Finvc = np.linalg.inv(F).T

            P = 2*c10*(F-Finvc) + 2*d1*Finvc*np.log(detF)

            fint_el_tens += np.dot(dNdx,P.T)*detJ

        return np.reshape(fint_el_tens,24)
        
    def K_el_classic(self,hexa,u):
        # K_el = np.zeros((24,24))
        K_el = np.zeros((8,3,8,3))
        X = self.X[hexa]
        u = u[self.DoFs[hexa]]
        d1 = self.Youngsmodulus*self.Poissonsratio/(2*(1+self.Poissonsratio)*(1-2*self.Poissonsratio))
        c10 = self.Youngsmodulus/(4*(1+self.Poissonsratio))

        gauss_points = 1/sqrt(3)*np.array([ [-1, -1, -1],
                                            [ 1, -1, -1],
                                            [ 1,  1, -1],
                                            [-1,  1, -1],
                                            [-1, -1,  1],
                                            [ 1, -1,  1],
                                            [ 1,  1,  1],
                                            [-1,  1,  1]])

        for (g1,g2,g3) in gauss_points:
            dNd_xi = 1/8*np.array( [[-(1-g2)*(1-g3) , -(1-g1)*(1-g3) , -(1-g1)*(1-g2) ],
                                    [ (1-g2)*(1-g3) , -(1+g1)*(1-g3) , -(1+g1)*(1-g2) ],
                                    [ (1+g2)*(1-g3) ,  (1+g1)*(1-g3) , -(1+g1)*(1+g2) ],
                                    [-(1+g2)*(1-g3) ,  (1-g1)*(1-g3) , -(1-g1)*(1+g2) ],
                                    [-(1-g2)*(1+g3) , -(1-g1)*(1+g3) ,  (1-g1)*(1-g2) ],
                                    [ (1-g2)*(1+g3) , -(1+g1)*(1+g3) ,  (1+g1)*(1-g2) ],
                                    [ (1+g2)*(1+g3) ,  (1+g1)*(1+g3) ,  (1+g1)*(1+g2) ],
                                    [-(1+g2)*(1+g3) ,  (1-g1)*(1+g3) ,  (1-g1)*(1+g2) ]])

            J = np.dot(dNd_xi.T,X)
            invJ = np.linalg.inv(J)
            detJ = np.linalg.det(J)

            dNdx = np.dot( dNd_xi , invJ.T )        # <<==== J goes transposed!!!

            F = np.eye(len(dNdx.T)) + np.dot(dNdx.T,u).T
            detF = np.linalg.det(F)
            F11 = F[0,0]
            F12 = F[0,1]
            F13 = F[0,2]
            F21 = F[1,0]
            F22 = F[1,1]
            F23 = F[1,2]
            F31 = F[2,0]
            F32 = F[2,1]
            F33 = F[2,2]
            Finv11= (F22*F33 - F23*F32)/detF
            Finv12=-(F12*F33 - F13*F32)/detF
            Finv13= (F12*F23 - F13*F22)/detF
            Finv21=-(F21*F33 - F23*F31)/detF
            Finv22= (F11*F33 - F13*F31)/detF
            Finv23=-(F11*F23 - F13*F21)/detF
            Finv31= (F21*F32 - F22*F31)/detF
            Finv32=-(F11*F32 - F12*F31)/detF
            Finv33= (F11*F22 - F12*F21)/detF

            LogDetF = np.log(detF)
            PdF1111=2*c10 + 2*Finv11**2*d1 + Finv11**2*(2*c10 - 2*d1*LogDetF)
            PdF1112=2*Finv11*Finv21*(c10 + d1 - d1*LogDetF)
            PdF1113=2*Finv11*Finv31*(c10 + d1 - d1*LogDetF)
            PdF1121=2*Finv11*Finv12*(c10 + d1 - d1*LogDetF)
            PdF1122=2*Finv11*Finv22*d1 + Finv12*Finv21*(2*c10 - 2*d1*LogDetF)
            PdF1123=2*Finv11*Finv32*d1 + Finv12*Finv31*(2*c10 - 2*d1*LogDetF)
            PdF1131=2*Finv11*Finv13*(c10 + d1 - d1*LogDetF)
            PdF1132=2*Finv11*Finv23*d1 + Finv13*Finv21*(2*c10 - 2*d1*LogDetF)
            PdF1133=2*Finv11*Finv33*d1 + Finv13*Finv31*(2*c10 - 2*d1*LogDetF)
            PdF1212=2*c10 + 2*Finv21**2*d1 + Finv21**2*(2*c10 - 2*d1*LogDetF)
            PdF1213=2*Finv21*Finv31*(c10 + d1 - d1*LogDetF)
            PdF1221=2*Finv12*Finv21*d1 + Finv11*Finv22*(2*c10 - 2*d1*LogDetF)
            PdF1222=2*Finv21*Finv22*(c10 + d1 - d1*LogDetF)
            PdF1223=2*Finv21*Finv32*d1 + Finv22*Finv31*(2*c10 - 2*d1*LogDetF)
            PdF1231=2*Finv13*Finv21*d1 + Finv11*Finv23*(2*c10 - 2*d1*LogDetF)
            PdF1232=2*Finv21*Finv23*(c10 + d1 - d1*LogDetF)
            PdF1233=2*Finv21*Finv33*d1 + Finv23*Finv31*(2*c10 - 2*d1*LogDetF)
            PdF1313=2*c10 + 2*Finv31**2*d1 + Finv31**2*(2*c10 - 2*d1*LogDetF)
            PdF1321=2*Finv12*Finv31*d1 + Finv11*Finv32*(2*c10 - 2*d1*LogDetF)
            PdF1322=2*Finv22*Finv31*d1 + Finv21*Finv32*(2*c10 - 2*d1*LogDetF)
            PdF1323=2*Finv31*Finv32*(c10 + d1 - d1*LogDetF)
            PdF1331=2*Finv13*Finv31*d1 + Finv11*Finv33*(2*c10 - 2*d1*LogDetF)
            PdF1332=2*Finv23*Finv31*d1 + Finv21*Finv33*(2*c10 - 2*d1*LogDetF)
            PdF1333=2*Finv31*Finv33*(c10 + d1 - d1*LogDetF)
            PdF2121=2*c10 + 2*Finv12**2*d1 + Finv12**2*(2*c10 - 2*d1*LogDetF)
            PdF2122=2*Finv12*Finv22*(c10 + d1 - d1*LogDetF)
            PdF2123=2*Finv12*Finv32*(c10 + d1 - d1*LogDetF)
            PdF2131=2*Finv12*Finv13*(c10 + d1 - d1*LogDetF)
            PdF2132=2*Finv12*Finv23*d1 + Finv13*Finv22*(2*c10 - 2*d1*LogDetF)
            PdF2133=2*Finv12*Finv33*d1 + Finv13*Finv32*(2*c10 - 2*d1*LogDetF)
            PdF2222=2*c10 + 2*Finv22**2*d1 + Finv22**2*(2*c10 - 2*d1*LogDetF)
            PdF2223=2*Finv22*Finv32*(c10 + d1 - d1*LogDetF)
            PdF2231=2*Finv13*Finv22*d1 + Finv12*Finv23*(2*c10 - 2*d1*LogDetF)
            PdF2232=2*Finv22*Finv23*(c10 + d1 - d1*LogDetF)
            PdF2233=2*Finv22*Finv33*d1 + Finv23*Finv32*(2*c10 - 2*d1*LogDetF)
            PdF2323=2*c10 + 2*Finv32**2*d1 + Finv32**2*(2*c10 - 2*d1*LogDetF)
            PdF2331=2*Finv13*Finv32*d1 + Finv12*Finv33*(2*c10 - 2*d1*LogDetF)
            PdF2332=2*Finv23*Finv32*d1 + Finv22*Finv33*(2*c10 - 2*d1*LogDetF)
            PdF2333=2*Finv32*Finv33*(c10 + d1 - d1*LogDetF)
            PdF3131=2*c10 + 2*Finv13**2*d1 + Finv13**2*(2*c10 - 2*d1*LogDetF)
            PdF3132=2*Finv13*Finv23*(c10 + d1 - d1*LogDetF)
            PdF3133=2*Finv13*Finv33*(c10 + d1 - d1*LogDetF)
            PdF3232=2*c10 + 2*Finv23**2*d1 + Finv23**2*(2*c10 - 2*d1*LogDetF)
            PdF3233=2*Finv23*Finv33*(c10 + d1 - d1*LogDetF)
            PdF3333=2*c10 + 2*Finv33**2*d1 + Finv33**2*(2*c10 - 2*d1*LogDetF)

            PdF = np.zeros((3,3,3,3))
            for ij in range(9):
                i, j = divmod(ij,3) # returns quotient and mod
                PdF[i,j,i,j] = locals()["PdF"+str(i+1)+str(j+1)+str(i+1)+str(j+1)] # "diagonal term"
                for kl in range(ij+1,9):
                    k, l = divmod(kl,3)

                    PdF[i,j,k,l] = PdF[k,l,i,j] = locals()["PdF"+str(i+1)+str(j+1)+str(k+1)+str(l+1)]

            # Kgp = np.tensordot(dNdx,np.tensordot(PdF.swapaxes(0,1),dNdx.T,axes=1),axes=1).swapaxes(2,3).reshape(24,24)
            Kgp = np.tensordot(dNdx,np.tensordot(PdF.swapaxes(0,1),dNdx.T,axes=1),axes=1).swapaxes(2,3)

            # K_el+=Kgp*detJ
            K_el+=Kgp*detJ

        return K_el.reshape(24,24)

    # Residuals at FEAssembly level
    # def get_K(self, Model,NowDiff=False):

    #     return eigen_backend.K_global_alt(Model.X,Model.u,self.hexas,self.DoFs,self.Youngsmodulus,self.Poissonsratio)


    def get_K_Cem(self, Model,NowDiff=False):
        if self.isRigid:
            return None

        if not hasattr(self,"SparseRow"):
            RC = eigen_backend.K_global_cr(Model.X,Model.u_temp,self.hexas,self.DoFs,self.Youngsmodulus,self.Poissonsratio)
            vals = eigen_backend.K_global_cr(Model.X,Model.u_temp,self.hexas,self.DoFs,self.Youngsmodulus,self.Poissonsratio)
            self.SparseRow = RC.rows
            self.SparseCol = RC.cols

        vals = eigen_backend.K_global_val(Model.X,Model.u_temp,self.hexas,self.DoFs,self.Youngsmodulus,self.Poissonsratio)
        Ksparse = sparse.csr_matrix((vals,(self.SparseRow,self.SparseCol)),shape=Model.K.shape)

        Model.K += Ksparse


    def get_K(self, Model,NowDiff=False):
        if self.isRigid:
            return None

        computeRC = False
        if not hasattr(self,"SparseRow"):
            self.SparseRow = np.empty(len(self.hexas)*576,dtype=int  )
            self.SparseCol = np.empty(len(self.hexas)*576,dtype=int  )
            computeRC = True

        V = np.empty(len(self.hexas)*576,dtype=float)
        for i,hexa in enumerate(self.hexas):
            V[i*576:(i+1)*576] = self.K_el(hexa, Model.u_temp).ravel()
            if computeRC:
                dofs = self.DoFs[hexa].ravel()
                self.SparseRow[i*576:(i+1)*576] = np.repeat(dofs,len(dofs))
                self.SparseCol[i*576:(i+1)*576] = np.  tile(dofs,len(dofs))
        Kbody = sparse.csr_matrix((V,(self.SparseRow,self.SparseCol)),shape=Model.K.shape)

        if NowDiff:
            return Kbody
        else:
            Model.K += Kbody

    def get_fint(self, Model,temp=False):
        if self.isRigid:
            return None

        u = Model.u_temp if temp else Model.u
        for hexa in self.hexas:
            fint_el = self.fint_el(hexa,u)
            dofs = self.DoFs[hexa].ravel()
            Model.fint[np.ix_(dofs)] += fint_el


    # functions for BFGS
    def compute_m(self,u):
        if self.isRigid:
            return 0

        m = 0
        for hexa in self.hexas:
            m += self.m_el(hexa,u)

        return m
    
    def compute_f(self,u, Model):
        force=np.zeros(Model.fint.shape)
        if self.isRigid:
            return 0

        for hexa in self.hexas:
            fint_el = self.fint_el(hexa,u)
            dofs = self.DoFs[hexa].ravel()
            force[dofs] += fint_el
        return force


    # VISUAL
    def plot(self,ax = None, u = None,plotWhat = "surf",nodes="all",plotcoords = False, sed = None,ref=10):

        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        if plotcoords: 
            from PyClasses.Utilities import plot_coords
            plot_coords(ax)
        if plotWhat in ["nodes","all"]:
            for idx,x in enumerate(self.X):
                if nodes=="all" or idx in nodes:
                    ax.scatter(x[0],x[1],x[2], c="blue", s = 0.2)
        if plotWhat in ["surf","all"]:
            if u is None:
                u = np.zeros_like(self.X)
            surfObj = self.surf.plot(ax,u, sed=sed,ref=ref)

        return surfObj


def K_hexa_glob(self,Model,hexa):
    dofs = self.DoFs[hexa].ravel()
    vals = self.K_el(hexa,Model.u_temp).ravel()     # ,-- "original"
    r = np.repeat(dofs,len(dofs))
    c = np.  tile(dofs,len(dofs))
    return vals, r ,c

