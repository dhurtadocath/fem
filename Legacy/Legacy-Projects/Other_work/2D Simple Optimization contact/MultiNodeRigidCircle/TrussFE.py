import numpy as np
from numpy.linalg import norm
from math import sqrt



class TrussFE:
    def __init__(self,X,C,name = "unnamed"):
        self.X = X
        self.C = C
        self.dim = len(X[0])
        self.name = name
        self.DoFs = np.empty_like(X, dtype=int)
        self.E = 1
        self.A = 0.3
        self.isRigid = False

    def Energy(self, u):
        de2 = 0.0
        for (i,j) in self.C:
            Xa, Xb = self.X[np.ix_([i,j])]
            ua, ub = self.u[self.DoFs[np.ix_([i,j])]]
            xa, xb = Xa+ua , Xb+ub
            L0 = norm(Xa-Xb)
            L = norm(xa-xb)

            de2 += (L-L0)**2/L0
        return 0.5*self.E*self.A*de2

    def fint_el(elid):
        