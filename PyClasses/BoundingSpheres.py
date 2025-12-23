from PyClasses.GregoryPatches import *
from PyClasses.Utilities import *
from pdb import set_trace


from ctypes import * # Library to import shared object

import numpy as np
from numpy.linalg import norm
from math import sqrt

# Import C++ backend at module level to reduce per-call overhead
from . import gregory_patch_backend



class BS:
    def __init__(self, element, method = "avgXmaxR", eps = 0 ):
        
        self.element = element
        self.method = method
        self.eps = eps

        if type(element) in [list,np.ndarray]:
            if type(element[0]) == BS:
                x , r = self.wrapBSs(element)
            elif type(element[0]) in [np.ndarray, list] and len(element[0]) == 3:   #CHANGE! not necessarily the first element. There are many 'None'
                x , r = self.wrapNodes(element, method)

            
            elif type(element[0]) == GrgPatch:      #CHANGE! not necessarily the first element. There are many 'None'
                print("NOT YET IMPLEMENTED! CHECK!")
        elif type(element) == GrgPatch:
            x , r = self.wrapPatch(element, method)
        elif element == None:
            x, r = None, None


        self.x = x
        self.r = r

    def wrapBSs(self, BSs):
        xmin, ymin, zmin = BSs[0].x       #just any known point in the sphere
        xmax, ymax, zmax = BSs[0].x
        for BS in BSs:
            if (BS.r!=None):
                xi,ri = BS.x , BS.r
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
        return xm , r*(1+self.eps)

    def wrapNodes(self, nodes, method):
        # TODO: (TO TRY) EPOS algorithm

        if method == "minmax":
            xmin, ymin, zmin = nodes[0]
            xmax, ymax, zmax = nodes[0]
            
            for pt in nodes:
                if pt[0]<xmin:
                    xmin = pt[0]
                elif pt[0]>xmax:
                    xmax = pt[0]
                if pt[1]<ymin:
                    ymin = pt[1]
                elif pt[1]>ymax:
                    ymax = pt[1]
                if pt[2]<zmin:
                    zmin = pt[2]
                elif pt[2]>zmax:
                    zmax = pt[2]

            xm = 0.5*np.array( [ xmin+xmax , ymin+ymax, zmin+zmax ] )
            r0 = 0.5*sqrt( (xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2 )
        
        elif method == "avgXmaxR":
            # xm = sum(nodes)/len(nodes)
            # diffs = nodes - xm*np.ones_like(nodes)
            # r0 = max([norm(diffs[ii]) for ii in range(len(nodes))])

            
            # c_points = (c_double * len(nodes[3]) * len(nodes))()  # Allocating memory for 2D array in C
            # for row in range(len(nodes)):
            #     for col in range(3):
            #         c_points[row][col] = nodes[row][col] # Adding each element of the Python array to the C one
            # # average method for C
            # xm = (c_double * 3)(0, 0, 0)
            # # func.avgCenter(c_points, xm)
            # # r0 = func.getMaxRadius(c_points, xm)
            # set_trace()

            # func.avgCenter(c_points, xm)
            # r0 = func.getMaxRadius(c_points, xm)

            xm = np.mean(nodes,axis=0)
            r0 = max([norm(node-xm) for node in nodes])



        else:
            print("Unrecognized method!!!!!!!!!!!!!!!!!!!!!!!")

        return np.array(xm), r0*(1+self.eps)

    def wrapPatches(self, patches, method):
        t0 = time.time()
        CTRLPTS = flatList([patch.CtrlPts for patch in patches])
        return self.wrapNodes(CTRLPTS, method)
        tf = time.time()
        print("wrapping ",len(patches), " patches in ",round(tf-t0,4), " seconds.")

    def wrapPatch(self, patch, method):
        ctrlpts = flatList(patch.CtrlPts)
        return self.wrapNodes(ctrlpts, method)

        # waiting for Quentin's answer
        # c_points = (c_double * len(ctrlpts[3]) * len(ctrlpts))()  # Allocating memory for 2D array in C
        # # average method for C
        # sphereCenter = (c_double * 3)(0, 0, 0)
        # func.avgCenter(c_points, sphereCenter)
        # Rad = func.getMaxRadius(c_points, sphereCenter)
        # return sphereCenter, Rad

    def ContainsNode(self,xp):
        # Use C++ backend for speed while maintaining exact logic: norm(xp-self.x) <= self.r
        # Import moved to module level to reduce per-call overhead
        return gregory_patch_backend.ContainsNode(self.x, self.r, xp)

    def CollidesWithBS(self,BS2, res = "bool"):
        if res == "bool":
            return norm(BS2.x-self.x) < (self.r + BS2.r)
        elif res == "value":
            return norm(BS2.x-self.x) - (BS2.r + self.r)

    def plot(self,ax,ref=20,color='b',alpha=0.25):
        # Create a transparent sphere
        r,(x0,y0,z0) = self.r,self.x  # Radius of the sphere
        phi, theta = np.mgrid[0.0:2.0*np.pi:ref*1j, 0.0:np.pi:ref*0.5j]
        x = x0+r * np.sin(theta) * np.cos(phi)
        y = y0+r * np.sin(theta) * np.sin(phi)
        z = z0+r * np.cos(theta)

        ax.plot_surface(x, y, z, color=color, alpha=alpha)