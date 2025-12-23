from PyClasses.BoundaryConditions import *
from PyClasses.Utilities import float_to_fraction, printif, checkVarsSize, flatList
from scipy import sparse
from scipy.sparse.linalg import spsolve, cgs, bicg, gmres, lsqr
from scipy.linalg import solve as slsolve
import matplotlib.pyplot as plt
import numpy as np
import time, pickle, sys, os, csv

# import petsc4py
# petsc4py.init(sys.argv)
# from petsc4py import PETSc

from pdb import set_trace

class FEModel:
    def __init__(self,bodies, contacts, BoundaryConditions, UpdateOnIteration = True):
        self.Mainname = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.bodies = bodies
        self.contacts = contacts
        self.UOI = UpdateOnIteration
        
        X , DoFs, n0 = [], [], 0
        for bid, body in enumerate(bodies):
            X.extend(body.X)
            if not hasattr(self, "dim"):
                self.dim = len(X[0])
            nf = n0 + len(body.X)
            body.DoFs = np.array( [ [ self.dim*n + i for i in range(self.dim) ] for n in range(n0,nf) ] )
            body.id = bid
            # body.surf.DoFs = body.DoFs[body.surf.nodes]
            for sub_surf in body.surf.sub_surfs:
                # sub_surf.DoFs =  body.DoFs[sub_surf.nodes]
                sub_surf.DoFs =  body.DoFs
            body.surf.id = bid
            n0 = nf

        X = np.array( X, dtype=float)
        self.PlotBoundaries = [[min(X[:,0]), max(X[:,0])],
                               [min(X[:,1]), max(X[:,1])],
                               [min(X[:,2]), max(X[:,2])]]
        self.X = X.ravel()
        self.ndof = len(self.X)

        self.u = np.zeros_like(self.X,dtype=float)
        self.u_temp = np.array(self.u)
        self.fint = np.zeros_like(self.u,dtype=float)
        self.fext = np.zeros_like(self.u,dtype=float)

        self.BCs = InitializeBCs(BoundaryConditions)    # Creates BC objects with their attributes (dofs, vals, times, etc)

    def printContactStates(self):
        for i_c, contact in enumerate(self.contacts):
            print("contact",i_c,":")
            contact.printStates()

    def plot(self, ax, ref = 10, specialPatches = None, almostSpecialPatches = None, specialNodes = None, time = None, fintC = False, plotHooks=False):
        for bid, body in enumerate(self.bodies):
            if body in [contact.slaveBody for contact in self.contacts]:
                body.surf.plot(ax, self.u, specialNodes = specialNodes,ref=ref)
            elif body in [contact.masterBody for contact in self.contacts]:
                body.surf.plot(ax, self.u, specialPatches = specialPatches, almostSpecialPatches = almostSpecialPatches,ref=ref)
            else:
                body.surf.plot(ax, self.u,ref=ref)

        if fintC:
            for contact in self.contacts: contact.plotForces(self.u, ax)
        if plotHooks:
            for contact in self.contacts: contact.plotHooks(ax)


        if time != None:
            ax.text2D(0.05, 0.95, "time: "+str(round(time,6))+" s", transform=ax.transAxes)

    def plotNow(self, fintC = False):
        #PLOT OPTIONS
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        (minx,maxx),(miny,maxy),(minz,maxz) = self.PlotBoundaries
        hg = max([dij[1]-dij[0] for dij in self.PlotBoundaries])/2    #half-gap to be used in every direction
        cx,cy,cz = (minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2

        ax.set_xlim3d((cx-hg, cx+hg))
        ax.set_ylim3d((cy-hg, cy+hg))
        ax.set_zlim3d((cz-0.8*hg, cz+0.8*hg))

        candidatePatches = sorted(set(flatList(self.contacts[0].candids))-set(self.contacts[0].actives))
        activeNodes = [self.contacts[0].slaveNodes[idx] if self.contacts[0].proj[idx,3]!=0 else None for idx in range(self.contacts[0].nsn)]

        # self.plot(ax, ref = 4,specialPatches = [(0,1,0,0.7),self.contacts[0].actives],almostSpecialPatches = [(1,0.6,0.0,0.7),candidatePatches],specialNodes = ["blue", activeNodes], fintC= fintC)
        self.plot(ax, ref = 4,specialPatches = None,almostSpecialPatches = None,specialNodes = {"blue": activeNodes}, fintC= fintC)
        plt.show()

    def plotContactNow(self, ctct_id=0, labels =True, block = True,SlaveQuads=False, context = False):
        """Enable this to visualize contact state"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.contacts[ctct_id].plotContact(self.u_temp,ax,labels=labels,SlaveQuads=SlaveQuads, context=context)
        self.contacts[ctct_id].plotHooks(ax)
        ax.set_zlim3d(-2, 2)
        plt.show(block = block)

    def savefig(self, ti,  elevation=[ 30 , 30], azimut=[-135 , -135], distance = [10, 10], times = None, iterative = None, dpi = 100, fintC = False, Hooks= False):
        stp = str(self.ntstps)
        kn = str(self.contacts[0].kn) if len(self.contacts)!= 0 else "NA"
        ct = str(self.contacts[0].cubicT) if len(self.contacts)!= 0 else "NA"
        maxGN = str(self.contacts[0].maxGN) if len(self.contacts)!= 0 else "NA"

        dir = self.Mainname+" cT"+ct+" "+stp+"stp kn"+kn+"maxGN"+maxGN

        if not os.path.exists(dir):
            os.mkdir(dir)

        #PLOT OPTIONS
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #for good appeareance try to keep a ratio  6range(x)=6range(y)=5range(z)
        (minx,maxx),(miny,maxy),(minz,maxz) = self.PlotBoundaries
        hg = max([dij[1]-dij[0] for dij in self.PlotBoundaries])/2    #half-gap to be used in every direction
        cx,cy,cz = (minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2

        ax.set_xlim3d((cx-hg, cx+hg))
        ax.set_ylim3d((cy-hg, cy+hg))
        ax.set_zlim3d((cz-0.8*hg, cz+0.8*hg))

        # self.plot(ax, ref = 4,specialPatches = [(0,1,0,0.7),self.contacts[0].activePatches],specialNodes = ["blue", self.contacts[0].activeNodes], time = time)
        AlmostImportant = sorted(set(flatList(self.contacts[0].candids))-set(self.contacts[0].actives)) if len(self.contacts)!= 0 else []

        if times!= None:
            t0,t,tf = times
        else:
            t = None

        ctct = self.contacts[0]
        # activeNodes = [ctct.slaveNodes[idx] for idx in ctct.proj[:,3].nonzero()[0]]
        idxNodes = ctct.proj[:,3].nonzero()[0]
        stickNodes = [ctct.slaveNodes[idx] for idx in idxNodes if ctct.Stick[idx] ]
        slipNodes = [ctct.slaveNodes[idx] for idx in idxNodes if not ctct.Stick[idx] ]
        # set_trace()
        # print("see how this works to create a list of stick and other list of slip nodes")
        # print("then use special nodes as a dict:     specialNodes={'blue':stickNodes,'red':slipNodes}")
        # set_trace()
        activePatches = [pr[0] for pr in ctct.proj if pr[3]!=0]
        self.plot(ax, ref = 4,specialPatches = [(0,1,0,0.75),activePatches],
                              almostSpecialPatches = [(1.0,0.64,0.0,0.75), AlmostImportant],
                            #   specialNodes = ["blue", activeNodes], 
                              specialNodes = {"blue": stickNodes,"red":slipNodes}, 
                              time = t,
                              fintC = fintC,
                              plotHooks = Hooks)

        #Camera effects
        tt = (t-t0)/(tf-t0)
        ax.view_init(elev=elevation[0]*(1-tt)+elevation[1]*tt,azim=azimut[0]*(1-tt)+azimut[1]*tt)
        ax.dist = distance[0]*(1-tt)+distance[1]*tt     #default dist = 10

        if iterative==None:
            plt.savefig(dir+"/fig"+str(ti)+".png",dpi = dpi)
        else:
            plt.savefig(dir+"/fig"+str(ti)+"-"+str(iterative)+".png",dpi = dpi)
        plt.close()

    def savedata(self,time,*args, dofs = "all", name = None, Sum=False):
        for arg in args:
            filename = self.Mainname+"_"+(arg if name is None else name)+".csv"
    
            # writing to csv file
            with open(filename, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
                
                # writing the fields
                if type(dofs)==list or type(dofs)==np.ndarray:
                    if Sum:
                        csvwriter.writerow([time,sum(getattr(self,arg)[dofs])])
                    else:
                        csvwriter.writerow([time]+(getattr(self,arg)[dofs]).tolist())
                elif type(dofs)==str:
                    if dofs == "all": csvwriter.writerow([time]+getattr(self,arg).tolist())
                    else: print("Unrecognized datatype to be stored from string. Not saving this data"), set_trace()
                else:
                    print("Unrecognized datatype to be stored. Not saving this data"), set_trace()
    
    def saveNodalData(self,body_id,time,*args, nodes = "all", name = None, Sum=False, tracing = False):
    
        if type(nodes)==list or type(nodes)==np.ndarray:
            dofs = self.bodies[body_id].DoFs[nodes]
        elif nodes == "all":
            dofs = self.bodies[body_id].DoFs
        for arg in args:
            filename = self.Mainname+"_"+(arg if name is None else name)+".csv"
    
            if tracing: set_trace()

            # writing to csv file
            with open(filename, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
                # writing the fields
                if Sum:
                    sumX = sum(getattr(self,arg)[dofs.T[0]])
                    sumY = sum(getattr(self,arg)[dofs.T[1]])
                    sumZ = sum(getattr(self,arg)[dofs.T[2]])
                    csvwriter.writerow([time,sumX,sumY,sumZ])
                else:
                    csvwriter.writerow([time]+(getattr(self,arg)[dofs]).tolist())

    def saveContactData(self,ctct_id,time, slaveNodes = None):
        #TODO: change all this code. Now possibly store just hook+proj (?)

        # import pandas as pd
        # filename = self.Mainname+"_ctctState"+str(ctct_id)+".xslx"

        # ctct = self.contacts[ctct_id]
        # slaveNodes = ctct.slaveNodes if slaveNodes is None else slaveNodes
        # ftrs_labels = ["patch", "tx", "ty", "gn", "kn"]
        # ftrs = [getattr(ctct,attr) for attr in ["proj[0]","proj[1]","proj[2]","proj[3]","kn"]]
        # if ctct.mu!=0:  
        #     ftrs.extend( [getattr(ctct,attr) for attr in ["hook[0]","hook[1]","hook[2]","hook[3]","Stick"]] )
        #     ftrs_labels.extend( ["Hpatch", "Htx", "Hty", "gn0", "Slip?"] )
        # header, tabs = slaveNodes,ftrs_labels

        # writer = pd.ExcelWriter(filename, engine='xlsxwriter')

        # for tab in tabs:

        #     time_col = pd.DataFrame()

        ctct = self.contacts[ctct_id]
        stringthings = ["gn", "kn", "gt"]
        slaveNodes = range(ctct.nsn) if slaveNodes is None else slaveNodes

        def states(idx):
            return [ctct.proj[idx,3], ctct.alpha_p[idx]*ctct.kn, ctct.hook[idx,3]]
        

        for stidx, stringthing in enumerate(stringthings):
            filename = self.Mainname+"_ctct"+str(ctct_id)+"_"+stringthing+".csv"
    
            # writing to csv file
            with open(filename, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
                
                # writing the fields
                csvwriter.writerow([time]+[states(idx)[stidx] for idx in slaveNodes])
        # pass

    def applyBCs(self,t,tpre):

        print("Applying BCs from: ",round(tpre,6)," to: ",round(t,6))
        diri = []
        for bc in self.BCs:
            if bc.t0 <= t <= bc.tf or bc.t0 <= tpre <= bc.tf:
                # load factor
                if tpre < bc.t0:          # just entered in time interval
                    LF = (t - bc.t0) / (bc.tf - bc.t0)
                elif t > bc.tf:            # just exited time interval
                    LF = (bc.tf - tpre) / (bc.tf - bc.t0)
                else:                   # going through time interval (normal case)
                    LF = (t - tpre ) / (bc.tf - bc.t0)

                if bc.BCtype == "dirichlet":          # Dirichlet
                    diri.extend(bc.DoFs)
                    if LF!=0:
                        for dof in bc.DoFs:
                            self.u[dof] += bc.vals[dof%self.dim]*LF
                elif LF!=0:                           # Neumann
                    for dof in bc.DoFs:
                        self.fext[dof] += bc.vals[dof%self.dim]*LF

        di = list(set(diri))      #Gets rid of eventually repeated DoFs
        self.diri = di
        self.free = np.delete(np.arange(len(self.X)), di)       # free DOFs 'b'

    def IsContactAssumptionCorrect(self):
        set_trace()
        for contact in self.contacts:

            if len(contact.activePairs)!=len(contact.GNs):
                return False

            for pair, gn in zip(contact.candidatePairs,contact.GNs):
                if ((gn<0) != (pair in contact.activePairs)): return False
        return True

    def setReferences(self):
        u_ref = np.array(self.u)
        fint_ref = np.array(self.fint)
        alpha_ref = list([contact.alpha_p for contact in self.contacts])
        new_ref   = list([contact.new for contact in self.contacts])
        xs_ref = list([contact.xs for contact in self.contacts])
        act_ref = list([contact.actives for contact in self.contacts])

        self.REF = [u_ref, fint_ref, alpha_ref, new_ref, xs_ref,act_ref]

    def getReferences(self, alphas=True,actives=False):
        self.u = np.array(self.REF[0])
        self.fint = np.array(self.REF[1])
        alpha_ref, new_ref, xs_ref,act_ref = self.REF[2:]
        self.u_temp = np.array(self.u)
        for icont, contact in enumerate(self.contacts):
            if alphas: contact.alpha_p = alpha_ref[icont]
            contact.new  = new_ref[icont]
            contact.xs   = xs_ref[icont]
            if actives: contact.actives = act_ref[icont]

    def get_fint(self, DispTime = False):
        t0_fint = time.time()
        self.fint = np.zeros_like(self.u,dtype=float)
        for body in self.bodies: body.get_fint(self)
        if DispTime: print("Getting fint : ",time.time()-t0_fint,"s")


        for contact in self.contacts:
            contact.getfintC(self, DispTime=DispTime)      # uses model.u


    def get_K(self, DispTime = False):
        t0_K = time.time()
        self.K = sparse.coo_matrix((self.ndof,self.ndof),dtype=float)
        for body in self.bodies:
            body.get_K(self)    # uses model.u_temp

        printif(DispTime,"Getting K : ",time.time()-t0_K,"s")

        for contact in self.contacts:
            contact.getKC(self, DispTime=DispTime)     #uses model.u_temp


    def residual(self, printRes = False):
        RES = np.linalg.norm(self.fint - self.fext)
        if printRes: print("resid :",RES)
        return RES


    def Solve(self, t0 = 0, tf = 1, TimeSteps = 1, max_iter=10 , recover = False, ForcedShift = False):
        self.ntstps = TimeSteps; TimeDisp=True
        
        dt_base = (tf-t0)/TimeSteps; tolerance = 1e-10; ndof = len(self.X)
        t = t0 ; ti = 0
        dt = dt_base; u_ref = np.array(self.u)
        pickle.dump({body.name:[body.surf.quads,body.surf.nodes] for body in self.bodies},open("RecoveryOuters.dat","wb"))
        
        self.setReferences()
        skipBCs = False
        tmp = 0

        tracing = False

        if recover:
            self.REF,t, dt, ti = pickle.load(open("RecoveryData.dat","rb"))
            self.getReferences(actives=True)
            for contact in self.contacts:   contact.getCandidates(self.u)

            if ForcedShift:
                rem = t%dt_base     # remaining time from previous time increment
                t_pre = t-rem
                dt = t_pre+dt_base-t
                ForcedShift = False

        # for contact in self.contacts:   contact.getCandidates(self.u)

        stop = True


        while t+dt < tf+1e-4:
            print(" ")
            t += dt
            
            print("time: ",round(t,6),"s  -----------------------------------")

            if not skipBCs:
                self.applyBCs(t,t-dt)
            else:
                skipBCs = False

            di, fr = self.diri, self.free

            

            # NEWTON-RAPHSON
            RES, niter, res_pre = 1+tolerance, 0, 0.0
            while RES>tolerance:
                # getting K and KC
                # if t>0.1399:set_trace()
                self.get_K(DispTime=TimeDisp)  # <-- uses self.u_temp
                    
                # Linear System
                t0_lin = time.time()
                dua = self.u[di] - self.u_temp[di]      # difference due to dirichlet BCs applied

                Kaa=self.K[np.ix_(di,di)]
                Kab=self.K[np.ix_(di,fr)]
                Kba=self.K[np.ix_(fr,di)]
                Kbb=self.K[np.ix_(fr,fr)]

                fina=self.fint[di]
                finb=self.fint[fr]
                fexb=self.fext[fr]      # always zero (?)

                # # 1. Iterative Solver, P is preconditioner
                # P = sparse.spdiags([1./Kbb.diagonal()],0,Kbb.shape[0], Kbb.shape[1])
                # dub, _ = bicg(Kbb,fexb-finb-Kba.dot(dua), maxiter = 200,tol=1e-10,M=P)     # Uses all available processors

                # 2.  Direct Solver (can be very slow for big matrices)
                dub =spsolve(Kbb,fexb-finb-Kba.dot(dua))     # Uses all available processors

                """
                    # 3.  Direct Solver, with PETSc (needs to be installed)
                    # bside = PETSc.Vec().createWithArray(fexb-finb-Kba.dot(dua))
                    # dubP = bside.copy()
                    # KbbPepsi = scipy_to_petsc(Kbb) # import function from PyClasses.Utilities
                    # ksp = PETSc.KSP().create()
                    # ksp.setType('preonly')
                    # ksp.getPC().setType("lu")
                    # ksp.getPC().setFactorSolverType("mumps")
                    # # ksp.getPC().setFactorSolverType("umfpack")
                    # ksp.setOperators(KbbPepsi)
                    # ksp.setFromOptions()
                    # ksp.solve(bside,dubP)
                    # dub = dubP.array

                """

                self.fext[di] = fina + Kaa.dot(dua) + Kab.dot(dub)
                info = None
                printif(TimeDisp,"Solving Linear System: ",time.time()-t0_lin,"s", "" if info is None else "with %s iterations"%info)

                self.u[fr] += dub

                for ctct in self.contacts:
                    sDoFs  = ctct.slaveBody.DoFs[ctct.slaveNodes]
                    xs = np.array(ctct.slaveBody.X )[ctct.slaveNodes ] + np.array(self.u[sDoFs ])
                    ctct.xs = xs
                    # ctct.updateActive()


                self.printContactStates()

                
                self.get_fint(DispTime=TimeDisp)   # uses self.u


                RES = self.residual(printRes=True)

                # self.plotContactNow()
                # if abs(t-0.46)<1e-5:
                #     self.plotContactNow()
                #     set_trace()

                niter += 1
                if (niter == max_iter and RES>tolerance) or np.isnan(RES) or RES>1e+15:
                    print('Increment failed to converge !!!!! Redoing half')

                    t -= dt
                    self.getReferences(actives=True)    # resets u=u_temp <- u_ref
                    dt = dt/2
                    tmp += 1

                    break

                self.u_temp = np.array(self.u)

            else:
                #############################
                #### CONVERGED INCREMENT ####
                #############################
                keepactives = True
                keepalphas = True
                Redo = False
                for ctct in self.contacts:
                    ctct.actives_prev = list(ctct.actives)
                    ctct.getCandidates(self.u, CheckActive = True, TimeDisp=False, tracing=tracing)    # updates Patches->BSs (always) -> candidatePairs (on choice)
                    if ctct.actives_prev!=ctct.actives: 
                        Redo = True
                        keepactives = False
                
                self.printContactStates()
                
                if not Redo and keepactives:
                    for ctct in self.contacts:
                        if ctct.checkGNs(): Redo = True
                        keepalphas = False


                if Redo:
                    t-=dt
                    # skipBCs=True
                    self.getReferences(alphas=keepalphas,actives=keepactives)
                    # set_trace()
                    tmp += 1
                    continue

                ti += 1

                print("##################")
                self.savefig(ti,azimut=[-60, -60],elevation=[20,20],distance=[10,10],times=[t0,t,tf],fintC=False,Hooks=True)


                # for ctct in self.contacts:  ctct.UpdateHooks()
                
                self.setReferences()      

                sBody = self.contacts[0].slaveBody
                self.saveNodalData(sBody.id, t, "fint", nodes = sBody.SelectFlatSide("+z"), name="fint", Sum=True)
                self.saveNodalData(sBody.id, t, "u", nodes = [24], name="dispSlave", Sum=True)

                self.saveContactData(0,t)

                # set_trace()
                if dt != dt_base:
                    ratio = t/dt_base - int((t-t0)/dt_base)
                    dt = dt_base/(float_to_fraction(ratio)[1])

                # saving:
                pickle.dump([self.REF,t, dt, ti],open("RecoveryData.dat","wb"))

        else:
            print(self.u)
            print("Increments Redone : ",tmp)

