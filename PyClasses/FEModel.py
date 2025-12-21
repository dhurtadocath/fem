from PyClasses.BoundaryConditions import *
from PyClasses.Utilities import float_to_fraction, printif,brents_method, checkVarsSize, flatList,plot_coords,quadratic_fit_min_zeros
import PyClasses.FEAssembly
from scipy import sparse
from scipy.sparse.linalg import spsolve, cgs, bicg, gmres, lsqr
from scipy.linalg import solve as slsolve
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import time, pickle, sys, os, csv

# import petsc4py
# petsc4py.init(sys.argv)
# from petsc4py import PETSc

from pdb import set_trace

class FEModel:
    def __init__(self,bodies, contacts, BoundaryConditions, UpdateOnIteration = True,transform_2d=None,subname=""):
        self.Mainname = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.bodies = bodies
        self.contacts = contacts
        self.UOI = UpdateOnIteration
        self.transform_2d = transform_2d

        # main counter during the simulation
        self.COUNTS_NAMES = ["incr_accptd", "incr_rjctd", "NR_iters", "incr_mnmzd", "mnmzn_iters", "mnmzn_fn_evals","CG_iters"]
        self.COUNTS = np.zeros((7,),dtype = int)

        # main timer during the simulation
        # self.TIMERS_NAMES = ["Total", "ActSet_Updt", "fcon","fint","Ktot+NRlinsolv","Ktot+MINsolv"]
        self.TIMERS_NAMES = ["Total", "Ktot+NRlinsolv", "Ktot+MINsolv", "fint", "fcon","ActSet_Updt"]
        self.TIMERS = np.zeros((6,),dtype = float)

        self.SED = [[] for _ in range(len(bodies))]     # stores nodal Strain-energy density for every successful increment
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


        current_time=time.strftime("%Y%m%d%H%M", time.localtime())
        dir =  "OUTPUT_"+current_time+self.Mainname+subname+"/"
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.output_dir = dir
        pickle.dump(self,open(dir+"Model.dat","wb"))


    def printContactStates(self,only_actives=True,veredict=False):
        for i_c, contact in enumerate(self.contacts):
            print("contact",i_c,":")
            contact.printStates(only_actives=only_actives,veredict=veredict)

    def plot(self, ax, ref = 10, specialPatches = None, almostSpecialPatches = None,
              specialNodes = None, text = None, fintC = False, plotHooks=False,OnlyMasterSurf=False, u=None):
        
        if u is None:
            u = self.u

        for body in self.bodies:
            if body in [contact.slaveBody for contact in self.contacts]:
                # body.surf.plot(ax, u, specialNodes = specialNodes,ref=ref)
                body.surf.plot(ax, u, specialNodes = specialNodes,ref=1)
            elif body in [contact.masterBody for contact in self.contacts]:
                body.surf.plot(ax, u, specialPatches = specialPatches, almostSpecialPatches = almostSpecialPatches,ref=ref,onlyMaster=OnlyMasterSurf)
            else:
                body.surf.plot(ax, u,ref=ref)

        if fintC:
            for contact in self.contacts: contact.plotForces(u, ax)
        if plotHooks:
            for contact in self.contacts: contact.plotHooks(ax)

        if text is not None:
            ax.text2D(0.17, 0.7, text, transform=ax.transAxes)

    def plotNow(self, fintC = False, as2D= False,OnlyMasterSurf=False):
        #PLOT OPTIONS
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('ortho')
        # plt.axis('off')

        if as2D: ax.view_init(elev=0, azim=-90, roll=0)

        (minx,maxx),(miny,maxy),(minz,maxz) = self.PlotBoundaries
        hg = max([dij[1]-dij[0] for dij in self.PlotBoundaries])/2    #half-gap to be used in every direction
        cx,cy,cz = (minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2
        
        plot_coords(ax,orig=(minx-0.5,miny-0.5,minz-0.5))

        ax.set_xlim3d((cx-hg, cx+hg))
        ax.set_ylim3d((cy-hg, cy+hg))
        ax.set_zlim3d((cz-0.8*hg, cz+0.8*hg))

        # candidatePatches = sorted(set(flatList(self.contacts[0].candids))-set(self.contacts[0].actives))
        # activeNodes = [self.contacts[0].slaveNodes[idx] if self.contacts[0].proj[idx,3]!=0 else None for idx in range(self.contacts[0].nsn)]
        activeNodes = []

        # self.plot(ax, ref = 4,specialPatches = [(0,1,0,0.7),self.contacts[0].actives],almostSpecialPatches = [(1,0.6,0.0,0.7),candidatePatches],specialNodes = ["blue", activeNodes], fintC= fintC)
        self.plot(ax, ref = 10,specialPatches = None,almostSpecialPatches = None,specialNodes = {"blue": activeNodes}, fintC= fintC,OnlyMasterSurf=OnlyMasterSurf)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.show()

    def plotContactNow(self, ctct_id=0, labels =True, block = True,SlaveQuads=False, context = False):
        """Enable this to visualize contact state"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.contacts[ctct_id].plotContact(self.u_temp,ax,labels=labels,SlaveQuads=SlaveQuads, context=context)
        self.contacts[ctct_id].plotHooks(ax)
        ax.set_zlim3d(-2, 2)
        plt.show(block = block)

    def savefig(self, increment, iteration = None,  elevation=[ 30 , 30], azimut=[-45 , -45], distance = [10, 10], times = None, dpi = 200, fintC = False, Hooks= False, u = None,simm_time=None):
        #PLOT OPTIONS
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        if not os.path.exists(self.output_dir+"plots"):
            os.mkdir(self.output_dir+"plots")


        #for good appeareance try to keep a ratio  6range(x)=6range(y)=5range(z)
        (minx,maxx),(miny,maxy),(minz,maxz) = self.PlotBoundaries
        hg = max([dij[1]-dij[0] for dij in self.PlotBoundaries])/2    #half-gap to be used in every direction
        cx,cy,cz = (minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2

        ax.set_xlim3d((cx-hg, cx+hg))
        ax.set_ylim3d((cy-hg, cy+hg))
        # ax.set_zlim3d((cz-0.8*hg, cz+0.8*hg))
        ax.set_zlim3d((0, 5))

        # # 3d Example:
        # ax.set_xlim3d((-2, 14))
        # ax.set_ylim3d((-4, 4))
        # ax.set_zlim3d((-2, 6))

        ax.set_aspect("auto")
        ax.xaxis._axinfo["grid"].update({"linewidth":0.1, "color" : "gray"})
        ax.yaxis._axinfo["grid"].update({"linewidth":0.1, "color" : "gray"})
        ax.zaxis._axinfo["grid"].update({"linewidth":0.1, "color" : "gray"})
        ax.tick_params(axis='z', which='both', direction='out', length=0)
        ax.set_yticks([])


        if type(self.contacts[0].candids) == np.ndarray:
            cands = [[] if cand[0]==-1 else cand.tolist() for cand in self.contacts[0].candids]
        else:
            cands = self.contacts[0].candids
        AlmostImportant = sorted(set(flatList(cands))-set(self.contacts[0].actives)) if len(self.contacts)!= 0 else []

        if times!= None:
            t0,t,tf = times
            tt = (t-t0)/(tf-t0)
        else:
            t = simm_time if simm_time is not None else 0
            tt = 0

        ctct = self.contacts[0]
        # activeNodes = [ctct.slaveNodes[idx] for idx in ctct.proj[:,3].nonzero()[0]]
        idxNodes = ctct.proj[:,3].nonzero()[0]
        activeNodes = [ctct.slaveNodes[idx] for idx in idxNodes if ctct.actives[idx] is not None ]
        # set_trace()
        activePatches = [pr[0] for pr in ctct.proj if pr[3]!=0]
        self.plot(ax, ref = 10,specialPatches = [(0,1,0,0.75),activePatches],
                              almostSpecialPatches = [(1.0,0.64,0.0,0.75), AlmostImportant],
                              specialNodes = {"red": activeNodes,}, 
                              text = "time: "+str(round(t,12)) + ((", iter:"+str(iteration)) if iteration is not None else ""),
                              fintC = fintC,
                              plotHooks = Hooks,
                              OnlyMasterSurf=True,
                              u = u
                              )

        #Camera effects
        ax.view_init(elev=elevation[0]*(1-tt)+elevation[1]*tt,azim=azimut[0]*(1-tt)+azimut[1]*tt)
        ax.dist = distance[0]*(1-tt)+distance[1]*tt     #default dist = 10
        ax.set_proj_type('ortho')
        # plot_coords(ax,orig=(minx-0.5+1.0,miny-0.5,minz-0.5))
        plot_coords(ax,orig=(-5.3,0.0,0.15))

        plt.tight_layout()

        if iteration==None:
            plt.savefig(self.output_dir+"plots/fig"+str(increment)+".png",dpi = dpi)
        else:
            plt.savefig(self.output_dir+"plots/fig"+str(increment)+"-"+str(iteration)+".png",dpi = dpi)
        plt.close()

    def savedata(self,time,*args, dofs = "all", name = None, Sum=False):
        for arg in args:
            # if type(arg)==tuple:
            #     set_trace()
            filename = self.output_dir+"data_"+(arg if name is None else name)+".csv"
    
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
                    attr = getattr(self,arg)
                    if dofs == "all": 
                        if hasattr(attr,'__iter__'):
                            csvwriter.writerow([time]+attr.tolist())
                        else:
                            csvwriter.writerow([time]+[attr])
                    else: print("Unrecognized datatype to be stored from string. Not saving this data"), set_trace()
                else:
                    print("Unrecognized datatype to be stored. Not saving this data"), set_trace()
    
    def saveNodalData(self,body_id,time,*args, nodes = "all", name = None, Sum=False):
    
        if type(nodes)==list or type(nodes)==np.ndarray:
            dofs = self.bodies[body_id].DoFs[nodes]
        elif nodes == "all":
            dofs = self.bodies[body_id].DoFs
        for arg in args:
            filename = self.output_dir+(("sum_" if Sum else "")+(arg if name is None else name))+".csv"
    
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

        ctct = self.contacts[ctct_id]
        stringthings = ["gn", "kn"]
        slaveNodes = range(ctct.nsn) if slaveNodes is None else slaveNodes

        def states(idx):
            return [ctct.proj[idx,3], ctct.alpha_p[idx]*ctct.kn, ctct.hook[idx,3]]
        

        for stidx, stringthing in enumerate(stringthings):
            filename = self.output_dir+"ctct"+str(ctct_id)+"_"+stringthing+".csv"
    
            # writing to csv file
            with open(filename, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
                
                # writing the fields
                csvwriter.writerow([time]+[states(idx)[stidx] for idx in slaveNodes])

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

    def setReferences(self, justOutput = False):
        u_ref = self.u.copy()
        u_temp_ref = self.u_temp.copy()
        fint_ref = self.fint.copy()
        alpha_ref = list([contact.alpha_p.copy() for contact in self.contacts])
        new_ref   = list([contact.new.copy() for contact in self.contacts])
        xs_ref = list([contact.xs.copy() for contact in self.contacts])
        act_ref = list([contact.actives.copy() for contact in self.contacts])
        cand_ref = list([contact.candids.copy() for contact in self.contacts])

        FPconv_ref =  list([body.FPconv.copy() if not body.isRigid else None for body in self.bodies])
        EPcum_ref =  list([(body.EPcum).copy() if not body.isRigid else None for body in self.bodies])

        REF = [u_ref,u_temp_ref, fint_ref, alpha_ref, new_ref, xs_ref,act_ref,cand_ref,FPconv_ref,EPcum_ref]

        if justOutput:
            return REF

        self.REF = REF

    def getReferences(self, alphas=True,actives=False):
        self.u = self.REF[0].copy()
        self.u_temp = self.REF[1].copy()
        self.fint = self.REF[2].copy()
        alpha_ref, new_ref, xs_ref,act_ref,cand_ref,FPconv_ref,EPcum_ref = self.REF[3:]
        # self.u_temp = np.array(self.u)
        for icont, contact in enumerate(self.contacts):
            if alphas: contact.alpha_p = alpha_ref[icont]
            contact.new  = new_ref[icont].copy()
            contact.xs   = xs_ref[icont].copy()
            if actives: 
                contact.actives = act_ref[icont].copy()
                contact.candids = cand_ref[icont].copy()

        for i_b,body in enumerate(self.bodies):
            if not body.isRigid:
                body.FPconv = FPconv_ref[i_b].copy()
                body.FPtemp = FPconv_ref[i_b].copy()
                body.EPcum = EPcum_ref[i_b].copy()


    def get_fint(self, DispTime = False, temp = False):
        t0_fint = time.time()
        self.fint = np.zeros_like(self.u,dtype=float)
        # for body in self.bodies: body.get_fint_fast(self, temp = temp)
        # for body in self.bodies: body.get_fint(self, temp = temp)
        u = self.u if temp==False else self.u_temp
        fint_t0 = time.time()
        for body in self.bodies: 
            self.fint += body.compute_mf_plastic(u,self)[0]
        if DispTime: print("Getting fint : ",time.time()-t0_fint,"s")
        self.TIMERS[3] += time.time()-fint_t0

        u = self.u if temp==False else self.u_temp

        fcon_t0 = time.time()
        for ctct in self.contacts:
            if temp: ctct.patch_changes = []                 # to keep track of the changes during iterations
            sDoFs  = ctct.slaveBody.DoFs[ctct.slaveNodes]
            ctct.xs = np.array(ctct.slaveBody.X )[ctct.slaveNodes ] + np.array(u[sDoFs ])      # u_temp
            
            # if ctct.ANNmodel is not None:
            #     ctct.get_fintCamilo(self,DispTime = False, tracing = False)
            # else:
            if self.IterUpdate:
                ctct.getfintC_unilateral(self, DispTime=DispTime)      # uses xs
            else:
                ctct.getfintC(self, DispTime=DispTime)      # uses xs
        self.TIMERS[4] += time.time()-fcon_t0

    def get_K(self, DispTime = False):
        t0_K = time.time()
        self.K = sparse.coo_matrix((self.ndof,self.ndof),dtype=float)

        for body in self.bodies:
            # body.get_K(self)    # uses model.u_temp
            self.K += body.compute_k_plastic(self.u_temp,self)

        printif(DispTime,"Getting K : ",time.time()-t0_K,"s")


        for contact in self.contacts:
            # if contact.ANNmodel is not None:
            #     contact.get_KCamilo(self,DispTime = False, tracing = False)
            # else:

            if self.IterUpdate:
                contact.getKC_unilateral(self, DispTime=DispTime)     #uses model.u_temp
            else:
                contact.getKC(self, DispTime=DispTime)     #uses model.u_temp


    def NR(self,tol=1e-10,maxiter=10,plotIters=False):

        if self.transform_2d is not None:
            return self.NR_2d(tol=tol,maxiter=maxiter,plotIters=plotIters)

        # NEWTON-RAPHSON
        di, fr = self.diri, self.free
        TimeDisp = False
        RES, niter = 1+tol, 0

        self.get_fint(DispTime=TimeDisp,temp=True)   # uses self.u_temp     (no dirichlet)

        dua = self.u[di] - self.u_temp[di]      # difference due to dirichlet BCs applied
        
        RES_prev = 1e100
        RES = 1e99
        minRES = float(RES)

        # while RES>tol and RES<5*RES_prev and niter<maxiter and not np.isnan(RES):
        while RES>tol and niter<maxiter and not np.isnan(RES):
            self.COUNTS[2] += 1

            with open(self.output_dir+"COUNTERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.COUNTS_NAMES)
                csvwriter.writerow(self.COUNTS.tolist())

            self.TIMERS[0] = time.time()-self.solve_timer_0
            with open(self.output_dir+"TIMERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.TIMERS_NAMES)
                csvwriter.writerow([round(timer, 3) for timer in self.TIMERS])

            RES_prev = RES

            # getting K and KC
            NR_solver_t0 = time.time()
            self.get_K(DispTime=TimeDisp)  # <-- uses self.u_temp           (no dirichlet)

            # Linear System
            Kaa=self.K[np.ix_(di,di)]
            Kab=self.K[np.ix_(di,fr)]
            Kba=self.K[np.ix_(fr,di)]
            Kbb=self.K[np.ix_(fr,fr)]

            fina=self.fint[di]
            finb=self.fint[fr]
            fexb=self.fext[fr]      # always zero (?)

            dub =spsolve(Kbb,(fexb-finb-Kba.dot(dua)).T)     # Uses all available processors
            self.TIMERS[1] += time.time()-NR_solver_t0

            self.fext[di] = fina + Kaa.dot(dua) + Kab.dot(dub)

            dua = np.zeros_like(self.u[di])

            self.u[fr] += dub

            self.get_fint(DispTime=TimeDisp)   # uses self.u

            RES = self.residual(printRes=True)

            if RES<minRES:
                minRES = float(RES)
            
            for ic, ctct in enumerate(self.contacts):
                pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
                with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([self.t if niter==0 else None,niter,RES]+ctct.patch_changes)

            if plotIters: self.plotContactNow()

            niter += 1
            self.u_temp = np.array(self.u)

        return RES<tol, RES

    def NR_2d(self,tol=1e-10,maxiter=10,plotIters=True):

        # N = self.transform_2d*np.sqrt(0.5)
        Ns,Nt = self.transform_2d

        # Give (S)ymmetry to u, u_temp
        self.u = Ns@self.u
        self.u_temp = Ns@self.u_temp


        # NEWTON-RAPHSON
        bool_di = np.zeros(self.ndof)

        bool_di[self.diri] = 1
        di = np.where(Nt.T@bool_di!=0)[0]
        fr = list(set(range(Nt.shape[1]))-set(di))

        TimeDisp = False
        RES, niter = 1+tol, 0


        # set_trace()

        self.get_fint(DispTime=TimeDisp,temp=True)   # uses self.u_temp     (no dirichlet)
        fint_r = Nt.T@Ns.T@self.fint
        fext_r = np.zeros_like(fint_r)
        u_r = Nt.T@self.u
        u_r_temp = Nt.T@self.u_temp
        
        RES_prev = 1e100
        RES = 1e99
        minRES = float(RES)

        Nta = Nt[:,di]
        Ntb = Nt[:,fr]
        # dua = (Nt.T@(self.u - self.u_temp))[di]      # difference due to dirichlet BCs applied
        dua = (u_r - u_r_temp)[di]      # difference due to dirichlet BCs applied

        # set_trace() 
        # while RES>tol and RES<5*RES_prev and niter<maxiter and not np.isnan(RES):
        while RES>tol and  niter<maxiter and not np.isnan(RES):
            RES_prev = RES
            self.COUNTS[2] += 1

            with open(self.output_dir+"COUNTERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.COUNTS_NAMES)
                csvwriter.writerow(self.COUNTS.tolist())

            self.TIMERS[0] = time.time()-self.solve_timer_0
            with open(self.output_dir+"TIMERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.TIMERS_NAMES)
                csvwriter.writerow([round(timer, 3) for timer in self.TIMERS])



            NR_solver_t0 = time.time()
            # getting K and KC
            self.get_K(DispTime=TimeDisp)  # <-- uses self.u_temp           (no dirichlet)
            Kr = Nt.T@Ns.T@self.K@Ns@Nt

            # Linear System
            Kaa=Kr[np.ix_(di,di)]
            Kab=Kr[np.ix_(di,fr)]
            Kba=Kr[np.ix_(fr,di)]
            Kbb=Kr[np.ix_(fr,fr)]

            fina=fint_r[di]
            finb=fint_r[fr]
            fexb=fext_r[fr]     # always zero (?)

            # dub =spsolve(Kbb,fexb-finb-Kba.dot(dua))     # Uses all available processors
            dub =spsolve(Kbb,(fexb-finb-Kba.dot(dua)).T)     # Uses all available processors
            self.TIMERS[1] += time.time()-NR_solver_t0

            fext_r[di] = (fina + Kaa.dot(dua) + Kab.dot(dub))
            # self.fext = Ns@self.fext


            dua[:] = 0.0

            u_r[fr] += dub

            self.u[self.free] = (Ns@Nt@u_r)[self.free]
            self.u = Ns@self.u

            # self.plotNow(as2D=True,OnlyMasterSurf=True)

            self.get_fint(DispTime=TimeDisp)   # uses self.u
            fint_r = Nt.T@Ns.T@self.fint


            # RES = self.residual(printRes=True)
            RES = norm(fint_r-fext_r)
            print("resid :",RES)


            if RES<minRES:
                minRES = float(RES)
            
            for ic, ctct in enumerate(self.contacts):
                pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
                with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([self.t if niter==0 else None,niter,RES]+ctct.patch_changes)

            if plotIters: self.plotContactNow()

            niter += 1
            self.u_temp = np.array(self.u)

            # # if (niter == maxiter and RES>tol) or np.isnan(RES) or RES>1e+15:
            # if (niter == maxiter and RES>tol) or np.isnan(RES) or RES>1e+15:
            #     print('Increment failed to converge !!!!! Redoing half')
            #     return False

        return RES<tol, RES

    def  BFGS(self,FUNJAC, u0, tol = 1e-10, free_ind = None,ti = None,simm_time = None, plot = False,unilateral=True):
        

        # # Profiling
        # import cProfile
        # import pstats
        # import io
        # pr = cProfile.Profile()
        # pr.enable()




        if free_ind is None:
            free_ind = self.free
        
        nfr = len(free_ind)

        f , m_new = FUNJAC(u0,unilateral=unilateral)
        u = u0.copy()
        f_2 = f[free_ind]
        K_new_inv = np.eye(nfr)
        f_new = np.zeros(nfr)
        m0 = 0
        
        for ctct in self.contacts:
            ctct.patch_changes = []

        iter = 0
        while np.linalg.norm(f_2 - f_new) > 0 and np.linalg.norm(f_2) > tol:
            self.COUNTS[4] += 1

            with open(self.output_dir+"COUNTERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.COUNTS_NAMES)
                csvwriter.writerow(self.COUNTS.tolist())

            self.TIMERS[0] = time.time()-self.solve_timer_0
            with open(self.output_dir+"TIMERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.TIMERS_NAMES)
                csvwriter.writerow([round(timer, 3) for timer in self.TIMERS])


            iter += 1
            print("ITER:",iter)
            f_old = f_new.copy()
            f_new = f_2.copy()
            K_old_inv = K_new_inv.copy()
            delta_f = f_new - f_old

            min_t0 = time.time()
            if iter == 1:
                h_new = -np.dot(K_old_inv, f_new)
            else:
                K_new_inv = K_old_inv + ((np.inner(delta_u, delta_f) + np.inner(delta_f, np.dot(K_old_inv, delta_f)))*(np.outer(delta_u,delta_u)))/ (np.dot(delta_u, delta_f) ** 2)- (np.outer(np.dot(K_old_inv, delta_f),delta_u) + np.inner(np.outer(delta_u, delta_f),K_old_inv)) / np.dot(delta_u, delta_f)
                h_new = -np.dot(K_new_inv, f_new)

            if not np.isfinite(norm(h_new)):
                set_trace()
            
            self.TIMERS[2] += time.time()-min_t0

            a2,f2,f_2,ux = self.linesearch(FUNJAC, u, h_new, f_new, free_ind, alpha_init=1, c_par2=0.9,unilateral=unilateral)

            self.write_m_and_f(0.0,norm(f_2),iter)
                       
            delta_u = ux[free_ind] - u[free_ind]
            u = ux

            if plot:
                if self.transform_2d is None:
                    # 3D case
                    self.savefig(ti,iter,distance=[10,10],u = ux,simm_time=simm_time)
                else:
                    # for 2D case
                    Ns,Nt = self.transform_2d
                    self.savefig(ti,iter,azimut=[-90, -90],elevation=[0,0],distance=[10,10],u = Ns@Nt@ux,simm_time=simm_time)
    
            print("\talpha:",a2,"\tf2:",f2,"\t\t|f_2|:",norm(f_2))


            # if iter>10:
            #     pr.disable()
            #     s = io.StringIO()
            #     ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            #     ps.print_stats()

            #     print("\n".join(s.getvalue().split("\n")[:100]))
            #     set_trace()


        return u, m0, iter , norm(f_2)

    def LBFGS(self,FUNJAC, u0, tol = 1e-10,nli=10, free_ind = None,ti = None,simm_time = None, plot=False,unilateral=True):

        if free_ind is None:
            free_ind = self.free

        nfr = len(free_ind)

        f , m_new = FUNJAC(u0,unilateral=unilateral)

        u = u0.copy()
        f_2 = f[free_ind]
        m_new = norm(f_2)
        self.write_m_and_f(m_new,norm(f),0)
        f_new = np.zeros(nfr)
        m0 = 0
        DF = np.zeros((nfr,nli))
        DU = np.zeros((nfr,nli))
        rho = np.zeros((nli,1))

        
        for ctct in self.contacts:
            ctct.patch_changes = []

        iter = 0
        while np.linalg.norm(f_2 - f_new) > 0 and np.linalg.norm(f_2) > tol:
            self.COUNTS[4] += 1

            with open(self.output_dir+"COUNTERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.COUNTS_NAMES)
                csvwriter.writerow(self.COUNTS.tolist())
            
            self.TIMERS[0] = time.time()-self.solve_timer_0
            with open(self.output_dir+"TIMERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.TIMERS_NAMES)
                csvwriter.writerow([round(timer, 3) for timer in self.TIMERS])

            min_t0 = time.time()
            iter += 1
            print("ITER:",iter)

            f_old = f_new.copy()
            f_new = f_2.copy()
            delta_f = f_new - f_old

            if iter == 1:
                h_new = -f_new
            else:
                gamma = np.zeros((nli,1))
                h_new = f_new.copy()
                for j in reversed(range(max(0,nli-iter+1),nli)):
                    rho_j = rho[j]
                    delta_u = DU[:,j]
                    delta_f = DF[:,j]
                    gamma_j = rho_j*delta_u@h_new
                    h_new -= gamma_j*delta_f
                    gamma[j] = gamma_j

                for j in range(max(0,nli-iter+1),nli):
                    rho_j = rho[j]
                    delta_u = DU[:,j]
                    delta_f = DF[:,j]
                    gamma_j = gamma[j]
                    eta = rho_j*delta_f@h_new
                    h_new += (gamma_j - eta)*delta_u

                h_new = -h_new
            self.TIMERS[2] += time.time()-min_t0

            if not np.isfinite(norm(h_new)):
                set_trace()

            a2,f2,f_2,ux = self.linesearch(FUNJAC, u, h_new, f_new, free_ind,unilateral=unilateral)

            self.write_m_and_f(0.0,norm(f_2),iter)
                       
            delta_u = ux[free_ind] - u[free_ind]
            u = ux.copy()

            DU[:,:nli-1] = DU[:,1:nli].copy()
            DU[:,-1] = delta_u.copy()
            DF[:,:nli-1] = DF[:,1:nli].copy()
            DF[:,-1] = f_2-f_new
            rho[:nli-1] = rho[1:nli].copy()
            rho[-1] = 1/((f_2-f_new)@delta_u)

            if plot:
                if self.transform_2d is None:
                    # 3D case
                    self.savefig(ti,iter,distance=[10,10],u = Ns@Nt@ux,simm_time=simm_time)
                else:
                    # for 2D case
                    Ns,Nt = self.transform_2d
                    self.savefig(ti,iter,azimut=[-90, -90],elevation=[0,0],distance=[10,10],u = Ns@Nt@ux,simm_time=simm_time)

            print("\talpha:",a2,"\tf2:",f2,"\t\t|f_2|:",norm(f_2))

        return u, m0, iter , norm(f_2)

    def linesearch(self, FUNJAC, u, h_new, f_new, free_ind, alpha_init=1, c_par2=0.9, unilateral=True):
        ###################
        ### LINE SEARCH ###
        ###################
        # from joblib import Memory
        # cache_dir = './cache_directory'
        # memory = Memory(cache_dir, verbose=0)
        
        # @memory.cache
        def func(alpha,FUNJAC,u,free_ind,h_new,checkSecant=False):
            ux = u.copy()
            ux[free_ind] = u[free_ind] + alpha*h_new
            if checkSecant:
                fb,fc,ft,_,_ = FUNJAC(ux,split=True,unilateral=unilateral)
                fb = fb[free_ind]
                fc = fc[free_ind]
                ft = ft[free_ind]
                fb = h_new@fb
                fc = h_new@fc
                ft = h_new@ft

                return fb,fc,ft

            f , _ = FUNJAC(ux)
            f_3 = f[free_ind]
            f3 = h_new@f_3

            # self.ALs.append(alpha)
            # self.FFs.append(f3)
            return f3, f_3, ux

        # self.ALs = []   # Stores the values found during linesearch
        # self.FFs = []   # Stores the values found during linesearch

        a1 = 0
        f1 = h_new@f_new
        f_1 = f_new

        tol2 = min(abs(f1)/1e6 , 1e-12)

        a2 = 2*alpha_init
        f2 = np.nan
        while np.isnan(f2):
            a2 /= 2
            f2,f_2,ux = func(a2,FUNJAC,u,free_ind,h_new)

        ready_to_bisect = f2>0
        if ready_to_bisect:
            a_pos, f_pos, f__pos = a2,f2,f_2

        min_found = abs(f2)<tol2 and (np.dot(h_new, f_2) >= c_par2 * np.dot(h_new, f_new))

        print("\talphas:",[a1,a2],"\tf",[f1,f2])

        if not ready_to_bisect and not min_found:

            # Compute (a valid) a3
            delta = 2*(a2-a1)
            f3 = np.nan
            while np.isnan(f3):                
                delta /= 2
                a3 = a2 + delta                
                f3,f_3,ux = func(a3,FUNJAC,u,free_ind,h_new)

            print("\talphas:",[a1,a2,a3],"\tf",[f1,f2,f3])

            ready_to_bisect = f3>0
            if ready_to_bisect:
                a_pos, f_pos, f__pos = a3,f3,f_3
                a1,f2,f_1 = a2,f2,f_2   
                a2,f2,f_2 = a3,f3,f_3       # from here it will go directly to bisection
            min_found = abs(f3)<tol2 and (np.dot(h_new, f_3) >= c_par2 * np.dot(h_new, f_new))

            while not ready_to_bisect and not min_found:

                try:
                    parab = quadratic_fit_min_zeros([[a1,f1],[a2,f2],[a3,f3]])  # if singular or no-zeros, goes to 'except'
                    if parab["a"]>0:
                        a0 = max(parab["zeros"])
                    else:
                        a0 = min(parab["zeros"])    # >0 already guaranteed by previous while loop
                    # is_alpha_pos = a0>0
                    is_alpha_on_the_right = a0>a3    # at this point I NEED to move to the right. I dont want 0<a0<a{1,2,3} !!
                except:
                    parab = {'zeros':None}
                    is_alpha_on_the_right = False # This will force to search to the right (in while loop below)

                # Make sure the parabola crosses X in ascending manner.
                while parab["zeros"] is None or not is_alpha_on_the_right:
                    print("\tparabola not ready. Moving to right...")
                    # Let's move to the right
                    delta = 1.5*2*(a2-a1) # step 50% bigger each time and initially double because it will do at least one bisection
                    a1 = a2
                    f1 = f2

                    a2 = a3
                    f2 = f3

                    # Compute (a valid) a3
                    f3 = np.nan
                    while np.isnan(f3):                
                        delta /= 2
                        a3 = a2 + delta                
                        f3,f_3,ux = func(a3,FUNJAC,u,free_ind,h_new)

                    ready_to_bisect = f3>0
                    min_found = abs(f3)<tol2 and (np.dot(h_new, f_3) >= c_par2 * np.dot(h_new, f_new))
                    if ready_to_bisect or min_found:
                        a1,f2,f_1 = a2,f2,f_2   
                        if f3>0:
                            a_pos, f_pos, f__pos = a3,f3,f_3
                            a2,f2,f_2 = a3,f3,f_3
                        else:
                            a2,f2,f_2 = a_pos,f_pos,f__pos


                        break

                    print("\talphas:",[a1,a2,a3],"\tf",[f1,f2,f3])
                    try:
                        parab = quadratic_fit_min_zeros([[a1,f1],[a2,f2],[a3,f3]])  # if singular or no-zeros, goes to 'except'
                        if parab["a"]>0:
                            a0 = max(parab["zeros"])
                        else:
                            a0 = min(parab["zeros"])    # >0 already guaranteed by previous while loop
                        # is_alpha_pos = a0>0
                        is_alpha_on_the_right = a0>a3    # at this point I NEED to move to the right. I dont want 0<a0<a{1,2,3} !!
                    except:
                        parab = {'zeros':None}
                        is_alpha_on_the_right = False # This will force to search to the right (in while loop below)


                if ready_to_bisect or min_found:
                    # this would mean they were found in the previous while loop
                    break

                delta = 2*(a0-a3)
                f0 = np.nan
                while np.isnan(f0):
                    delta /= 2
                    a0 = a3 + delta
                    f0,f_0,ux = func(a0,FUNJAC,u,free_ind,h_new)
                
                ready_to_bisect = f0>0
                if ready_to_bisect:
                    a_pos, f_pos, f__pos = a0,f0,f_0

                min_found = abs(f0)<tol2 and (np.dot(h_new, f_0) >= c_par2 * np.dot(h_new, f_new))
                print("\talphas:",[a1,a2,a3,a0],"\tf",[f1,f2,f3,f0])

                if min_found or ready_to_bisect:
                    a1,f2,f_1 = a3,f3,f_3
                    a2,f2,f_2 = a0,f0,f_0
                    break
                
                a1,f1,f_1 = a2,f2,f_2
                a2,f2,f_2 = a3,f3,f_3
                a3,f3,f_3 = a0,f0,f_0


        ####################
        # Quadratic Search #
        ####################
        secant = False
        bisection = False
        if not min_found:

            print("\t## Performing quadratic secant search ##")
            if f2>0:
                a3,f3,f_3 = a2,f2,f_2
            else:
                a3,f3,f_3 = a_pos, f_pos, f__pos

            a2 = (a1+a3)/2
            f2,f_2,ux = func(a2,FUNJAC,u,free_ind,h_new)

            f0,f_0 = f1,f_1 # to enter the loop

            qbs_iter = 0
            while not (abs(f0)<tol2 and (np.dot(h_new, f_0) >= c_par2 * np.dot(h_new, f_new))):
                qbs_iter += 1

                if qbs_iter>20 and f2>0:
                    secant = True
                    break

                try:
                    parab = quadratic_fit_min_zeros([[a1,f1],[a2,f2],[a3,f3]])  # if singular or no-zeros, goes to 'except'
                    if parab["a"]>0:
                        a0 = max(parab["zeros"])
                    else:
                        a0 = min(parab["zeros"])    # >0 already guaranteed by previous while loop, NOT!
                    
                    if a0<0:
                        secant = True
                        break

                except:
                    secant = True
                    break
                
                f0,f_0,ux = func(a0,FUNJAC,u,free_ind,h_new)

                if np.isnan(f0):
                    secant = True
                    break

                # storing values that give a f_pos in case we need to call it in the bisection/secant method
                if 0<f0<f_pos: 
                    a_pos, f_pos, f__pos = a0,f0,f_0


                print("\talphas:",[a1,a2,a3],[a0],"\tf",[f1,f2,f3],[f0])

                if a0>a3:    # In this case f1,f2,f3 are all negative. since f0~0 and f is increasing in the interval
                    a1,f1,f_1 = a2,f2,f_2
                    a2,f2,f_2 = a3,f3,f_3
                    a3,f3,f_3 = a0,f0,f_0

                elif a0>a2:      # In this case f1,f2<0,  f3>0

                    d01 = np.sqrt((a0-a1)**2+(f0-f1)**2)
                    d03 = np.sqrt((a0-a3)**2+(f0-f3)**2)

                    if (f0>0 or abs(f3/f0)>4) and d03/d01>1:
                        # in this case f0 is better to be kept than f3
                        a3,f3,f_3 = a0,f0,f_0
                        
                    else:
                        a1,f1,f_1 = a2,f2,f_2
                        a2,f2,f_2 = a0,f0,f_0

                elif a0>a1:      # Here f2,f3>0 but f0 could be positive and in that case it should NOT replace f1

                    d01 = np.sqrt((a0-a1)**2+(f0-f1)**2)
                    d03 = np.sqrt((a0-a3)**2+(f0-f3)**2)

                    if (f0<0 or abs(f1/f0)>4) and d01/d03>1:
                        # in this case f0 is better to be kept than f1
                        a1,f1,f_1 = a0,f0,f_0
                        
                    else:
                        a3,f3,f_3 = a2,f2,f_2
                        a2,f2,f_2 = a0,f0,f_0

                else:
                    # Less likely to reach this part since f1<0 and f increases
                    a3,f3,f_3 = a2,f2,f_2
                    a2,f2,f_2 = a1,f1,f_1
                    a1,f1,f_1 = a0,f0,f_0


            if secant:

                ###################
                # Secant (linear) #
                ###################

                if f2<0:
                    a2,f2,f_2 = a_pos, f_pos, f__pos


                print("\t## parabola failed. Finishing off with Secant method ##")
                print("\tInitially, we have:")
                print("\t\talphas:",[a1,a2,a3],"\tf",[f1,f2,f3])

                sec_iter = 0
                while not (abs(f2)<tol2 and (np.dot(h_new, f_2) >= c_par2 * np.dot(h_new, f_new))) and f2 not in [f1,f3]:
                    if f2>0:
                        a3,f3,f_3 = a2,f2,f_2.copy()
                    else:
                        a1,f1,f_1 = a2,f2,f_2.copy()

                    a2 = (a1*f3 - a3*f1)/(f3-f1)
                    f2,f_2,ux = func(a2,FUNJAC,u,free_ind,h_new)
                    print("\talphas:",[a1,a3],[a2],"\tf",[f1,f3],[f2])

                    sec_iter +=1 
                    if sec_iter>10 or not (0.01<(a2-a1)/(a3-a1)<0.99):
                        bisection = True
                        break
            else:
                a2,f2,f_2 = a0,f0,f_0

            if bisection:
                #############
                # Bisection #
                #############

                print("\t## secant failed. Finishing off with bisection method ##")
                print("\tInitially, we have:")
                print("\t\talphas:",[a1,a3],"\tf",[f1,f3])

                a2,f2,f_2 = a1,f1,f_1.copy()    # just to enter the loop
                bisect_iter = 0
                while not (abs(f2)<tol2 and (np.dot(h_new, f_2) >= c_par2 * np.dot(h_new, f_new))):
                    if f2>0:
                        a3,f3,f_3 = a2,f2,f_2.copy()
                    else:
                        a1,f1,f_1 = a2,f2,f_2.copy()

                    a2 = (a1+a3)/2

                    f2,f_2,ux = func(a2,FUNJAC,u,free_ind,h_new)
                    print("\talphas:",[a1,a3],[a2],"\tf",[f1,f3],[f2])

                    if a2 in [a1,a3]:
                        break

                    bisect_iter +=1 
                    # if bisect_iter>40:
                    #     print("\tmaxiter reached! returning current value")
                    #     break

        # Here, one could plot all the values stored in self.ALs and self.FFs

        return a2,f2,f_2,ux

    def TR(self,FUNJAC,HESS, u0, free_ind = None, tol = 1e-10,tol2 = 1e-12,ti=None,simm_time=None,plot=False, unilateral=True,precond = False):
        from ilupp import icholt

        if free_ind is None:
            free_ind = self.free

        nfr = len(free_ind)

        TR_rad = 0.0001*nfr
        TR_rad_min = 0*nfr
        TR_rad_max = 10000000*nfr
        # TR_rad_max = 0.0001

        ff , m_new = FUNJAC(u0,unilateral=unilateral)
        f_new = ff[free_ind]

        rho = 1
        iter = 0
        update_signal = 1

        h_full = np.zeros_like(u0)

        u = u0.copy()


        while norm(f_new)>tol:
            self.COUNTS[4] += 1

            with open(self.output_dir+"COUNTERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.COUNTS_NAMES)
                csvwriter.writerow(self.COUNTS.tolist())

            self.TIMERS[0] = time.time()-self.solve_timer_0
            with open(self.output_dir+"TIMERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.TIMERS_NAMES)
                csvwriter.writerow([round(timer, 3) for timer in self.TIMERS])


            iter += 1
            print("Iter:",iter)

            min_t0 = time.time()
            if update_signal==1:
                # K = sparse.csr_matrix(HESS(u))
                K = HESS(u)
                K = K[np.ix_(free_ind,free_ind)]
                if not precond:
                    M = np.eye(nfr)
                elif precond == "diag":
                    M = sparse.linalg.inv(sparse.diags(K.diagonal()))
                elif precond == "tril":
                    M = sparse.linalg.inv(sparse.csr_matrix(np.triu(K,k=0)))
                elif precond == "icho":
                    cnt_precon=0
                    K0 = K.copy()

                    if np.any(K.diagonal() < 0):
                        K.setdiag(np.abs(K.diagonal()))
                        print("Negative diagonal component detected in K. Making Kii=abs(Kii).")
                        # sys.exit(1)

                    failed = True
                    while failed:
                        M1 = icholt(K, add_fill_in=0, threshold=0.0)
                        M = M1@M1.T
                        cond_icho = np.linalg.cond(M.todense())
                        print("Condition number of icho:",cond_icho)
                        if cond_icho<1e+10:
                            break
                        else:
                            cnt_precon += 1
                            print("no invK. Using diagonal preconditioner,(",np.linalg.cond(K.todense()),")")
                            Kdiag = sparse.diags(K0.diagonal())*10**(-8+cnt_precon)
                            # Kdiag = sparse.identity(K.shape[0]) * 10**(-8 + cnt_precon)
                            # K = K0 + sparse.identity(K0.shape[0]) * 10**(-8 + cnt_precon)
                            K = K0 + Kdiag
                            continue

                        failed = False

                elif precond == "same":
                    M = sparse.linalg.inv(K)

                condK = np.linalg.cond(K.todense())

            r_new = -f_new.copy()

            p = sparse.linalg.spsolve(M, r_new).reshape(-1)
            h = np.zeros(nfr)
            q = p.copy()

            signal = 1
            signalx = 0

            while signal==1:

                self.COUNTS[6] += 1


                if p.T@K@p<=0: 
                    a = p.T@M@p
                    b = 2*p.T@M@h
                    c = h.T@M@h - TR_rad**2

                    alpha=(-b+np.sqrt(b**2-4*a*c))/(2*a)
                    
                    h += alpha*p
                    if alpha < 0:
                        print("Negative alpha1. Stopping the simulation.")
                        sys.exit(1)

                    signalx = 1
                    signal = 2
                    break

                alpha = float(r_new@q/(p.T@K@p))
                
                if (h+alpha*p).T@M@(h+alpha*p) >= TR_rad**2:
                    a = p.T@M@p
                    b = 2*p.T@M@h
                    c = h.T@M@h - TR_rad**2

                    alpha=(-b+np.sqrt(b**2-4*a*c))/(2*a)

                    if alpha < 0:
                        print("Negative alpha2. Stopping the simulation.")
                        sys.exit(1)

                    h += alpha*p
                    signalx = 1
                    signal = 2
                    break

                var_diego = r_new.T@q
                h += alpha*p
                r_new -= np.array(alpha*K@p).reshape(-1)
                if not np.isfinite(norm(r_new)):
                    print("|r_new| wrong. Stopping the simulation.")
                    sys.exit(1)

                if norm(r_new)<max(tol2,1e-5*norm(f_new)):
                    signal = 0
                    break

                q = sparse.linalg.spsolve(M, r_new).reshape(-1)
                beta = float(r_new@q/var_diego)
                p = q + beta*p

            self.TIMERS[2] += time.time()-min_t0

            h_full[free_ind] = h.copy()
            ff , m_h = FUNJAC(u + h_full,unilateral=unilateral)
            f_h = ff[free_ind]
        
            rho = 0.5*(f_new+f_h)@h/(0.5*h.T@K@h + f_new.T@h)
            # rho = -(m_new-m_h)/(0.5*h.T@K@h + f_new.T@h)
            if np.isnan(rho):
                rho = -1e5

            if rho<0.25:
                TR_rad = max(0.25*TR_rad,TR_rad_min)
            else:
                
                if (rho>0.75) and signalx==1:
                    TR_rad = min(2*TR_rad,TR_rad_max)

            update_signal = 0

            if rho>0.25:
                u += h_full
                m_new = m_h
                f_new = f_h.copy()
                update_signal = 1


            self.write_list([norm(h),norm(f_new),condK],iter)
            if plot and iter%50==0:                 # Trust-region uses approx 1 eval per iter so we plot only every 10 iters
                if self.transform_2d is None:
                    # 3D case
                    self.savefig(ti,iter,distance=[10,10],u = u,simm_time=simm_time)
                else:
                    # for 2D case
                    Ns,Nt = self.transform_2d
                    self.savefig(ti,iter,azimut=[-90, -90],elevation=[0,0],distance=[10,10],u = Ns@Nt@u,simm_time=simm_time)



            # print("\th:",norm(h), "\tf:",norm(f_new),"\trho:",rho,"\tTR:",TR_rad, "\tm_h:",m_h, "\tf_h:",norm(f_h), "\tdm:",m_h-m_new)
            print("\th:",norm(h), "\tf:",norm(f_new),"\trho:",rho,"\tTR:",TR_rad, "\tm_h:",m_h, "\tf_h:",norm(f_h), "\tdmda:",f_new@h)
            if TR_rad < 1e-50:
                print("TR_rad is smaller than 1e-50. Stopping the simulation.")
                sys.exit(1)


            for ctct in self.contacts:
                print("actives:",ctct.actives)


        return u, m_new, iter,norm(f_new)

    def TR_backup20241129(self,FUNJAC,HESS, u0, free_ind = None, tol = 1e-10,tol2 = 1e-12,ti=None,simm_time=None,plot=False, unilateral=True,precond = False):
        from ilupp import icholt

        if free_ind is None:
            free_ind = self.free

        nfr = len(free_ind)

        TR_rad = 0.0001*nfr
        TR_rad_min = 0*nfr
        TR_rad_max = 10000000*nfr

        ff , m_new = FUNJAC(u0,unilateral=unilateral)
        f_new = ff[free_ind]

        rho = 1
        iter = 0
        update_signal = 1

        h_full = np.zeros_like(u0)

        u = u0.copy()


        while norm(f_new)>tol:
            self.COUNTS[4] += 1

            with open(self.output_dir+"COUNTERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.COUNTS_NAMES)
                csvwriter.writerow(self.COUNTS.tolist())

            iter += 1
            print("Iter:",iter)

            if update_signal==1:
                # K = sparse.csr_matrix(HESS(u))
                K = HESS(u)
                K = K[np.ix_(free_ind,free_ind)]
                if not precond:
                    M = np.eye(nfr)
                elif precond == "diag":
                    M = sparse.linalg.inv(sparse.diags(np.asarray(K.diagonal())[0]))
                elif precond == "tril":
                    M = sparse.linalg.inv(sparse.csr_matrix(np.triu(K,k=0)))
                elif precond == "icho":
                    cnt_precon=0
                    K0 = K.copy()

                    failed = True
                    while failed:
                        icho = icholt(sparse.csr_matrix(K), add_fill_in=0, threshold=0.0)
                        cond_icho = np.linalg.cond(icho.todense())
                        print("Condition number of icho:",cond_icho)
                        if cond_icho<1e+10:
                            M = sparse.linalg.inv(icholt(sparse.csr_matrix(K), add_fill_in=0, threshold=0.0).T)

                        else:
                            cnt_precon += 1
                            print("no invK. Using diagonal preconditioner,(",np.linalg.cond(np.asarray(K)),")")
                            Kdiag = sparse.diags(np.asarray(K0.diagonal())[0])*10**(-8+cnt_precon)
                            # Kdiag = sparse.identity(K.shape[0]) * 10**(-8 + cnt_precon)
                            # K = K0 + sparse.identity(K0.shape[0]) * 10**(-8 + cnt_precon)
                            K = K0 + Kdiag
                            continue

                        failed = False

                elif precond == "same":
                    M = sparse.csr_matrix(np.linalg.inv(K))

                condK = np.linalg.cond(K)

            r_new = -M.T@f_new.copy()
            p = r_new.copy()
            h = np.zeros(nfr)
            signal = 1
            signalx = 0

            while signal==1:

                self.COUNTS[6] += 1

                r_old = r_new.copy()
                alpha = float(r_old@r_old/(p.T@M.T@K@M@p))

                if norm(M@(h+alpha*p))<TR_rad and p.T@M.T@K@M@p>0:            # In trust region AND convex direction
                # if (h+alpha*p).T@M@(h+alpha*p) < TR_rad**2 and p.T@M.T@K@M@p>0:            # In trust region AND convex direction
                    h += alpha*p
                    r_new = r_old - np.array(alpha*M.T@K@M@p).reshape(-1)
                    beta = (r_new@r_new)/(r_old@r_old)
                    p = r_new+beta*p
                    if norm(r_new)<max(tol2,1e-5*norm(f_new)):
                        signal = 0
                else:
                    a = p.T@M.T@M@p
                    b = 2*p.T@M.T@M@h
                    c = h.T@M.T@M@h - TR_rad**2
                    alpha = 1e-2
                    eq = a*alpha**2 + b*alpha + c
                    while abs(eq)>1e-10:
                        stiff = 2*a*alpha + b
                        d_alpha = -eq/stiff
                        alpha += d_alpha
                        eq = a*alpha**2 + b*alpha + c
                    h += alpha*p
                    signalx = 1
                    signal = 2

            h = M@h
            h_full[free_ind] = h.copy()
            ff , m_h = FUNJAC(u + h_full,unilateral=unilateral)
            f_h = ff[free_ind]
        
            rho = 0.5*(f_new+f_h)@h/(0.5*h.T@K@h + f_new.T@h)
            # rho = -(m_new-m_h)/(0.5*h.T@K@h + f_new.T@h)
            if np.isnan(rho):
                rho = -1e5

            if rho<0.25:
                TR_rad = max(0.25*TR_rad,TR_rad_min)
            else:
                
                if (rho>0.75) and signalx==1:
                    TR_rad = min(2*TR_rad,TR_rad_max)

            update_signal = 0

            if rho>0.25:
                u += h_full
                m_new = m_h
                f_new = f_h.copy()
                update_signal = 1


            self.write_list([norm(h),norm(f_new),condK],iter)
            if plot and iter%50==0:                 # Trust-region uses approx 1 eval per iter so we plot only every 10 iters
                if self.transform_2d is None:
                    # 3D case
                    self.savefig(ti,iter,distance=[10,10],u = u,simm_time=simm_time)
                else:
                    # for 2D case
                    Ns,Nt = self.transform_2d
                    self.savefig(ti,iter,azimut=[-90, -90],elevation=[0,0],distance=[10,10],u = Ns@Nt@u,simm_time=simm_time)



            # print("\th:",norm(h), "\tf:",norm(f_new),"\trho:",rho,"\tTR:",TR_rad, "\tm_h:",m_h, "\tf_h:",norm(f_h), "\tdm:",m_h-m_new)
            print("\th:",norm(h), "\tf:",norm(f_new),"\trho:",rho,"\tTR:",TR_rad, "\tm_h:",m_h, "\tf_h:",norm(f_h), "\tdmda:",f_new@h)
            if TR_rad < 1e-50:
                print("TR_rad is smaller than 1e-50. Stopping the simulation.")
                sys.exit(1)


            for ctct in self.contacts:
                print("actives:",ctct.actives)


        return u, m_new, iter,norm(f_new)

    def TR_backup20241125(self,FUNJAC,HESS, u0, free_ind = None, tol = 1e-10,tol2 = 1e-12,ti=None,simm_time=None,plot=False, unilateral=True):
        if free_ind is None:
            free_ind = self.free

        nfr = len(free_ind)

        TR_rad = 0.01*nfr
        TR_rad_min = 0*nfr
        TR_rad_max = 1*nfr

        ff , m_new = FUNJAC(u0,unilateral=unilateral)
        f_new = ff[free_ind]

        rho = 1
        iter = 0
        update_signal = 1

        h_full = np.zeros_like(u0)

        u = u0.copy()

        while norm(f_new)>tol:
            self.COUNTS[4] += 1


            with open(self.output_dir+"COUNTERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(self.COUNTS_NAMES)
                csvwriter.writerow(self.COUNTS.tolist())


            iter += 1
            print("Iter:",iter)

            if update_signal==1:
                K = HESS(u)
                K = K[np.ix_(free_ind,free_ind)]

                condK = np.linalg.cond(K)

            r_new = -f_new.copy()
            p = r_new.copy()
            h = np.zeros(nfr)
            signal = 1

            while signal==1:
                r_old = r_new.copy()
                alpha = float(r_old@r_old/(p.T@K@p))
                # if p.T@K@p < 0:
                #     set_trace()
                #     signal = 3
                # elif norm(h+alpha*p)<TR_rad:
                if norm(h+alpha*p)<TR_rad and p.T@K@p>0:            # In trust region AND convex direction
                    h += alpha*p
                    r_new = r_old - np.array(alpha*K@p).reshape(-1)
                    beta = (r_new@r_new)/(r_old@r_old)
                    p = r_new+beta*p
                    if norm(r_new)<max(tol2,1e-5*norm(f_new)):
                        signal = 0
                else:
                    a = p@p
                    b = 2*p@h
                    c = h@h - TR_rad**2
                    alpha = 1e-2
                    eq = a*alpha**2 + b*alpha + c
                    while abs(eq)>1e-10:
                        stiff = 2*a*alpha + b
                        d_alpha = -eq/stiff
                        alpha += d_alpha
                        eq = a*alpha**2 + b*alpha + c
                    h += alpha*p
                    signal = 2

            h_full[free_ind] = h.copy()
            ff , m_h = FUNJAC(u + h_full,unilateral=unilateral)
            f_h = ff[free_ind]
        
            rho = 0.5*(f_new+f_h)@h/(0.5*h.T@K@h + f_new.T@h)
            # rho = -(m_new-m_h)/(0.5*h.T@K@h + f_new.T@h)

            if rho<0.25:
                TR_rad = max(0.25*TR_rad,TR_rad_min)
            else:
                
                if (rho>0.75) and (norm(h)<TR_rad*1.01) and (norm(h)>TR_rad*0.99):
                    TR_rad = min(2*TR_rad,TR_rad_max)

            update_signal = 0

            if rho>0.25:
                # print(h.reshape(1,-1))
                u += h_full
                m_new = m_h
                f_new = f_h.copy()
                update_signal = 1


            self.write_list([norm(h),norm(f_new),condK],iter)
            if plot and iter%50==0:                 # Trust-region uses approx 1 eval per iter so we plot only every 10 iters
                if self.transform_2d is None:
                    # 3D case
                    self.savefig(ti,iter,distance=[10,10],u = u,simm_time=simm_time)
                else:
                    # for 2D case
                    Ns,Nt = self.transform_2d
                    self.savefig(ti,iter,azimut=[-90, -90],elevation=[0,0],distance=[10,10],u = Ns@Nt@u,simm_time=simm_time)


            # print("\th:",norm(h), "\tf:",norm(f_new),"\trho:",rho,"\tTR:",TR_rad, "\tm_h:",m_h, "\tf_h:",norm(f_h), "\tdm:",m_h-m_new)
            print("\th:",norm(h), "\tf:",norm(f_new),"\trho:",rho,"\tTR:",TR_rad, "\tm_h:",m_h, "\tf_h:",norm(f_h), "\tdmda:",f_new@h)
            for ctct in self.contacts:
                print("actives:",ctct.actives)


        return u, m_new, iter,norm(f_new)


    def write_list(self,the_list,iter,iter2a=0,iter2=0,case=0):
        for ic, ctct in enumerate(self.contacts):
            pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
            with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([self.t if iter==0 else None,iter]+the_list+ctct.patch_changes)

    def write_m_and_f(self,m,f,iter,iter2a=0,iter2=0,case=0):
        for ic, ctct in enumerate(self.contacts):
            pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
            with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([self.t if iter==0 else None,[iter,iter2a,iter2,case],[m,f]]+ctct.patch_changes)


    def Energy_and_Force(self,u,split=False,show=False,unilateral=True):
        self.COUNTS[5] += 1

        if self.transform_2d is not None:
            Ns,Nt = self.transform_2d
            u = (Ns@Nt@u)

        En,EnC = 0.0, 0.0
        force_body = np.zeros_like(self.fint)
        force_contact = np.zeros_like(self.fint)

        fint_t0 = time.time()
        for body in self.bodies: 

            # force_bi, Ebi = body.compute_mf_plastic(u,self)
            force_bi, Ebi = body.compute_mf_plastic(u,self)
            En += Ebi
            force_body+=force_bi

            # En+=body.compute_m(u)
        self.TIMERS[3] += time.time()-fint_t0
        
        fcon_t0 = time.time()
        for ctct in self.contacts:

            if unilateral:
                mCi,fCi = ctct.compute_mf_unilateral(u,self)
            else:
                mCi,fCi = ctct.compute_mf(u,self)     # Bilateral constraint (active set)
            EnC+=mCi
            force_contact+=fCi
        self.TIMERS[4] += time.time()-fcon_t0

        force = force_body + force_contact

        if show:
            print("Eb:",En,"\tEc:",EnC)

        if self.transform_2d is not None:
            if split:
                return Nt.T@Ns.T@force_body,Nt.T@Ns.T@force_contact,Nt.T@Ns.T@force, En,EnC

            return Nt.T@Ns.T@force, En+EnC

        if split:
            return force_body,force_contact,force, En,EnC
        # return force, En+EnC - self.fext[self.free]@self.u[self.free]
        return force, En+EnC

    def Hessian(self,u):

        if self.transform_2d is not None:
            Ns,Nt = self.transform_2d
            u = (Ns@Nt@u)

        K = sparse.coo_matrix((self.ndof,self.ndof),dtype=float)
        # K = np.zeros((self.ndof,self.ndof),dtype=float)

        Kint_t0 = time.time()
        for body in self.bodies: 
            K += body.compute_k_plastic(u,self)    # uses model.u_temp
        self.TIMERS[5] += time.time()-Kint_t0

        for ctct in self.contacts:
            K += ctct.compute_k(u)     #uses model.u_temp
        # print('En',En,"\tEnC",EnC,'\t\tEn_tot',En+EnC)

        if self.transform_2d is not None:

            return Nt.T@Ns.T@K@Ns@Nt




        return K

        # t0_K = time.time()
        # for body in self.bodies:

        # printif(DispTime,"Getting K : ",time.time()-t0_K,"s")

        # for contact in self.contacts:
        #     contact.getKC(self, DispTime=DispTime)     #uses model.u_temp


    def minimize(self,tol=1e-10,maxiter=10,plotIters=False,ti=None,simm_time=None,method = "BFGS", plot = False):

        if self.transform_2d is not None:
            Ns,Nt = self.transform_2d
            u0 = Nt.T@Ns@np.array(self.u)                                 # This reduces the vector u0
            bool_di = np.zeros(self.ndof)
            bool_di[self.diri] = 1
            di = np.where(Nt.T@bool_di!=0)[0]
            fr = list(set(range(Nt.shape[1]))-set(di))      # free dofs in the reduced model

        else:
            u0=np.array(self.u)
            fr = None

        unilateral = not 'ActSet' in method

        if "LBFGS" in method:
            nli = int(method.replace("LBFGS",""))

            self.u, m_new, iter,res = self.LBFGS(self.Energy_and_Force,u0,nli=nli,free_ind = fr,ti=ti,simm_time=simm_time,plot=plot,unilateral=unilateral)
        elif "BFGS" in method:
            self.u, m_new, iter,res = self.BFGS(self.Energy_and_Force,u0,free_ind = fr,ti=ti,simm_time=simm_time,plot=plot,unilateral=unilateral)

        elif "TR" in method:
            precond = False
            if "diag" in method:
                precond = "diag"
            elif "tril" in method:
                precond = "tril"
            elif "icho" in method:
                precond = "icho"
            elif "same" in method:
                precond = "same"
            self.u, m_new, iter,res = self.TR(self.Energy_and_Force,self.Hessian, u0, free_ind = fr,ti=ti,simm_time=simm_time,plot=plot,unilateral=unilateral,precond=precond)


        if self.transform_2d is not None:
            self.u = (Ns@Nt@self.u)


        return True,res


    def solve_TR_plastic(self,tol=1e-10,maxiter=10,plotIters=False):
        u0=np.array(self.u)
        self.u, m_new, iter = self.TR_plastic(self.Energy_and_Force,self.Hessian,u0)
        return True


    def residual(self, printRes = False):
        # if norm(self.fext) > 1e-14 :
        #     RES = np.linalg.norm(self.fint - self.fext)/norm(self.fext)
        # else:
        #     RES = np.linalg.norm(self.fint - self.fext)
        RES = np.linalg.norm(self.fint - self.fext)
        # print("fext: ",norm(self.fext))
        if printRes: print("resid :",RES)
        return RES


    def Solve(self, t0 = 0, tf = 1, TimeSteps = 1, max_iter=10 , recover = False, ForcedShift = False,
               IterUpdate = False,minimethod = "BFGS",plot=1):
        self.ntstps = TimeSteps
        self.IterUpdate = IterUpdate
        
        dt_base = (tf-t0)/TimeSteps; tolerance = 1e-10; ndof = len(self.X)
        t = t0 ; ti = 0
        # MaxBisect = 2**20   # Here we don't want to use minimization. Let's just see what NR can do
        MaxBisect = 2**10
        dt = dt_base; u_ref = np.array(self.u)
        num = 0
        den = 1
        self.bisect = int(np.log2(den))
        self.Redos_due_to_kn_adjustment = 0


        pickle.dump({body.name:[body.surf.quads,body.surf.nodes] for body in self.bodies},open(self.output_dir+"RecoveryOuters.dat","wb"))
        
        self.setReferences()
        skipBCs = False

        tracing = False

        if recover:
            self.REF,t, dt, ti,[num,den],self.COUNTS,self.TIMERS = pickle.load(open(recover,"rb"))
            self.bisect = int(np.log2(den))
            
            self.getReferences(actives=True)
            for contact in self.contacts:   contact.getCandidates(self.u)
            # for contact in self.contacts:   contact.getCandidatesANN(self.u)

            if ForcedShift:
                rem = t%dt_base     # remaining time from previous time increment
                t_pre = t-rem
                dt = t_pre+dt_base-t
                ForcedShift = False

        DoMinimization = False


        redo_count = 0
        
        print("\ndirectory:\n"+self.output_dir)

        self.solve_timer_0 = time.time()

        while t+dt < tf+1e-4:

            if ForcedShift:
                rem = t%dt_base     # remaining time from previous time increment
                t_pre = t-rem
                dt = t_pre+dt_base-t
                ForcedShift = False
                num = den


            print(" ")
            t += dt
            num = min(num+1,den)

            self.t = t
            print("time: ",round(t,6),"s  -----------------------------------")
            print("("+str(num-1)+"/"+str(den)+" --> "+str(num)+"/"+str(den)+" in the increment)")

            for body in self.bodies:
                body.DELTA_EPcum = np.zeros((len(body.elements),8))


            self.u_temp = np.array(self.u)  # copy of 'u' wihout dirichlet (to be used in Newton ONLY)

            if not skipBCs:
                self.applyBCs(t,t-dt)
            else:
                skipBCs = False

            print("du_diri:",norm(self.u[self.diri]-self.u_temp[self.diri]))

            ########################
            ### Solver Algorithm ###
            ########################
            actives_before_solving = list(self.contacts[0].actives)

            if DoMinimization:
                self.COUNTS[3] += 1
                converged, res = self.minimize(tol=tolerance,ti=ti,simm_time=t,method=minimethod,plot=bool(plot>1))
                self.u_temp = np.array(self.u)  # copy of 'u' so that solution is directly used in NR
            else:
                # if abs(t-0.08)<1e-10:
                #     import pdb; pdb.set_trace()

                converged, res = self.NR(tol=tolerance,maxiter=max_iter)
            actives_after_solving = list(self.contacts[0].actives)

            # print("ACTIVE NODES:")
            # print("Before solving:",actives_before_solving)
            # print("After solving :",actives_after_solving)


            print("delta_epcum:",norm(self.bodies[0].DELTA_EPcum))


            if not converged:
                print('Increment failed to converge !!!!! Redoing half')
                t -= dt
                num -= 1

                self.getReferences(actives=True)    # resets u=u_temp <- u_ref

                pchfile = self.output_dir+"ctct"+str(0)+"iters_details.csv"
                with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                    csvwriter = csv.writer(csvfile)

                    if den >= MaxBisect:
                        print("MAXIMUM BISECTION LEVEL (%s) REACHED:"%MaxBisect)
                        # break
                        DoMinimization=True
                        csvwriter.writerow(['','','MINIMIZATION'])
                    else:

                        num = 2*num
                        den = 2*den
                        self.bisect = int(np.log2(den))

                        dt = dt_base/den

                        print("BISECTION LEVEL: %s:"%int(round(dt_base/dt)))
                        print("dt/dt_base=",dt/dt_base)
                        csvwriter.writerow(['','','Cutback'])

                for ctct in self.contacts:
                    ctct.actives_prev = []

                self.COUNTS[1] += 1

            else:

                #############################
                #### CONVERGED INCREMENT ####
                #############################
                Redo = False
                cycle_found = False
                for ic, ctct in enumerate(self.contacts):

                    actives_prev = list(ctct.actives)    # Actives after solving, before checking
                    acts_bool_prev = [True if el is not None else False for el in actives_prev]
                    
                    ActSet_Updt_t0 = time.time()
                    ctct.getCandidates(self.u, CheckActive = True, TimeDisp=False,tracing=tracing)    # updates Patches->BSs (always) -> candidatePairs (on choice)
                    self.TIMERS[5] += time.time()-ActSet_Updt_t0
                    
                    # ctct.getCandidatesANN(self.u, CheckActive = True, TimeDisp=False,tracing=tracing)    # updates Patches->BSs (always) -> candidatePairs (on choice)
                    print("After checking :",list(ctct.actives))



                    acts_bool = [True if el is not None else False for el in ctct.actives]


                    # if ctct.actives_prev[-1]!=ctct.actives: 
                    # if actives_prev!=ctct.actives: 
                    if acts_bool_prev!=acts_bool: 
                        print("Actives have changed! Will Redo increment...")
                        Redo = True

                        pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
                        with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                            csvwriter = csv.writer(csvfile)
                            entered = [idx for idx in range(ctct.nsn) if (ctct.actives[idx] is not None and actives_after_solving[idx] is None)]
                            exited =  [idx for idx in range(ctct.nsn) if (ctct.actives[idx] is None and actives_after_solving[idx] is not None)]

                            if len(entered)==0 and len(exited)==0:
                                set_trace()

                            csvwriter.writerow(['','','RedoAct','IN']+entered+['OUT']+exited)


                        # CYCLE CHECK
                        if ctct.actives in ctct.actives_prev:
                            print("CYCLE DETECTED!!!")
                            t -= dt
                            num -= 1

                            self.getReferences(actives=True)    # resets u=u_temp <- u_ref

                            pchfile = self.output_dir+"ctct"+str(0)+"iters_details.csv"
                            with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                                csvwriter = csv.writer(csvfile)

                                if den >= MaxBisect:
                                    print("MAXIMUM BISECTION LEVEL (%s) REACHED:"%MaxBisect)
                                    # break
                                    DoMinimization=True
                                    csvwriter.writerow(['','','MINIMIZATION'])
                                else:

                                    num = 2*num
                                    den = 2*den
                                    self.bisect = int(np.log2(den))

                                    dt = dt_base/den

                                    print("BISECTION LEVEL: %s:"%int(round(dt_base/dt)))
                                    print("dt/dt_base=",dt/dt_base)
                                    csvwriter.writerow(['','','Cutback'])

                            self.COUNTS[1] += 1
                            cycle_found = True
                            for ctct in self.contacts:
                                ctct.actives_prev = []

                            break
                    

                if cycle_found:
                    continue

                ctct.actives_prev.append(list(ctct.actives))    # At this point, 'actives' is the observed state


                self.printContactStates(veredict=True)
                
                # if not Redo:
                #     for ctct in self.contacts:
                #         if ctct.checkGNs(): 
                #             Redo = True
                #             self.Redos_due_to_kn_adjustment += 1

                #             pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
                #             with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                #                 csvwriter = csv.writer(csvfile)
                #                 csvwriter.writerow(['','','RedoPen'])



                if Redo:        # don't write "else". It could have changed inside previous block
                    redo_count += 1
                    t-=dt
                    num -= 1
                    # skipBCs=True
                    # u_ref_redo = self.u.copy()
                    self.getReferences()
                    # self.u = u_ref_redo.copy()
                    self.COUNTS[1] += 1

                    continue


                ti += 1
                for ctct in self.contacts:
                    ctct.actives_prev = []

                for body in self.bodies:
                    if not body.isRigid:
                        body.FPconv = body.FPtemp.copy()
                        body.EPcum += body.DELTA_EPcum

                DoMinimization = False

                redo_count = 0


                print("##################")

                if plot:
                    if self.transform_2d is None:
                        # 3D case
                        self.savefig(ti,distance=[10,10],times=[t0,t,tf],fintC=False,Hooks=True)
                        # self.savefig(ti,azimut=[0, 0],elevation=[0,0],distance=[10,10],times=[t0,t,tf],fintC=False,Hooks=True)
                    else:
                        # for 2D case
                        self.savefig(ti,azimut=[-90, -90],elevation=[0,0],distance=[10,10],times=[t0,t,tf],fintC=False,Hooks=True)


                self.setReferences()   

                self.COUNTS[0] += 1   

                sBody = self.contacts[0].slaveBody
                sNodes_diri = []
                for node in range(len(sBody.X)):
                    if sBody.DoFs[node][0] in self.diri:
                        sNodes_diri.append(node)
                self.saveNodalData(sBody.id, t, "fint", nodes = sNodes_diri, Sum=True)
                self.savedata(t,'u','bisect','Redos_due_to_kn_adjustment')




                with open(self.output_dir+"SED.csv", 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(self.bodies[0].get_nodal_SED(self.u).ravel().tolist())

                
                if self.bodies[0].plastic_param[0]<1e20:        # If it is actually plastic
                    with open(self.output_dir+"EPcum.csv", 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(self.bodies[0].EPcum.ravel().tolist())



                self.saveContactData(0,t)

                self.Redos_due_to_kn_adjustment = 0

                if dt != dt_base:
                    fin = False
                    while fin==False:
                        if num%2==0:
                            num /= 2
                            den /= 2
                        else:
                            fin = True
                    num = round(num)    # could have changed to float during simplification but it's still an integer
                    den = round(den)    # could have changed to float during simplification but it's still an integer
                    self.bisect = int(np.log2(den))
                    dt = dt_base/den

                # saving:
                pickle.dump([self.REF,t, dt, ti,[num,den],self.COUNTS,self.TIMERS],open(self.output_dir+"RecoveryData.dat","wb"))

        else:

            self.TIMERS[0] = time.time()-self.solve_timer_0

            print("\n\n############################")
            print("### SIMULATION COMPLETED ###")
            print("############################\n")

        for val,count in zip(self.COUNTS,self.COUNTS_NAMES):
            print(count+":\t",val)


        print("\ndirectory:\n"+self.output_dir)


        with open(self.output_dir+"A_END_COUNTERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.COUNTS_NAMES)
            csvwriter.writerow(self.COUNTS.tolist())
        with open(self.output_dir+"COUNTERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.COUNTS_NAMES)
            csvwriter.writerow(self.COUNTS.tolist())


        with open(self.output_dir+"A_END_TIMERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.TIMERS_NAMES)
            csvwriter.writerow([round(timer, 3) for timer in self.TIMERS])
        with open(self.output_dir+"TIMERS.csv",'w') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.TIMERS_NAMES)
            csvwriter.writerow([round(timer, 3) for timer in self.TIMERS])



"""
                # if iter==126:
                #     set_trace()
                #     import matplotlib.pyplot as plt

                #     alphas = np.linspace(0.0, 500, 100)
                #     f_values = []
                #     for alpha in alphas:

                #         ux = u.copy()
                #         ux[free_ind] = u[free_ind] + alpha*h_new
                #         f , _ = FUNJAC(ux)
                #         f_3 = f[free_ind]
                #         f3 = h_new@f_3
                #         print(f3-h_new@f_new)
                #         print(f3)
                #         f_values.append(f3)

                #     plt.figure(figsize=(8, 6))
                #     plt.plot(alphas, f_values, label="dmda(alpha)", color="blue", linestyle="-", marker="")
                #     plt.xlabel("Alpha")
                #     plt.ylabel("dmda(alpha)")
                #     plt.title("Line Search Plot between alpha_0 and alpha_1")
                #     plt.legend()
                #     plt.grid(True)
                #     plt.show()
"""