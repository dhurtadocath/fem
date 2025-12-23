from PyClasses.BoundaryConditions import *
from PyClasses.Utilities import float_to_fraction, printif, checkVarsSize, flatList,plot_coords
import PyClasses.FEAssembly
from scipy import sparse
from scipy.sparse.linalg import spsolve, cgs, bicg, gmres, lsqr
from scipy.linalg import solve as slsolve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
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
        dir = "OUTPUT_"+current_time+self.Mainname+"/"
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.output_dir = dir
        pickle.dump(self,open(dir+"Model.dat","wb"))


    def printContactStates(self,only_actives=True,veredict=False):
        for i_c, contact in enumerate(self.contacts):
            print("contact",i_c,":")
            contact.printStates(only_actives=only_actives,veredict=veredict)

    def plot(self, ax, ref = 10, specialPatches = None, almostSpecialPatches = None,
              specialNodes = None, time = None, fintC = False, plotHooks=False):
        for body in self.bodies:
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
            ax.text2D(0.05, 0.95, "time: "+str(round(time,6)), transform=ax.transAxes)

    def plotNow(self, fintC = False):
        #PLOT OPTIONS
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('ortho')
        # plt.axis('off')

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
        self.plot(ax, ref = 10,specialPatches = None,almostSpecialPatches = None,specialNodes = {"blue": activeNodes}, fintC= fintC)
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
        ax.set_zlim3d((cz-0.8*hg, cz+0.8*hg))

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
                              specialNodes = {"blue": stickNodes,"red":slipNodes}, 
                              time = t,
                              fintC = fintC,
                              plotHooks = Hooks)

        #Camera effects
        tt = (t-t0)/(tf-t0)
        ax.view_init(elev=elevation[0]*(1-tt)+elevation[1]*tt,azim=azimut[0]*(1-tt)+azimut[1]*tt)
        ax.dist = distance[0]*(1-tt)+distance[1]*tt     #default dist = 10
        ax.set_proj_type('ortho')
        plot_coords(ax,orig=(minx-0.5+1.0,miny-0.5,minz-0.5))

        if iterative==None:
            plt.savefig(self.output_dir+"plots/fig"+str(ti)+".png",dpi = dpi)
        else:
            plt.savefig(self.output_dir+"plots/fig"+str(ti)+"-"+str(iterative)+".png",dpi = dpi)
        plt.close()



    def plot4D(self,UU,X,C, ax=None, undef=False):

        sed = np.array(self.SED)
        smin,smax = [min(sed),max(sed)]
        sed = (sed-smin)/(smax-smin)            # normalizing everything between 0 and 1

        if ax is None:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider, Button

            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')

        init_time = 0




        plotSphere(Circ,ax=ax)
        u = UU[0]
        plotTruss(X+u.reshape(-1,3),C,ax=ax,bar_types=btypes,show_nodes=True,showtypes=0)
        
        axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])

        time_slider = Slider(
            ax=axtime,
            label='time',
            valmin=0.0,
            valmax=0.99,
            valstep=0.01,
            valinit=init_time,
        )


        # The function to be called anytime a slider's value changes
        def update(val):
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            zl = ax.get_zlim()
            az, el = ax.azim, ax.elev
            ax.clear()
            idx = round(100*time_slider.val)
            plotSphere(Circ,ax=ax)
            acts = np.asarray(ACTIVES[idx]).nonzero()[0]
            actives = [slaves[i] for i in acts]
            plotTruss(X+UU[idx].reshape(-1,3),C,ax=ax,bar_types=btypes,show_nodes=True,showtypes=0,actives=actives)
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            ax.set_zlim(zl)
            ax.azim = az
            ax.elev = el

        # register the update function with each slider
        time_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')


        def reset(event):
            time_slider.reset()
        button.on_clicked(reset)

        plt.show()





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
            filename = self.output_dir+(arg if name is None else name)+".csv"
    
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

    def get_fint(self, DispTime = False, temp = False):
        t0_fint = time.time()
        self.fint = np.zeros_like(self.u,dtype=float)
        for body in self.bodies: body.get_fint(self, temp = temp)
        if DispTime: print("Getting fint : ",time.time()-t0_fint,"s")

        u = self.u if temp==False else self.u_temp
        for ctct in self.contacts:
            if temp: ctct.patch_changes = []                 # to keep track of the changes during iterations
            sDoFs  = ctct.slaveBody.DoFs[ctct.slaveNodes]
            xs = np.array(ctct.slaveBody.X )[ctct.slaveNodes ] + np.array(u[sDoFs ])      # u_temp
            ctct.xs = xs
            ctct.getfintC(self, DispTime=DispTime)      # uses xs


    def get_K(self, DispTime = False):
        t0_K = time.time()
        self.K = sparse.coo_matrix((self.ndof,self.ndof),dtype=float)
        for body in self.bodies:
            body.get_K(self)    # uses model.u_temp

        printif(DispTime,"Getting K : ",time.time()-t0_K,"s")

        for contact in self.contacts:
            contact.getKC(self, DispTime=DispTime)     #uses model.u_temp


    def solve_NR(self,tol=1e-10,maxiter=10,plotIters=False):
        # NEWTON-RAPHSON
        di, fr = self.diri, self.free
        TimeDisp = False
        RES, niter = 1+tol, 0

        self.get_fint(DispTime=TimeDisp,temp=True)   # uses self.u_temp     (no dirichlet)


        dua = self.u[di] - self.u_temp[di]      # difference due to dirichlet BCs applied
        while RES>tol:
            # getting K and KC
            self.get_K(DispTime=TimeDisp)  # <-- uses self.u_temp           (no dirichlet)

            # Linear System

            Kaa=self.K[np.ix_(di,di)]
            Kab=self.K[np.ix_(di,fr)]
            Kba=self.K[np.ix_(fr,di)]
            Kbb=self.K[np.ix_(fr,fr)]

            fina=self.fint[di]
            finb=self.fint[fr]
            fexb=self.fext[fr]      # always zero (?)

            dub =spsolve(Kbb,fexb-finb-Kba.dot(dua))     # Uses all available processors
            self.fext[di] = fina + Kaa.dot(dua) + Kab.dot(dub)

            dua = np.zeros_like(self.u[di])

            self.u[fr] += dub


            self.get_fint(DispTime=TimeDisp)   # uses self.u

            RES = self.residual(printRes=True)
            
            for ic, ctct in enumerate(self.contacts):
                pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
                with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([self.t if niter==0 else None,niter,RES]+ctct.patch_changes)

            if plotIters: self.plotContactNow()

            niter += 1
            self.u_temp = np.array(self.u)

            if (niter == maxiter and RES>tol) or np.isnan(RES) or RES>1e+15:
                print('Increment failed to converge !!!!! Redoing half')
                return False

        return True            


    def BFGS(self,FUN, JAC, u0, tol = 1e-8, tol2 = 1e-10):
        alpha_init = 1
        c_par = 1e-4
        c_par2 = 0.9
        r_par = 0.5
        
        free_ind = self.free
        nfr = len(free_ind)
        m_new = FUN(u0)
        f = JAC(u0)
        self.write_m_and_f(m_new,norm(f),0)
        u = u0.copy()
        f_2 = f[free_ind]
        K_new_inv = np.eye(nfr)
        f_new = np.zeros(nfr)
        
        for ctct in self.contacts:
            ctct.patch_changes = []

        iter = 0
        while np.linalg.norm(f_2 - f_new) > tol and np.linalg.norm(f_2) > 0:
            printalphas = True
            iter += 1
            f_old = f_new.copy()
            f_new = f_2.copy()
            K_old_inv = K_new_inv.copy()
            delta_f = f_new - f_old
            
            if iter == 1:
                h_new = -np.dot(K_old_inv, f_new)
            else:
                K_new_inv = K_old_inv + ((np.inner(delta_u, delta_f) + np.inner(delta_f, np.dot(K_old_inv, delta_f)))*(np.outer(delta_u,delta_u)))/ (np.dot(delta_u, delta_f) ** 2)- (np.outer(np.dot(K_old_inv, delta_f),delta_u) + np.inner(np.outer(delta_u, delta_f),K_old_inv)) / np.dot(delta_u, delta_f)
                h_new = -np.dot(K_new_inv, f_new)
            
            alpha3 = alpha_init
            ux = u.copy()
            ux[free_ind] = u[free_ind] + alpha3 * h_new
            m_3 = FUN(ux)
            f = JAC(ux)
            f_3 = f[free_ind]
            self.write_m_and_f(m_3,norm(f_3),iter)
            
            signal1 = 0
            if m_3 <= m_new + c_par * alpha3 * np.dot(h_new, f_new) and abs(np.dot(h_new, f_3)) <= c_par2 * abs(np.dot(h_new, f_new)):
                signal1 = 1
            
            iter2a = 0
            while m_3 < m_new + c_par * alpha3 * np.dot(h_new, f_new) and np.dot(h_new, f_3) < -c_par2 * np.dot(h_new, f_new) and signal1 == 0:
                alpha3 = alpha3 / r_par
                ux = u.copy()
                ux[free_ind] = u[free_ind] + alpha3 * h_new
                m_3 = FUN(ux)
                f = JAC(ux)
                self.write_m_and_f(m_3,norm(f_3),iter,iter2a=iter2a)
                f_3 = f[free_ind]
                iter2a += 1
                if m_3 <= m_new + c_par * alpha3 * np.dot(h_new, f_new) and abs(np.dot(h_new, f_3)) <= c_par2 * abs(np.dot(h_new, f_new)):
                    signal1 = 1
            printif(printalphas,"alpha3:",alpha3)
            if signal1 == 0:
                alpha1 = 0
                alpha2 = alpha3 / 2
                ux = u.copy()
                ux[free_ind] = u[free_ind] + alpha2 * h_new
                m_2 = FUN(ux)
                f = JAC(ux)
                self.write_m_and_f(m_3,norm(f_3),iter,iter2a=iter2a)
                f_2 = f[free_ind]
                signal2 = 0
                iter2 = 0
                while signal2 == 0:
                    iter2 += 1
                    if alpha3 - alpha1 < tol2:
                        signal2 = 1
                        m_2 = m_new
                        f_2 = f_new
                    elif m_2 > m_new + c_par * alpha2 * np.dot(h_new, f_new):
                        alpha3 = alpha2
                        m_3 = m_2
                        f_3 = f_2
                        alpha2 = 0.5 * (alpha1 + alpha2)
                        ux = u.copy()
                        ux[free_ind] = u[free_ind] + alpha2 * h_new
                        m_2 = FUN(ux)
                        f = JAC(ux)
                        self.write_m_and_f(m_3,norm(f_3),iter,iter2a=iter2a,iter2=iter2)
                        f_2 = f[free_ind]
                    else:
                        if np.dot(h_new, f_2) < c_par2 * np.dot(h_new, f_new):
                            alpha1 = alpha2
                            alpha2 = 0.5 * (alpha2 + alpha3)
                            ux = u.copy()
                            ux[free_ind] = u[free_ind] + alpha2 * h_new
                            m_2 = FUN(ux)
                            f = JAC(ux)
                            self.write_m_and_f(m_3,norm(f_3),iter,iter2a=iter2a,iter2=iter2)
                            f_2 = f[free_ind]
                        elif np.dot(h_new, f_2) > -c_par2 * np.dot(h_new, f_new):
                            alpha3 = alpha2
                            m_3 = m_2
                            f_3 = f_2
                            alpha2 = 0.5 * (alpha1 + alpha2)
                            ux = u.copy()
                            ux[free_ind] = u[free_ind] + alpha2 * h_new
                            m_2 = FUN(ux)
                            f = JAC(ux)
                            self.write_m_and_f(m_3,norm(f_3),iter,iter2a=iter2a,iter2=iter2)
                            f_2 = f[free_ind]
                        else:
                            signal2 = 1
                printif(printalphas,"alpha3:",alpha3, "  (after signal2=0)")
                            
            delta_u = ux[free_ind] - u[free_ind]
            u = ux
            if signal1 == 1:
                m_new = m_3
                f_2 = f_3.copy()

        return u, m_new, iter


    def write_m_and_f(self,m,f,iter,iter2a=0,iter2=0):
        for ic, ctct in enumerate(self.contacts):
            pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
            with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([self.t if iter==0 else None,[iter,iter2a,iter2],[m,f]]+ctct.patch_changes)



    def Energy(self,u):
        En,EnC = 0.0, 0.0
        for body in self.bodies: 
            En+=body.compute_m(u)
        for ctct in self.contacts:
            EnC+=ctct.compute_m(u)
        # print('En',En,"\tEnC",EnC,'\t\tEn_tot',En+EnC)
        return En+EnC
    
    def Force(self,u):
        force = np.zeros_like(self.fint)
        for body in self.bodies: 
            force+=body.compute_f(u,self)  #called like that to differentiate from get_fint which takes u=Model.u and modifies Model.fint
        for ctct in self.contacts:
            force+=ctct.compute_f(u,self)
        # print('Force',np.linalg.norm(force))
        force[self.diri]=0.0
        return force




    def solve_BFGS(self,tol=1e-10,maxiter=10,plotIters=False):
        u0=np.array(self.u)
        # MIN = minimize(self.Energy,u0,method='BFGS',jac=self.Force,options={'disp':False})
        # self.u = MIN.x
        self.u, m_new, iter = self.BFGS(self.Energy,self.Force,u0)
        return True


    def residual(self, printRes = False):
        RES = np.linalg.norm(self.fint - self.fext)
        if printRes: print("resid :",RES)
        return RES


    def Solve(self, t0 = 0, tf = 1, TimeSteps = 1, max_iter=10 , recover = False, ForcedShift = False):
        self.ntstps = TimeSteps
        
        dt_base = (tf-t0)/TimeSteps; tolerance = 1e-10; ndof = len(self.X)
        t = t0 ; ti = 0
        MaxBisect = 32
        dt = dt_base; u_ref = np.array(self.u)

        pickle.dump({body.name:[body.surf.quads,body.surf.nodes] for body in self.bodies},open(self.output_dir+"RecoveryOuters.dat","wb"))
        
        self.setReferences()
        skipBCs = False
        tmp = 0

        tracing = False

        if recover:
            self.REF,t, dt, ti = pickle.load(open(self.output_dir+"RecoveryData.dat","rb"))
            self.getReferences(actives=True)
            for contact in self.contacts:   contact.getCandidates(self.u)

            if ForcedShift:
                rem = t%dt_base     # remaining time from previous time increment
                t_pre = t-rem
                dt = t_pre+dt_base-t
                ForcedShift = False

        DoMinimization = False

        while t+dt < tf+1e-4:
            print(" ")
            t += dt
            self.t = t
            print("time: ",round(t,6),"s  -----------------------------------")


            self.u_temp = np.array(self.u)  # copy of 'u' wihout dirichlet (to be used in Newton ONLY)

            if not skipBCs:
                self.applyBCs(t,t-dt)
            else:
                skipBCs = False

            ########################
            ### Solver Algorithm ###
            ########################
            if DoMinimization:
                converged = self.solve_BFGS(tol=tolerance)
                self.u_temp = np.array(self.u)  # copy of 'u' so that solution is directly used in NR
            converged = self.solve_NR(tol=tolerance,maxiter=max_iter)


            if not converged:
                print('Increment failed to converge !!!!! Redoing half')
                t -= dt
                self.getReferences(actives=True)    # resets u=u_temp <- u_ref


                pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
                with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                    csvwriter = csv.writer(csvfile)

                    if int(round(dt_base/dt)) > MaxBisect:
                        print("MAXIMUM BISECTION LEVEL (%s) REACHED:"%MaxBisect)
                        DoMinimization=True
                        csvwriter.writerow(['','','MINIMIZATION'])
                    else:
                        dt = dt/2
                        print("BISECTION LEVEL: %s:"%int(round(dt_base/dt)))
                        print("dt/dt_base=",dt/dt_base)
                        csvwriter.writerow(['','','Cutback'])
                tmp += 1

            else:
                #############################
                #### CONVERGED INCREMENT ####
                #############################
                Redo = False
                for ic, ctct in enumerate(self.contacts):
                    ctct.actives_prev.append(list(ctct.actives))
                    ctct.getCandidates(self.u, CheckActive = True, TimeDisp=False, tracing=tracing)    # updates Patches->BSs (always) -> candidatePairs (on choice)
                    if ctct.actives_prev[-1]!=ctct.actives: 
                        print("Actives have changed! Will Redo increment...")
                        Redo = True

                        pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
                        with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                            csvwriter = csv.writer(csvfile)
                            entered = list(set(ctct.actives).difference(set(ctct.actives_prev[-1])))
                            exited  = list(set(ctct.actives_prev[-1]).difference(set(ctct.actives)))

                            csvwriter.writerow(['','','RedoAct','IN']+entered+['OUT']+exited)



               
                self.printContactStates(veredict=True)
                
                if not Redo:
                    for ctct in self.contacts:
                        if ctct.checkGNs(): 
                            Redo = True

                            pchfile = self.output_dir+"ctct"+str(ic)+"iters_details.csv"
                            with open(pchfile, 'a') as csvfile:        #'a' is for "append". If the file doesn't exists, cretes a new one
                                csvwriter = csv.writer(csvfile)
                                csvwriter.writerow(['','','RedoPen'])




                if Redo:
                    t-=dt
                    skipBCs=True
                    tmp += 1
                    continue

                ti += 1
                for ctct in self.contacts:
                    ctct.actives_prev = []

                DoMinimization=False

                for ib, body in enumerate(self.bodies):
                    sed_incr = body.get_nodal_SED(self.u)
                    self.SED[ib].append(sed_incr)


                print("##################")
                # self.savefig(ti,azimut=[-60, -60],elevation=[20,20],distance=[10,10],times=[t0,t,tf],fintC=False,Hooks=True)
                # self.savefig(ti,azimut=[-90, -90],elevation=[0,0],distance=[10,10],times=[t0,t,tf],fintC=False,Hooks=True)

                self.setReferences()      

                sBody = self.contacts[0].slaveBody
                sNodes_diri = []
                for node in range(len(sBody.X)):
                    if sBody.DoFs[node][0] in self.diri:
                        sNodes_diri.append(node)
                self.saveNodalData(sBody.id, t, "fint", nodes = sNodes_diri, name="fint", Sum=True)
                self.saveNodalData(sBody.id, t, "u", nodes = "all", name=None, Sum=False)
                self.saveNodalData(sBody.id, t, "u", nodes = [sNodes_diri[0]], name="dispSlave", Sum=True)
                self.savedata(t,'u')

                self.saveContactData(0,t)

                if dt != dt_base:
                    ratio = t/dt_base - int((t-t0)/dt_base)
                    dt = dt_base/(float_to_fraction(ratio)[1])

                # saving:
                pickle.dump([self.REF,t, dt, ti],open(self.output_dir+"RecoveryData.dat","wb"))

        else:
            print(self.u)
            print("Increments Redone : ",tmp)

