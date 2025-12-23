from pdb import set_trace

def InitializeBCs(BClist):      # Function called from FEModel
    BCs = []
    for bci in BClist:
        body, nodes, BCtype, directions, values = bci[:5]

        # assert len(directions) == len(values), "BCs: directions given don't match values"

        if len(bci)>5:
            times = bci[5]
        else:
            times = [0.0 , 1.0]

        direct_idxs = []
        if "x" in directions or "X" in directions:
            direct_idxs.append(0)
        if "y" in directions or "Y" in directions:
            direct_idxs.append(1)
        if "z" in directions or "Z" in directions:
            direct_idxs.append(2)

        # set_trace()

        DoFs = body.DoFs[nodes][:,direct_idxs].ravel()      # flat array of DoFs involved

        BCs.append(BC(BCtype, values, DoFs, times[0], times[1]))
    return BCs


#############
### CLASS ###
#############

class BC:       # Doesn't have methods, just attributes
    def __init__(self, BCtype, vals, DoFs, t0,tf):
        self.BCtype = BCtype    # Dirichlet: 0  Neumann: 1
        self.vals = vals
        self.DoFs = DoFs
        self.t0 = t0
        self.tf = tf

