# Abaqus-python distribution does not support pickle to read saved dictionary

# Importing basic IO libraries
import pickle
import numpy as np
from pdb import set_trace

# import Diego's model
Model = pickle.load(open("data/Model.dat","rb"))

set_trace()

# Identifying keys of the model
ModelKeys = np.array(['pos', 'connect', 'EA', 'mu', 'ID_slaves', 'ID_base',
                      'Sph_pos', 'Sph_R', 'dirichlet_base'])

print(np.shape(Model[ModelKeys[0]]))
print(np.shape(Model[ModelKeys[1]]))
np.savetxt('data/'+ModelKeys[0]+'.csv',np.array(Model[ModelKeys[0]]), delimiter=',')
np.savetxt('data/'+ModelKeys[1]+'.csv',np.array(Model[ModelKeys[1]],dtype=int), delimiter=',')
np.savetxt('data/'+ModelKeys[4]+'.csv',np.array(Model[ModelKeys[4]]), delimiter=',')
np.savetxt('data/'+ModelKeys[5]+'.csv',np.array(Model[ModelKeys[5]]), delimiter=',')

