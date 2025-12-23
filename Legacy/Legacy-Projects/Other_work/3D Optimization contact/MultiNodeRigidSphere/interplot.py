"""import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_axes(111)

X = np.array([0,1,2,3,4,5])                                                                                                        
y_ascend1 = 1*np.array([0,1,2,3,4,5])                                                                                         
y_ascend2 = 2*np.array([0,1,2,3,4,5])                                                                                         
y_ascend3 = 3*np.array([0,1,2,3,4,5])                                                                                         
y_descend = np.array([5,4,3,2,1,0])  


l1=[]

l1.append(ax.plot(X,y_ascend1))
l1.append(ax.plot(X,y_ascend2))
l1.append(ax.plot(X,y_ascend3))
l2 = ax.plot(X,y_descend)


# l1[0][0].remove()

plt.show()


"""

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_axes(111)

X = np.array([0,1,2,3,4,5])                                                                                                        
y_ascend1 = 1*np.array([0,1,2,3,4,5])                                                                                         
y_ascend2 = 2*np.array([0,1,2,3,4,5])                                                                                         
y_ascend3 = 3*np.array([0,1,2,3,4,5])                                                                                         
y_descend = np.array([5,4,3,2,1,0])  

# Plot the ascending lines and store the Line2D objects in a list
ascending_lines = []
ascending_lines.append(ax.plot(X, y_ascend1)[0])
ascending_lines.append(ax.plot(X, y_ascend2)[0])
ascending_lines.append(ax.plot(X, y_ascend3)[0])

# Plot the descending line and store the Line2D object
line_descend, = ax.plot(X, y_descend)

# In this line, I would like to delete all the ascending lines in ascending_lines
for line in ascending_lines:
    line.remove()

# You can also clear the list if you no longer need references to the removed lines
ascending_lines.clear()

plt.show()
