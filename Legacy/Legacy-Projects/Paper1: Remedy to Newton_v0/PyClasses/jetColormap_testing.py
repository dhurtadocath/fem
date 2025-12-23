import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with your FEM data)
values = np.random.rand(10, 10)

# Plot the values using the 'jet' colormap
plt.imshow(values, cmap='jet', interpolation='none')
plt.colorbar(label='Scalar Values')  # Add a colorbar for reference
plt.title('FEM Visualization with Jet Colormap')
plt.show()
