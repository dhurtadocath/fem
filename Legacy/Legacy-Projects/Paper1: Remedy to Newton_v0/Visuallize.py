import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
from time import time
import colorsys

cwd = os.path.dirname(os.path.abspath(__file__))


model = pickle.load(open(cwd+"/Model.dat","rb"))
ptt = model.bodies[1]
ptt.Translate([-6,0,0])
ndofs = 3*(len(model.bodies[0].X)+len(ptt.X))
ptt.surf.ComputeGrgPatches(np.zeros(ndofs),range(len(ptt.surf.nodes)))


# Assuming 'cwd' is correctly defined somewhere above this snippet
folderPath = os.path.join(cwd, 'csv_files')
# folderPath = os.path.join(cwd, 'csv_files_edges')

# Step 1: Determine the total number of rows across all CSV files
total_rows = 0
for filename in os.listdir(folderPath):
    if filename.endswith('.csv'):  # Ensuring we only process CSV files
        with open(os.path.join(folderPath, filename), 'r') as file:
            total_rows += sum(1 for _ in file)  # Adjust if your CSVs have headers

# Step 2: Initialize a numpy array with shape (total_rows, 7)
data = np.zeros((total_rows, 7))

# Step 3: Read each CSV file and append its contents to the numpy array
current_index = 0
for filename in os.listdir(folderPath):
    if filename.endswith('.csv'):  # Ensuring we only process CSV files
        file_path = os.path.join(folderPath, filename)
        # Assuming the CSV files do not have headers; if they do, use skiprows=1
        file_data = np.loadtxt(file_path, delimiter=',')
        n_rows_i = file_data.shape[0]
        data[current_index:current_index+n_rows_i, :] = file_data
        current_index += n_rows_i



# set_trace()


# reducing data
num_samples = int(len(data)/2)
indices = np.random.choice(range(len(data)), size=num_samples, replace=False)
data = data[indices]




def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        # Cycle through the hue while keeping saturation and value constant
        hue = i / n
        saturation = 0.7 + 0.3 * (i % 2)  # Alternate saturation to add variety
        value = 0.5 + 0.5 * ((i + 1) % 2)  # Alternate value to add variety
        rgba = colorsys.hsv_to_rgb(hue, saturation, value) + (1.0,)  # Add alpha value of 1
        colors.append(rgba)
    return np.array(colors)



def plot_patches_old(input_string):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    ptt.plot(ax,u= np.zeros_like(ptt.X),ref=10)

    # Process the input string to get a list of patches to display
    patches_to_display = []
    for part in input_string.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            patches_to_display.extend(range(start, end + 1))
        else:
            patches_to_display.append(int(part))
    

    distinct_colors = generate_distinct_colors(len(patches_to_display))
    # Filter points based on the selected patches
    for ip, patch_number in enumerate(patches_to_display):
        # Assuming the patch number corresponds directly to an index in 'distinct_colors'
        patch_color = distinct_colors[ip]
        # Filter points belonging to the current patch
        patch_points = data[data[:,3] == patch_number]
        if len(patch_points) > 0:
            ax.scatter(patch_points[:,0], patch_points[:,1], patch_points[:,2], c=[patch_color], s=0.1)
    
    plt.show()



def plot_patches(input_string):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ptt.plot(ax, u=np.zeros_like(ptt.X), ref=5)

    # Process the input string to get a list of patches to display, along with their inside/outside status
    patches_to_display = []
    conditions = []  # List to hold 'inside' or 'outside' conditions
    for part in input_string.split(','):
        if 'i' in part or 'o' in part:  # Check for inside/outside specifier
            patch_number_str, condition = part[:-1], part[-1]  # Extract patch number and condition ('i' or 'o')
            patch_number = int(patch_number_str)
            patches_to_display.append(patch_number)
            conditions.append(condition)
        else:
            patch_number = int(part)
            patches_to_display.append(patch_number)
            conditions.append(None)  # No condition specified

    distinct_colors = generate_distinct_colors(len(patches_to_display))
    
    for ip, patch_number in enumerate(patches_to_display):
        patch_color = distinct_colors[ip]
        condition = conditions[ip]
        
        if condition == 'i':  # Inside
            patch_points = data[(data[:,3] == patch_number) & (data[:, -1] <= 0)]
        elif condition == 'o':  # Outside
            patch_points = data[(data[:,3] == patch_number) & (data[:, -1] > 0)]
        else:  # No specific condition, include all points for this patch
            patch_points = data[data[:,3] == patch_number]

        if len(patch_points) > 0:
            ax.scatter(patch_points[:,0], patch_points[:,1], patch_points[:,2], c=[patch_color], s=0.1)
    
    plt.show()
    ax.text

# while True:
#     input_string = input("Enter patch numbers: ")
#     # How to see if any string in a list of strings is a substring of  another string?
#     exit_strings = ['q','Q',"exit",'quit','Quit','QUIT','exit','Exit','EXIT']
#     if (input_string in exit_strings) or any(word in input_string for word in exit_strings):
#         break
#     else:
#         try:
#             plot_patches(input_string)
#         except:
#             print('Invalid entry')

# for patches in [[int(2*pi),int(2*pi+1)] for pi in range(int(96/2)) ]:
#     print("plotting patches: ",patches)
#     patches_string = ""
#     for patch in patches:
#         patches_string += str(patch)+'i,'+str(patch)+'o,'
#     plot_patches(patches_string[:-1])

# patches = [2,6,10,14,72,73,74,75,62,58,54,50,23,22,21,20]
# patches = [57,61,71,70,69,68,13,9,53,49,27,26,25,24,1,5]
# patches = [60,56,52,48,31,30,29,28,0,4,8,12,64,65,66,67]
# patches = [95,94,93,92,88,84,80,81,82,83,87,91]
# patches = [89,85,86,90]

patches = range(96)
# patches = [4,5,6,8,9,10]

patches_string = ""
# for patch in patches:
#     patches_string += str(patch)+'i,'
for patch in patches:
    patches_string += str(patch)+'o,'

print("plotting patches: ",patches)
plot_patches(patches_string[:-1])

