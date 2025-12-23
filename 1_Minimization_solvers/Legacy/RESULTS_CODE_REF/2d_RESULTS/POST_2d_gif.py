import sys
import os
import pickle
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd+"/..")      #this will actually refer to the good FEAssembly.py file

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pdb import set_trace
from PyClasses import FEModel
from PyClasses.Utilities import plot_coords
import imageio


def center_crop(image, width=None, height=None):
    assert width or height, 'At least one of width or height must be specified!'
    # use specified value, or fall back to the other one
    width = width or height
    height = height or width
    # determine/calculate relevant values
    old_height, old_width = image.shape[:2]
    c_x, c_y = old_width // 2, old_height // 2
    dx, dy = width // 2, height // 2
    # perform the crop
    return image[c_y-dy:c_y+dy, c_x-dx:c_x+dx]


def generate_gif(model, TIMES, UU, SED, gif_filename='animation.gif'):
    from matplotlib.colors import Normalize
    from matplotlib import cm

    # Ensure the folder for frames exists
    if not os.path.exists('frames'):
        os.makedirs('frames')
    
    sed = np.array(SED)
    smin, smax = [sed.min(), sed.max()]
    sed = (sed - smin) / (smax / 4 - smin)  # normalize

    # Create a figure for plotting
    # fig = plt.figure()
    fig = plt.figure(figsize=(24, 12))  # Adjust the width (12) and height (8) as needed

    ax = fig.add_subplot(111, projection='3d')

    # Set limits for the plot based on the model
    limits = np.zeros((3, 2))
    for body in model.bodies:
        x_all = body.X + UU[:, body.DoFs]
        for i in range(3):
            if x_all[:, :, i].min() < limits[i, 0]:
                limits[i, 0] = x_all[:, :, i].min()
            if x_all[:, :, i].max() > limits[i, 1]:
                limits[i, 1] = x_all[:, :, i].max()

    limits[2][0] = 1.0
    ax.set_xlim3d(limits[0, 0], limits[0, 1])
    ax.set_ylim3d(limits[1, 0], limits[1, 1])
    # ax.set_zlim3d(limits[2, 0], limits[2, 1])
    ax.set_zlim3d(0, 5)
    ax.set_aspect("equal")
    # ax.axis('off')

    # Create the color map and normalization for the colorbar
    cmap = cm.get_cmap('jet')
    # norm = Normalize(vmin=smin, vmax=smax/4)  # Adjust as needed based on your SED data

    # # Add colorbar to the plot
    # cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, aspect=10)
    # cbar.set_label('Strain Energy Density')  # Adjust label as necessary
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=smin,vmax=smax/4)),
                         ax=ax, shrink=0.2, aspect=30,extend='max',orientation='horizontal',
                          location='bottom',pad = 0.0, anchor = (0.5,3.4))
    cbar.set_label('Strain Energy Density')
    # cbar.set_label('Plastic Multiplier')



    axin2 = ax.inset_axes([-.08, -.022, 0.02, 0.02], projection='3d',transform = ax.transData,zorder=1000000)
    # axin2 = ax.inset_axes([-.08, 0.0, 0.02, 0.02], projection='3d',transform = ax.transData,zorder=1000000)
    axin2.axis('off')
    axin2.set_xlim3d(0, 1)
    axin2.set_ylim3d(0, 1)
    axin2.set_zlim3d(0, 1)


    if model.transform_2d is not None:
        ax.set_proj_type('ortho')
        ax.view_init(elev=0, azim=-90, roll=0)


    else:
        ax.view_init(elev=5, azim=-60)

    axin2.view_init(elev=ax.elev, azim=ax.azim)

    # Store the frame file names
    frame_files = []

    for idx, time in enumerate(TIMES):
        if abs(100 * time - round(100 * time)) < 1e-10:  # Only consider integer time steps
            # Clear the previous plot, except the rigid body
            ax.cla()
            ax.set_xlim3d(limits[0, 0], limits[0, 1])
            ax.set_ylim3d(limits[1, 0], limits[1, 1])
            # ax.set_zlim3d(limits[2, 0], limits[2, 1])
            ax.set_zlim3d(0, 5)
            ax.set_aspect("equal")
            ax.axis('off')

            # Plot the deformed body
            u = UU[idx]
            sedi = sed[idx]
            model.bodies[1].surf.plot(ax, u=None, sed=None, ref=10, onlyMaster=True)
            model.bodies[0].plot(ax, u=u, sed=sedi, ref=10)
            model.bodies[0].plot(ax, u=u, sed=None, ref=1)

            # Add the time label
            ax.text2D(-0.05, 0.7, f"time: {round(time, 8)}", transform=ax.transAxes,
                      backgroundcolor=(1.0, 1.0, 1.0, 0.7), fontsize='x-large')
            
            # Save the frame
            frame_filename = f'frames/frame_{idx:04d}.png'
            plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0.1)
            frame_files.append(frame_filename)

    # Create GIF
    images = [imageio.imread(frame) for frame in frame_files]
    imageio.mimsave(gif_filename, images, fps=10, loop=0)  # Adjust fps as needed

    # Optionally, clean up the frame files
    for frame in frame_files:
        os.remove(frame)

    print(f"GIF saved as {gif_filename}")



with open(cwd + "/Model.dat", "rb") as file:
    model = pickle.load(file)

ptt = model.bodies[1]
ndofs = 3*(len(model.bodies[0].X)+len(ptt.X))
ptt.surf.ComputeGrgPatches(np.zeros(ndofs),model.contacts[0].masterNodes,exactNodesGiven=True)

data = np.loadtxt(cwd+'/data_u.csv', delimiter=',')
TIMES = data[:,0]
UU = data[:,1:]

SED = []
EPCUM = []
epcum_gauss = np.loadtxt(cwd+'/EPcum.csv', delimiter=',').reshape(len(UU),-1,8)

for incr,u in enumerate(UU):
    SED.append(model.bodies[0].get_nodal_SED(u))
    # EPCUM.append(model.bodies[0].get_nodal_EPCUM(epcum_gauss[incr]))



# Assuming your model and data are already loaded:
generate_gif(model, TIMES, UU, SED)
