import sys
import os
import pickle
cwd = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(cwd+"/..")      #this will actually refer to the good FEAssembly.py file
sys.path.append(cwd+"/../../..")      #this will actually refer to the good FEAssembly.py file

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pdb import set_trace
from PyClasses import FEModel
from PyClasses.Utilities import plot_coords
import imageio
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.gridspec import GridSpec


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


def generate_gif(model, TIMES, UU, SED, plastic,gif_filename='animation.gif'):
    from matplotlib.colors import Normalize
    from matplotlib import cm

    # Use LaTeX fonts
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    # Ensure the folder for frames exists
    if not os.path.exists('frames'):
        os.makedirs('frames')
    
    sed = np.array(SED)
    smin, smax = [sed.min(), sed.max()]
    sed = (sed - smin) / (smax / 4 - smin)  # normalize

    # Create a figure for plotting
    # fig = plt.figure()
    fig = plt.figure(figsize=(24, 12))  # Adjust the width (12) and height (8) as needed
    gs = GridSpec(4, 2, width_ratios=[85, 15], height_ratios=[1, 1, 1, 1], wspace=-0.4, hspace=0.5)

    ax = fig.add_subplot(gs[:, 0], projection='3d')
    ax_fx = fig.add_subplot(gs[1, 1])
    ax_fz = fig.add_subplot(gs[2, 1])

    # Read the data from 'sum_fint.csv'
    data = pd.read_csv(os.path.join(cwd,'sum_fint.csv'), header=None)
    times = data[0]
    sumFx = data[1]
    sumFz = data[3]

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
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=smin,vmax=smax/4)),
    #                      ax=ax, shrink=0.2, aspect=30,extend='max',orientation='horizontal',
    #                       location='bottom',pad = 0.0, anchor = (0.5,3.4))
    # cbar.set_label('Strain Energy Density')


    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=smin,vmax=smax/4)),
                         ax=ax, shrink=0.2, aspect=30, extend='max', orientation='horizontal',
                         location='bottom',pad = 0.0, anchor = (0.515,3.35))
                        # location='bottom', pad=0.0, anchor=(0.5, 2.0))
    ticks = np.linspace(smin, smax/4, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"${tick:.4f}$" for tick in ticks])

    if plastic:
        cbar.set_label(r'$\mathrm{Plastic\ Multiplier}$')
    else:
        cbar.set_label(r'$\mathrm{Strain\ Energy\ Density}$')




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
        # if abs(100 * time - round(100 * time)) < 1e-10:  # Only consider integer time steps
        if (idx%3==0) or (abs(100 * time - round(100 * time)) < 1e-10) or (idx==len(TIMES)-1):  # Only consider integer time steps

            exact_time = abs(100 * time - round(100 * time)) < 1e-10

            # # some values only, for testing
            # if not 100*round(time,2)%10==0:
            #     continue

            # Clear the previous plot, except the rigid body
            axin2.cla()
            ax.cla()
            ax.set_xlim3d(limits[0, 0], limits[0, 1])
            ax.set_ylim3d(limits[1, 0], limits[1, 1])
            # ax.set_zlim3d(limits[2, 0], limits[2, 1])
            ax.set_zlim3d(0, 5)
            ax.set_aspect("equal")
            ax.axis('off')


            axin2 = ax.inset_axes([-.087, -.018, 0.02, 0.02], projection='3d',transform = ax.transData,zorder=1000000, facecolor='none')
            # axin2 = ax.inset_axes([-.08, 0.0, 0.02, 0.02], projection='3d',transform = ax.transData,zorder=1000000)
            axin2.axis('off')
            axin2.set_xlim3d(0, 1)
            axin2.set_zlim3d(0, 1)
            axin2.set_proj_type('ortho')


            print(f"time: {time:.10f}")

            # Plot the deformed body
            u = UU[idx]
            sedi = sed[idx]
            model.bodies[1].surf.plot(ax, u=None, sed=None, ref=10, onlyMaster=True)
            model.bodies[0].plot(ax, u=u, sed=sedi, ref=10)
            model.bodies[0].plot(ax, u=u, sed=None, ref=1)

            # Add the time label
            ax.text2D(0.05, 0.7, f"$\mathrm{{time: {time:.2f}}}$" if exact_time else f"$\mathrm{{time: {time:.10f}}}$", transform=ax.transAxes,
                      backgroundcolor=(1.0, 1.0, 1.0, 0.7), fontsize='x-large')
            
            if not exact_time:
                ax.set_facecolor((0.9, 0.9, 0.9))  # Brighter gray
            else:
                ax.set_facecolor('white')
            
            axin2.view_init(elev=ax.elev, azim=ax.azim)
            plot_coords(axin2)

            # Update Fx vs time plot
            ax_fx.cla()
            # ax_fx.plot(times, sumFx, color='gray', linewidth=0.5)
            ax_fx.plot(times[times <= time], sumFx[times <= time], color='blue', linewidth=2)
            # import pdb; pdb.set_trace()
            try:
                ax_fx.scatter([time], [sumFx[np.isclose(times, time, rtol=1e-10)]], color='blue', s=50) # was 1e-12
            except:
                import pdb; pdb.set_trace()
            ax_fx.set_xlim(0, 1)
            ax_fx.set_xlabel(r'$\mathrm{Time}$')
            ax_fx.set_ylabel(r'$\mathrm{Fx}$')
            ax_fx.set_title(r'$\mathrm{Horizontal\ force}$')

            # Update Fz vs time plot
            ax_fz.cla()
            # ax_fz.plot(times, sumFz, color='gray', linewidth=0.5)
            ax_fz.plot(times[times <= time], sumFz[times <= time], color='blue', linewidth=2)
            # ax_fz.scatter([time], [sumFz[times == time]], color='blue', s=50)
            ax_fz.scatter([time], [sumFz[np.isclose(times, time, rtol=1e-10)]], color='blue', s=50)
            ax_fz.set_xlim(0, 1)
            ax_fz.set_xlabel(r'$\mathrm{Time}$')
            ax_fz.set_ylabel(r'$\mathrm{Fz}$')
            ax_fz.set_title(r'$\mathrm{Vertical\ force}$')

            reps = 2 if exact_time else 1
            for rep in range(reps):
                # Save the frame
                frame_filename = f'frames/frame_{idx:04d}_rep_{rep}.png'
                plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0.1)
                frame_files.append(frame_filename)


    # Add 10 more frames of the last saved figure without changes
    # if round(time, 2) == 1.0:
    for extra_idx in range(10, 30):
        extra_frame_filename = f'frames/frame_{idx:04d}_extra_{extra_idx:02d}.png'
        plt.savefig(extra_frame_filename, bbox_inches='tight', pad_inches=0.1)
        frame_files.append(extra_frame_filename)


    # Create GIF
    images = [imageio.imread(frame) for frame in frame_files]
    imageio.mimsave(os.path.join(cwd, gif_filename), images, fps=16, loop=0)  # Adjust fps as needed
    gif_filename='animation_NO_LOOP.gif'
    imageio.mimsave(os.path.join(cwd, gif_filename), images, fps=16)  # Adjust fps as needed

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


plotData = []

for incr,u in enumerate(UU):
    # SED.append(model.bodies[0].get_nodal_SED(u))
    # # EPCUM.append(model.bodies[0].get_nodal_EPCUM(epcum_gauss[incr]))


    if 'elastic' in cwd:
        plotData.append(model.bodies[0].get_nodal_SED(u))
        plastic = False
    elif 'plastic' in cwd:
        epcum_gauss = np.loadtxt(cwd+'/EPcum.csv', delimiter=',').reshape(len(UU),-1,8)
        plotData.append(model.bodies[0].get_nodal_EPCUM(epcum_gauss[incr]))
        plastic = True
    else:
        plotData.append(model.bodies[0].get_nodal_SED(u))
        plastic = False

        # raise ValueError("Unknown model subname. Please check the model's subname.")


plotData = [np.maximum(0.0, data) for data in plotData]


# Assuming your model and data are already loaded:
generate_gif(model, TIMES, UU, plotData, plastic = plastic)
