import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys
import os

sys.path.insert(0,"./../../")

import toolkit_local.plotting as pp
import toolkit_local.hdf5 as hdf
import toolkit_local.cond_mat as cm 



def repeated_quiver_plot(
    X, Y, u, v,
    reps_x=2, reps_y=2,
    arrow_len=1.0,
    ax=None,
    cmap='viridis',
    log_color=True,
    show_colorbar=True,
    colorbar_label='Magnitude',
    vrange=None,
    **kwargs
):
    """
    Repeat a vector field periodically in x and y using uniform-length quiver arrows.
    Now correctly calculates tile spacing based on input coordinates.
    """
    # Get sorted unique x/y coordinates
    x_vals = np.unique(X)
    y_vals = np.unique(Y)

    if len(x_vals) > 1:
        x_period = x_vals[1] - x_vals[0]
    else:
        x_period = 1.0
    x_period *= len(x_vals)

    if len(y_vals) > 1:
        y_period = y_vals[1] - y_vals[0]
    else:
        y_period = 1.0
    y_period *= len(y_vals)

    X_tiled, Y_tiled, u_tiled, v_tiled = [], [], [], []

    for i in range(reps_x):
        for j in range(reps_y):
            dx = i * x_period
            dy = j * y_period
            X_tiled.append(X + dx)
            Y_tiled.append(Y + dy)
            u_tiled.append(u)
            v_tiled.append(v)

    X_big = np.concatenate(X_tiled)
    Y_big = np.concatenate(Y_tiled)
    u_big = np.concatenate(u_tiled)
    v_big = np.concatenate(v_tiled)

    return colored_magnitude_quiver(
        X_big, Y_big, u_big, v_big,
        arrow_len=arrow_len,
        ax=ax,
        cmap=cmap,
        log_color=log_color,
        show_colorbar=show_colorbar,
        colorbar_label=colorbar_label,
        vrange=vrange,
        **kwargs
    )

def colored_magnitude_quiver(
    X, Y, u, v,
    arrow_len=1.0,
    ax=None,
    cmap='viridis',
    log_color=False,
    show_colorbar=True,
    colorbar_label='Magnitude',
    cmin=0.0001,
    vrange=None,
    **kwargs
):
    """
    Quiver plot with uniform arrow lengths and color indicating original magnitude.

    Parameters:
    - X, Y: grid coordinates
    - u, v: vector components
    - arrow_len: length to normalize all arrows to
    - ax: optional matplotlib Axes
    - cmap: matplotlib colormap name
    - log_color: use logarithmic color scale
    - show_colorbar: whether to add a colorbar
    - colorbar_label: label for the colorbar
    - **kwargs: passed to ax.quiver
    """
    if ax is None:
        fig, ax = plt.subplots()

    mag = np.sqrt(u**2 + v**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        u_norm = np.where(mag > 0, u / mag * arrow_len, 0)
        v_norm = np.where(mag > 0, v / mag * arrow_len, 0)

    if vrange == None :
        norm = LogNorm(vmin=0.01, vmax=mag.max()) if log_color else None # mag[mag>0].min()
    else : 
        norm = LogNorm(*vrange) if log_color else None # mag[mag>0].min()

    q = ax.quiver(
        X, Y, u_norm, v_norm, mag,
        cmap=cmap,
        norm=norm,
        scale_units='xy',
        scale=1,
        pivot='mid',
        width=0.003,
        headwidth=10,
        headlength=10,
        headaxislength=5,
        **kwargs
    )
    # ax.set_aspect('equal')

    if show_colorbar:
        cbar = plt.colorbar(q, ax=ax)
        cbar.set_label(colorbar_label)

    return q, ax, 


if len(sys.argv) !=2 :
    k_res=8

else : 
    k_res=int(sys.argv[1])


dk = 2*np.pi/k_res


ms = [1.0,3.0]

    

file_folder = "./h5_files/"
fname = file_folder + f"L={k_res}_m={ms[0]}.h5"
fname2 = file_folder + f"L={k_res}_m={ms[1]}.h5"


fig_folder = f"./figs/"

os.makedirs(fig_folder,exist_ok=True)

kxs, kys = np.linspace(0,1,k_res+1)[:-1], np.linspace(0,1,k_res+1)[:-1]

kx_mesh, ky_mesh = np.meshgrid(kxs,kys,indexing="ij")


U = hdf.load_hdf5_to_numpy(fname,"U")
U2 = hdf.load_hdf5_to_numpy(fname2,"U")

v1 = hdf.load_hdf5_to_numpy(fname,"p1_1_loc")
v2 = hdf.load_hdf5_to_numpy(fname2,"p1_1_loc")

v1 = v1/np.abs(v1)
v2 = v2/np.abs(v2)

bc_x1, bc_y1 = cm.get_berry_connection_of_state_central((U @ v1).reshape(k_res,k_res,2) )
bc_x2, bc_y2 = cm.get_berry_connection_of_state_central((U2 @ v2).reshape(k_res,k_res,2))



arrow_len=dk*2


fig, axs = pp.create_gridspec_figure((1,2),wspace=0.4)
ax1 = axs[0][0]
ax2 = axs[0][1]

skip_val=2


max_bcs = [np.max(np.sqrt(bc_x1[::skip_val,::skip_val]**2 + bc_y1[::skip_val,::skip_val]**2))]

max_bcs_arg = np.argmax(max_bcs)
max_bc = max_bcs[max_bcs_arg]
vrange = (0.01,1)

q1, ax1 = repeated_quiver_plot(2*np.pi*kx_mesh[::skip_val,::skip_val],2*np.pi*ky_mesh[::skip_val,::skip_val],bc_x1[::skip_val,::skip_val]/max_bc,bc_y1[::skip_val,::skip_val]/max_bc
                                  ,arrow_len=arrow_len,ax=ax1
                                  ,show_colorbar=False,cmap="plasma",vrange=vrange,reps_x=1,reps_y=1)
q2, ax2 = repeated_quiver_plot(2*np.pi*kx_mesh[::skip_val,::skip_val],2*np.pi*ky_mesh[::skip_val,::skip_val],bc_x2[::skip_val,::skip_val]/max_bc,bc_y2[::skip_val,::skip_val]/max_bc
                                  ,arrow_len=arrow_len,ax=ax2
                                  ,show_colorbar=False,cmap="plasma",vrange=vrange,reps_x=1,reps_y=1)


ticks_tmp = [2*np.pi,np.pi,0]
ticklabels_tmp = [r"$2\pi$",r"$\pi$",0]


for ax_tmp in [ax1,ax2] : 
    ax_tmp.set_aspect("equal")
    ax_tmp.set(xlabel=r'$k_x$',ylabel=r'$k_y$',xticks=ticks_tmp, xticklabels=ticklabels_tmp,
                                               yticks=ticks_tmp, yticklabels=ticklabels_tmp)


pp.add_cbar(q1,fig,ax1)
pp.add_cbar(q2,fig,ax2)


ax1.set_title(r"$m=$" + f"{ms[0]}") 
ax2.set_title(r"$m=$" + f"{ms[1]}") 

fig.savefig(fig_folder +  f"main_text_figure_4_cd_N={k_res}_berry_connections.png",bbox_inches="tight")
