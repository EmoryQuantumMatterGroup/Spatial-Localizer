"""Module for various plotting methods"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .hams import *
from .matrices import *



def create_gridspec_figure(grid_shape: tuple, side_length=500, wspace = 0.2, hspace = 0.2):
    """
    Creates a matplotlib figure with axes arranged according to grid_shape,
    ensuring each subplot is a square of the given side_length.

    Parameters:
    grid_shape (tuple): Shape of the grid (rows, cols).
    side_length (int): Length of each subplot square.

    Returns:
    fig, axes: Matplotlib figure and list of axes.
    """
    rows, cols = grid_shape
    fig = plt.figure(figsize=(cols * side_length / 100, rows * side_length / 100))  # Convert to inches

    spec = gridspec.GridSpec(rows, cols, figure=fig, wspace=wspace, hspace=hspace)

    axes = []
    for i in range(rows):
        row_axes = []
        for j in range(cols):
            ax = fig.add_subplot(spec[i, j])
            # ax.set_xticks([])  # Hide ticks for clarity
            # ax.set_yticks([])
            # ax.set_xlim(0, 1)
            # ax.set_ylim(0, 1)
            row_axes.append(ax)
        axes.append(row_axes)

    return fig, axes

def add_cbar(im,fig,ax,left=0.005,bottom=-0.001,width=0.02,height=0,title=None,ticks=None, tick_labels=None, title_pad=4.0) : 
    
    cax = fig.add_axes([ax.get_position().x1+left, ax.get_position().y0 + bottom, width, ax.get_position().height + height ])
    
    
    
    if ticks=="phases" : 
        cbar = fig.colorbar(im, cax=cax, ticks=[np.pi,-np.pi,0])
        cbar.ax.set_yticklabels([r"$\pi$", r"$-\pi$",0])
        
    elif ticks is not None:
        cbar = fig.colorbar(im, cax=cax, ticks=ticks)

        if tick_labels is not None:
            print("here")
            cbar.ax.set_yticklabels((tick_labels))
        # cbar.update_ticks()  # refresh tick artists
        
    else : 
        # cbar.set_ticks(ticks=[np.pi,-np.pi,0],labels=[r"$\pi$", r"$-\pi$",0])
        cbar = fig.colorbar(im, cax=cax,)
    
    if type(title) != type(None) : 
        cbar.set_label(title, rotation=270,labelpad=title_pad)
        
    return cbar
        





