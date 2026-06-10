import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


import sys
import os
os.makedirs("./figs/",exist_ok=True)
sys.path.insert(0,"./../../")

import toolkit_local.plotting as pp
import toolkit_local.hdf5 as hdf





if len(sys.argv) == 1 : 

    scan_res=101
else : 

    scan_res = int(sys.argv[1])

fname_topo = f"./h5_files/disc_topo_scan_res={scan_res}.h5"

mus_topo     = hdf.load_hdf5_to_numpy(fname_topo,"./mus")
extent_topo     = hdf.load_hdf5_to_numpy(fname_topo,"./extent")
dis_uc_topo     = hdf.load_hdf5_to_numpy(fname_topo,"./dis_uc")



fname_triv = f"./h5_files/disc_triv_scan_res={scan_res}.h5"


mus     = hdf.load_hdf5_to_numpy(fname_triv,"./mus")
extent     = hdf.load_hdf5_to_numpy(fname_triv,"./extent")
dis_uc     = hdf.load_hdf5_to_numpy(fname_triv,"./dis_uc")

cmap = "magma_r"




fig,axs = pp.create_gridspec_figure((1,2),wspace=0.45)

ax1,ax3 = axs[0]


norm = LogNorm(1e-2,1)



for ii in range(3) : 

    uc_plt = dis_uc_topo*np.exp(1j*ii*2*np.pi/3)*np.exp(1j*-np.pi/2)
    
    ax1.plot(uc_plt.T.real,uc_plt.T.imag,c='grey',alpha=0.7)

for ii in range(3) : 

    uc_plt = dis_uc*np.exp(1j*ii*2*np.pi/3)*np.exp(1j*-np.pi/2)
    
    ax3.plot(uc_plt.T.real,uc_plt.T.imag,c='grey',alpha=0.7)





im1 = ax1.imshow(mus_topo.T,cmap=cmap,norm=norm,extent=extent)
im3 = ax3.imshow(mus.T     ,cmap=cmap,norm=norm,extent=extent)

pp.add_cbar(im1,fig,ax1,title=r"$\mu(\mathbf{r})$",title_pad=15)
pp.add_cbar(im3,fig,ax3,title=r"$\mu(\mathbf{r})$",title_pad=15)



ax3.set(xlabel=r"$x$",ylabel=r"$y$",xticks=[],yticks=[],title=r"Trivial Phase")
ax1.set(xlabel=r"$x$",ylabel=r"$y$",xticks=[],yticks=[],title=r"Topological Phase")





fig.savefig(f"./figs/main_text_figure_2_ab_LIF_both_phases.png",bbox_inches="tight")



    


