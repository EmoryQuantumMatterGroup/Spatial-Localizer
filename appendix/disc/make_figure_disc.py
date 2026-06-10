import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


import sys


sys.path.insert(0,"./../../")


import toolkit_local.plotting as pp
import toolkit_local.hdf5 as hdf

ham="disc"

scan_res = int(sys.argv[1])


fname_topo = f"./{ham}_h5s/bound_comparisons_topo_scan_res={scan_res}.h5"

As_topo      = hdf.load_hdf5_to_numpy(fname_topo,"/As")
Bs_topo      = hdf.load_hdf5_to_numpy(fname_topo,"/Bs")
S1s_topo     = hdf.load_hdf5_to_numpy(fname_topo,"/S1s")
S2s_topo     = hdf.load_hdf5_to_numpy(fname_topo,"/S2s")
middles_topo = hdf.load_hdf5_to_numpy(fname_topo,"/middles")
comp1s_topo  = hdf.load_hdf5_to_numpy(fname_topo,"/comp1s")
comp2s_topo  = hdf.load_hdf5_to_numpy(fname_topo,"./comp2s")
comp3s_topo  = hdf.load_hdf5_to_numpy(fname_topo,"./comp3s")
mus_topo     = hdf.load_hdf5_to_numpy(fname_topo,"./mus")
extent_topo     = hdf.load_hdf5_to_numpy(fname_topo,"./extent")
dis_uc_topo     = hdf.load_hdf5_to_numpy(fname_topo,"./dis_uc")



fname_triv = f"./{ham}_h5s/bound_comparisons_triv_scan_res={scan_res}.h5"

As      = hdf.load_hdf5_to_numpy(fname_triv,"/As")
Bs      = hdf.load_hdf5_to_numpy(fname_triv,"/Bs")
S1s     = hdf.load_hdf5_to_numpy(fname_triv,"/S1s")
S2s     = hdf.load_hdf5_to_numpy(fname_triv,"/S2s")
middles = hdf.load_hdf5_to_numpy(fname_triv,"/middles")
comp1s  = hdf.load_hdf5_to_numpy(fname_triv,"/comp1s")
comp2s  = hdf.load_hdf5_to_numpy(fname_triv,"./comp2s")
comp3s  = hdf.load_hdf5_to_numpy(fname_triv,"./comp3s")
mus     = hdf.load_hdf5_to_numpy(fname_triv,"./mus")
extent     = hdf.load_hdf5_to_numpy(fname_triv,"./extent")
dis_uc     = hdf.load_hdf5_to_numpy(fname_triv,"./dis_uc")




bound_cmap = "plasma_r"



# if True : # plot without our bound

fig,axs = pp.create_gridspec_figure((3,2),wspace=0.45)

ax1,ax3 = axs[0]
ax5,ax6 = axs[1]
ax2,ax4 = axs[2]


norm_saturated= LogNorm(vmin=1e-18)



im1 = ax1.imshow(comp1s_topo.T.__abs__(),cmap=bound_cmap,norm=LogNorm(vmin=10**(-4),vmax=1),extent=extent_topo)

pp.add_cbar(im1,fig,ax1)


im5 = ax5.imshow(comp2s_topo.T.__abs__(),cmap=bound_cmap,extent=extent_topo,norm=norm_saturated) #np.min([np.min(np.abs(comp2s)),np.min(np.abs(comp2s_topo))])

pp.add_cbar(im5,fig,ax5)


im2 = ax2.imshow(comp3s_topo.T.__abs__(),cmap=bound_cmap,extent=extent_topo,norm=LogNorm(vmax=1)) #np.min([np.min(np.abs(comp3s)),np.min(np.abs(comp3s_topo))])

pp.add_cbar(im2,fig,ax2)

for ii in range(3) : 

    uc_plt = dis_uc_topo*np.exp(1j*ii*2*np.pi/3)*np.exp(1j*-np.pi/2)
    
    ax1.plot(uc_plt.T.real,uc_plt.T.imag,c='grey',alpha=0.7)
    ax2.plot(uc_plt.T.real,uc_plt.T.imag,c='grey',alpha=0.7)
    ax5.plot(uc_plt.T.real,uc_plt.T.imag,c='grey',alpha=0.7)

ax1.set( ylabel=r"$y$",xticklabels=[],yticklabels=[],title=r"$E_1$")
ax2.set(xlabel=r"$x$",ylabel=r"$y$", xticklabels=[],yticklabels=[],title=r"$E_3$")

# ax1.set(title="Eq. 8 Lower Bound")
# ax2.set(title="Loring Bound")

norm_saturated



im3 = ax3.imshow(comp1s.T.__abs__(),cmap=bound_cmap,norm=LogNorm(vmin=10**(-6),vmax=1),extent=extent)

pp.add_cbar(im3,fig,ax3)

im6 = ax6.imshow(comp2s.T.__abs__(),cmap=bound_cmap,extent=extent,norm=norm_saturated)

pp.add_cbar(im6,fig,ax6)

im4 = ax4.imshow(comp3s.T.__abs__(),cmap=bound_cmap,norm=LogNorm(vmax=1,vmin=np.min([np.min(comp3s),np.min(comp3s_topo)])),extent=extent)

pp.add_cbar(im4,fig,ax4)

for ii in range(3) : 

    uc_plt = dis_uc*np.exp(1j*ii*2*np.pi/3)*np.exp(1j*-np.pi/2)
    
    ax3.plot(uc_plt.T.real,uc_plt.T.imag,c='grey',alpha=0.7)
    ax4.plot(uc_plt.T.real,uc_plt.T.imag,c='grey',alpha=0.7)
    ax6.plot(uc_plt.T.real,uc_plt.T.imag,c='grey',alpha=0.7)

ax3.set( xticklabels=[],yticklabels=[],title=r"$E_1$")
ax4.set(xlabel=r"$x$", xticklabels=[],yticklabels=[],title=r"$E_3$")

ax6.set( xticklabels=[],yticklabels=[],title=r"$E_2$",)
ax5.set(ylabel=r"$y$", xticklabels=[],yticklabels=[],title=r"$E_2$")

# ax3.set(title="Eq. 8 Lower Bound")
# ax4.set(title="Loring Bound")

width2 = 0.58

fig.text(0.12,0.881,"(a)")
fig.text(width2,0.881,"(b)")
fig.text(0.12,0.61,"(c)")
fig.text(width2,0.61,"(d)")

fig.text(0.12,0.3375,"(e)")
fig.text(width2,0.3375,"(f)")

fig.savefig(f"./figs/appendix_figure_2_disclination_bounds.png",bbox_inches="tight")


    


