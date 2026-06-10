import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os
from matplotlib.ticker import MultipleLocator

sys.path.insert(0,"./../../")

from toolkit_local import plotting as pp
from toolkit_local import hdf5 as hdf



L = int(sys.argv[1])
scan_res=int(sys.argv[2])

fname = f"./h5_files/wse2_bound_info_L={L}_scan_res={scan_res}.h5"

deviations_from_WC_WF = hdf.load_hdf5_to_numpy(fname,"/deviations_from_WC_WF")
variances_wf = hdf.load_hdf5_to_numpy(fname,"/variances_wf")

fig, axs = pp.create_gridspec_figure((1,2),wspace=0.4)

for ax in fig.axes : 
    ax.set(xticks=[],yticks=[])


ax_p1_mean_dev_wc_p1, ax_wf_mean_dev_wc_wf, = axs[0]


extent=2*list(0.89*np.array([-1.,1.]))


im_comms = ax_p1_mean_dev_wc_p1.imshow(np.sqrt(deviations_from_WC_WF),cmap="plasma_r",origin="lower",norm=LogNorm(vmin=1e-8),extent=extent)
pp.add_cbar(im_comms,fig,ax_p1_mean_dev_wc_p1)
ax_p1_mean_dev_wc_p1.set(title=r"WC Deviations")


im_p1_wc_deviations = ax_wf_mean_dev_wc_wf.imshow((np.sqrt(variances_wf)),cmap='plasma_r',origin="lower",norm=LogNorm(),extent=extent)
pp.add_cbar(im_p1_wc_deviations,fig,ax_wf_mean_dev_wc_wf)
ax_wf_mean_dev_wc_wf.set(title=r"Variance Deviations")



ax_wf_mean_dev_wc_wf.set(xlabel=r"$x$")
ax_p1_mean_dev_wc_p1.set(xlabel=r"$x$",ylabel=r"$y$")

fig.text(0.12,0.84,"(a)")
fig.text(0.575,0.84,"(b)")


if True :
    start_pt = np.array([0,0.5])
        
    theta = np.pi/3

    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])

    points = [np.copy(start_pt)]

    for ii in range(5) : 
        start_pt = rot_mat @ start_pt
        points.append(start_pt.copy())
        

    for ii in range(6) : 
        
        if ii == 5 :
            p1 = points[ii]
            p2 = points[0]
            
        else : 
            p1 = points[ii]
            p2 = points[ii+1]
        
        ax_p1_mean_dev_wc_p1.plot([(p1)[0],(p2)[0]],[(p1)[1],(p2)[1]],c='grey')
        ax_wf_mean_dev_wc_wf.plot([(p1)[0],(p2)[0]],[(p1)[1],(p2)[1]],c='grey')



fig.savefig(f"./figs/appendix_figure_4_wf_deviations_N={L}.png",bbox_inches="tight")
# fig.savefig(f"./figs/appendix_figure_4_wf_deviations_L={L}.pdf",bbox_inches="tight")



