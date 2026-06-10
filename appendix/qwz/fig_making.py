import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LogNorm

sys.path.insert(0,"./../../")

from toolkit_local import plotting as pp
from toolkit_local import hdf5 as hdf

L = int(sys.argv[1])
scan_res=int(sys.argv[2])



def plot_stuff_same_value_no_chern_wf(val_index=0) :
    imshow_kwargs = {"extent" : 2*[-1,1], "origin" : "lower", "cmap" : "plasma_r"}

    # fname_topo = f"./h5_files_new/qwz_bound_info_L={L}_m={m}_scan_res={scan_res}_NEW3_p1_data.h5"
    
    
    values_plotting = []
    
    norms = [{"norm" : LogNorm(1e-3,1)}, {"norm" : LogNorm(1e-3,1)},
              {"norm" : LogNorm(vmax=1e-0)}, {"norm" : LogNorm(1e-2,vmax=1)},]
    
    set_dicts = [{"title": r"OAL, $|p_1(\mathbf{r})\rangle$"},{"title": r"OAL, WF"},{"title": r"Chern, $|p_1(\mathbf{r})\rangle$"},{}]
    
    ax_counter = 0
    for m in [3,1] : 
        for mode in ["","wfs_"] : 
    
            values = [f"/{mode}deviations_from_WC",f"/{mode}deviations_from_extract",f"/{mode}var_both",f"/{mode}coherence"]
            
            fname_tmp = f"./h5_files/qwz_bound_info_L={L}_m={m}_scan_res={scan_res}.h5"
            
            values_plotting.append(hdf.load_hdf5_to_numpy(fname_tmp,values[val_index]))
            
            # set_dicts[ax_counter]["title"] = f"m={m}, state={"wf" if mode=="wfs_" else "p1"}"
            ax_counter += 1
            
    


    fig, axs = pp.create_gridspec_figure((1,3))

    # fig.suptitle(values[val_index][5:])

    for ax in fig.axes : 
        ax.set(xticks=[],yticks=[])


    ax_p1_oal, ax_wf_oal, ax_p1_chern,  = axs[0]



    axes_all = [ax_p1_oal, ax_wf_oal, ax_p1_chern]


        
    for ii, ax_tmp in enumerate(axes_all) : 
        
        im_tmp = ax_tmp.imshow(values_plotting[ii],**imshow_kwargs,**(norms[val_index]))
        
        if ii==2 :
            pp.add_cbar(im_tmp,fig,ax_tmp)
        
        ax_tmp.set(**(set_dicts[ii]))
    



    ax_p1_oal.set(ylabel="$y$",xlabel="$x$")

    ax_p1_chern.set(xlabel="$x$")
    ax_wf_oal.set(xlabel="$x$")



    heights = [0.88,0.46]
    widths = [0.125,0.58]
    height=0.865
    fig.text(0.125,height,"(a)")
    fig.text(0.4,  height,"(b)")
    fig.text(0.67,height,"(c)")


    # fig.text(widths[0],heights[0],"(a)")
    # fig.text(widths[1],heights[0],"(b)")

    # fig.text(widths[0],heights[1],"(c)")
    # fig.text(widths[1],heights[1],"(d)")


    for ax_tmp in [ax_p1_oal, ax_wf_oal, ax_p1_chern] : 
        
        points_tmp = [(-0.5,-0.5),(0.5,-0.5),(0.5,0.5),(-0.5,0.5)]
        
        for ii in range(len(points_tmp)) : 
            
            p1 = points_tmp[ii-1]
            p2 = points_tmp[ii]

            ax_tmp.plot([p1[0],p2[0]],[p1[1],p2[1]],c='grey',alpha=0.6)



    fig.savefig(f"./figs/appendix_figure_{11+val_index}_N={L}_{values[val_index][5:]}.png",bbox_inches="tight")
    # fig.savefig(f"./figs/aaNEW_L={L}_{values[val_index][5:]}.pdf",bbox_inches="tight")    

    
plot_stuff_same_value_no_chern_wf(0)    
plot_stuff_same_value_no_chern_wf(1)
plot_stuff_same_value_no_chern_wf(2)
plot_stuff_same_value_no_chern_wf(3)
    
    
