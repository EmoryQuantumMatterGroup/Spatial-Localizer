import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.ticker import MultipleLocator


sys.path.insert(0,"./../../")

from toolkit_local import plotting as pp
# from toolkit.cond_mat import get_K_vectors
# from toolkit import matrices as mm
# from toolkit import hams as hh
# from toolkit.localizer_general import Localizer
# from toolkit.video import make_video
from toolkit_local import hdf5 as hdf




# Ls = [16,17,18,19,20,21,22,23,24,25,26,27,28,32,36,40,44,48]
Ls = [16,17,18,19,20,21,22,23,24,25,26,27,28]#,32,36,40]
m=1
# diff_gaps        = np.zeros(len(Ls))
# diff_gaps_scaled = np.zeros(len(Ls))
# diff_coherence   = np.zeros(len(Ls))
# diff_variance    = np.zeros(len(Ls))
# variances = np.zeros((len(Ls),2))
gaps = np.zeros((len(Ls),2))
# coherence = np.zeros((len(Ls),2))
Ss_WC = np.zeros((len(Ls),4))
Ss_all = np.zeros((len(Ls),2,4))
#  = np.zeros((len(Ls),2))

for ii, L in enumerate(Ls) : 
    
    fname = f"./h5_files_scaling/qwz_scaling_info_L={L}_m={1}.h5"
    # diff_gaps       [ii] = hdf.load_hdf5_to_numpy(fname,"/diff_gaps")[0]
    # diff_gaps_scaled[ii] = hdf.load_hdf5_to_numpy(fname,"/diff_gaps_scaled")[0]
    # diff_coherence  [ii] = hdf.load_hdf5_to_numpy(fname,"/diff_coherence")[0]
    # diff_variance   [ii] = hdf.load_hdf5_to_numpy(fname,"/diff_variance")[0]
    # variances[ii] = hdf.load_hdf5_to_numpy(fname,"/var_both")
    gaps[ii] = hdf.load_hdf5_to_numpy(fname,"/gaps")[:2]
    # coherence[ii] = hdf.load_hdf5_to_numpy(fname,"/coherence")
    Ss_WC[ii] = hdf.load_hdf5_to_numpy(fname,"/Ss_WC")
    Ss_all[ii] = hdf.load_hdf5_to_numpy(fname,"/Ss_WC")
    
    

L_arr = np.array(Ls)


Ls_t = Ls #[16,17,18,19,20,21,22,23,24,25,26,27,28,32]


gaps_t = np.zeros((len(Ls_t),2))

Ss_WC_t = np.zeros((len(Ls_t),4))

for ii, L in enumerate(Ls_t) : 
    
    fname = f"./h5_files_scaling/qwz_scaling_info_L={L}_m={3}.h5"
    gaps_t[ii] = hdf.load_hdf5_to_numpy(fname,"/gaps")[:2]
    
    Ss_WC_t[ii] = hdf.load_hdf5_to_numpy(fname,"Ss_WC")
    
    

L_arr_t = np.array(Ls_t)




fig, axs = pp.create_gridspec_figure((3,2),hspace=0.4,wspace=0.4) 

alpha=1.925
slope = -2.0
fit_line =alpha*(L_arr**(slope)) 

alpha2=0.22
slope2 = -2.0
fit_line2 =alpha2*(L_arr**(slope2)) 

ax_gaps1, ax_s11 = axs[0]


ax_s11.plot(Ls,np.abs(1-Ss_WC[:,0]),'o',c='b',label=r"$|1-s_1 |$")
ax_s11.set(yscale="log",xscale="log",title="$|1-s_1|$")
ax_s11.plot(Ls,fit_line2,c='k',label=r"Fit line $\propto N^{-2}$")
ax_s11.legend()

ax_gaps1.plot(Ls,gaps[:,0],'o',c='b',label=r"$\mu(\mathbf{r}^*)$")
ax_gaps1.set(yscale="log",xscale="log",title=r"$\mu(\mathbf{r}^*)$")
ax_gaps1.plot(Ls,fit_line,c='k',label=r"Fit line $\propto N^{-2}$")
ax_gaps1.legend()


alpha=4
slope = -2.0
fit_line =alpha*(L_arr**(slope)) 

alpha2=0.22
slope2 = -2.0
fit_line2 =alpha2*(L_arr**(slope2)) 

ax_gaps2, ax_s12 = axs[1]


ax_s12.plot(Ls,np.abs(1-Ss_WC[:,0]),'o',c='b',label=r"$|1-s_1 |$")
ax_s12.set(yscale="log",xscale="log",title="$|1-s_1|$")
ax_s12.plot(Ls,fit_line2,c='k',label=r"Fit line $\propto N^{-2}$")
ax_s12.legend()


ax_gaps2.plot(Ls,gaps[:,1],'o',c='b',label=r"$\mu(\mathbf{r}_{\text{max}})$")
ax_gaps2.set(yscale="log",xscale="log",title=r"$\mu(\mathbf{r}_{\text{max}})$")
ax_gaps2.plot(Ls,fit_line,c='k',label=r"Fit line $\propto N^{-2}$")
ax_gaps2.legend()



alpha=0.0278
slope = -2.0
fit_line =alpha*(L_arr_t**(slope)) 

alpha2=0.0046
slope2 = -2.0
fit_line2 =alpha2*(L_arr_t**(slope2)) 

ax_gaps_t, ax_s1_t = axs[2]


ax_s1_t.plot(Ls_t,np.abs(1-Ss_WC_t[:,0]),'o',c='b',label=r"$|1-s_1 |$")
ax_s1_t.set(yscale="log",xscale="log",xlabel="N",title="$|1-s_1|$")
ax_s1_t.plot(Ls_t,fit_line2,c='k',label=r"Fit line $\propto N^{-2}$")
ax_s1_t.legend()

ax_gaps_t.plot(Ls_t,gaps_t[:,0],'o',c='b',label=r"$\mu(\mathbf{r}^*)$")
ax_gaps_t.set(yscale="log",xscale="log",xlabel="N",title=r"$\mu(\mathbf{r}^*)$")
ax_gaps_t.plot(Ls_t,fit_line,c='k',label=r"Fit line $\propto N^{-2}$")
ax_gaps_t.legend()


for ax_tmp in [ax_gaps_t, ax_s1_t,ax_gaps2, ax_s12,ax_gaps1, ax_s11] : 
    ax_tmp.set(xticks=Ls,xticklabels=[Ls[ii] if (ii==0) or (ii==len(Ls)-1) else "" for ii in range(len(Ls))])



xs = [0.12,0.57]
ys = [0.8875,0.605,0.3225]

fig.text(xs[0],ys[0],"(a)")
fig.text(xs[0],ys[1],"(c)")
fig.text(xs[0],ys[2],"(e)")
fig.text(xs[1],ys[0],"(b)")
fig.text(xs[1],ys[1],"(d)")
fig.text(xs[1],ys[2],"(f)")




fig.savefig("./figs/appendix_figure_1_scaling_fig.png",bbox_inches="tight")
