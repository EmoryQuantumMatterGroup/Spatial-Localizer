import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys
import os

sys.path.insert(0,"./../../")

import toolkit_local.matrices as mm
import toolkit_local.plotting as pp
import toolkit_local.hdf5 as hdf






if len(sys.argv) !=2 :
    k_res=8

else : 
    k_res=int(sys.argv[1])


dk = 2*np.pi/k_res


ms = [1.0,3.0]
L = k_res
    

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

X_QWZ = hdf.load_hdf5_to_numpy(fname,"X")
Y_QWZ = hdf.load_hdf5_to_numpy(fname,"Y")

four_t_inv = np.kron(mm.ft_2D(L,L),np.eye(2)).conj().T


v1_wf = np.roll((four_t_inv @ U @  mm.normalize(v1/np.abs(v1))).reshape(L,L,2),(L//2,L//2,),(0,1)).flatten()
v1_oc = np.roll((four_t_inv @ U @  mm.normalize(v1)            ).reshape(L,L,2),(L//2,L//2,),(0,1)).flatten()
v2_wf = np.roll((four_t_inv @ U2 @ mm.normalize(v2/np.abs(v2))).reshape(L,L,2),(L//2,L//2,),(0,1)).flatten()



fig,axs = pp.create_gridspec_figure((1,2),side_length=500,wspace=0.2,hspace=0.35)


ax_r_QWZ =    axs[0][0]
ax_r_QWZ_t =  axs[0][1]

# ax_r_QSH.set_title("WF and Coherent State QSH")
# ax_r_QWZ.set_title("WF and Coherent State QWZ")
# ax_bc_QWZ.set_title("Berry Connection QWZ")
# ax_pf_QSH.set_title("Pfaffian of TRS Sewing Matrix QSH")

wf_string = r"$|\psi(\mathbf{r})|$"

ylims=[1e-18,1.5]#[1e-5,1]

yticks = [10**(-3*ii) for ii in range(7)]#yticks = [10**(-ii) for ii in range(6)]

if True : # plotting QWZ rvec
    
    rvec_QWZ = v1_wf

    rvec_QWZ_oc = v1_oc
    mean_x, mean_y      = mm.exp_val(X_QWZ, rvec_QWZ).real, mm.exp_val(Y_QWZ,rvec_QWZ).real

    
    mean_x_oc, mean_y_oc = mm.exp_val(X_QWZ, rvec_QWZ_oc).real, mm.exp_val(Y_QWZ,rvec_QWZ_oc).real
    

   
    rvec_QWZ = np.sum(np.abs(rvec_QWZ).reshape(L**2,2),axis=-1)

    rvec_QWZ_oc = np.sum(np.abs(rvec_QWZ_oc).reshape(L**2,2),axis=-1)
    
    r_x = np.abs(X_QWZ.diagonal().real - mean_x)
    r_y = np.abs(Y_QWZ.diagonal().real - mean_y)
    r_x_oc = np.abs(X_QWZ.diagonal().real - mean_x_oc)
    r_y_oc = np.abs(Y_QWZ.diagonal().real - mean_y_oc)
    
    r = np.sqrt(r_x**2 + r_y**2)[::2]
    r_oc = np.sqrt(r_x_oc**2 + r_y_oc**2)[::2]
    
    trim_val = 0.1
    
    

    ax_r_QWZ.scatter(r[r>trim_val],rvec_QWZ[r>trim_val]      ,facecolors='none',edgecolors='blue'       ,label="WF")
    ax_r_QWZ.scatter(r_oc[r_oc>trim_val],rvec_QWZ_oc[r_oc>trim_val],facecolors='none',edgecolors='deepskyblue',label="Coherent State")
    
    temp_fit = np.linspace(1,30,100)
    ax_r_QWZ.plot(temp_fit, 1.7e-1*(temp_fit )**(-2),'--',c='red',zorder=20,label=r'Fit Line $\propto 1/r^2$')
    
    ax_r_QWZ.set_xscale("log")
    ax_r_QWZ.set_yscale("log")
    ax_r_QWZ.set_ylim(*ylims)
    
    
if True : # plotting QWZ rvec
    
    rvec_QWZ_t = v2_wf


    mean_x, mean_y      = mm.exp_val(X_QWZ, rvec_QWZ_t).real, mm.exp_val(Y_QWZ,rvec_QWZ_t).real

    



    
    rvec_QWZ_t = np.sum(np.abs(rvec_QWZ_t).reshape(L**2,2),axis=-1)


    
    r_x = np.abs(X_QWZ.diagonal().real - mean_x)
    r_y = np.abs(Y_QWZ.diagonal().real - mean_y)

    
    r = np.sqrt(r_x**2 + r_y**2)[::2]

    
    trim_val = 0.1
    
    

    ax_r_QWZ_t.scatter(r[r>trim_val],rvec_QWZ_t[r>trim_val],facecolors='none',edgecolors='blue'       ,label="WF")
    
    
    ax_r_QWZ_t.set_xscale("log")
    ax_r_QWZ_t.set_yscale("log")
    ax_r_QWZ_t.set_ylim(*ylims)
    





ax_r_QWZ.set(ylabel=wf_string,xlabel=r'$|\mathbf{r}|$',yticks=yticks,title="m=1.0")
ax_r_QWZ_t.set(xlabel=r'$|\mathbf{r}|$'               ,yticks=[]    ,title="m=3.0")

    
fig.savefig(f"./figs/main_text_figure_4_ab_decays_N={L}.png",bbox_inches="tight")