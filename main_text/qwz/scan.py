import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os

sys.path.insert(0,"./../../")

import toolkit_local.matrices as mm
import toolkit_local.plotting as pp
import toolkit_local.hams as hh
import toolkit_local.localizer_general as lg 
import toolkit_local.hdf5 as hdf
import toolkit_local.cond_mat as cm 


scan_res = 51 ### INCREASE FOR HIGHER RESOLUTION

if len(sys.argv) !=3 :
    k_res=4
    m=1
else : 
    k_res=int(sys.argv[1])
    m=float((sys.argv[2]))
    

dk = 2*np.pi/k_res
stop = k_res**2



folder = "./h5_files/"

os.makedirs(folder,exist_ok=True)

fname = folder + f"gaps_L={k_res}_m={m}.h5"




xs_line = np.linspace(-0.5,0.5,scan_res)
xs, ys = np.meshgrid(xs_line,xs_line,indexing="ij")

kxs, kys = np.linspace(0,1,k_res+1)[:-1], np.linspace(0,1,k_res+1)[:-1]

kx_mesh, ky_mesh = np.meshgrid(kxs,kys,indexing="ij")

hams = np.zeros((k_res,k_res,2,2),dtype=complex)


for ii in range(k_res) : 
    for jj in range(k_res) : 
        
        hams[ii,jj] = hh.getHam_QWZ_bloch(2*np.pi*kx_mesh[ii,jj],2*np.pi*ky_mesh[ii,jj],m=m) 
        
        

ws,vs = np.linalg.eigh(hams)

hop = np.eye(k_res,k=-1)
hop[0,-1] = 1 

X_rspace, Y_rspace = hh.getX_2D_Chern(k_res), hh.getY_2D_Chern(k_res)

X = np.kron(np.kron(hop,np.eye(k_res)),np.eye(2))
Y = np.kron(np.kron(np.eye(k_res),hop),np.eye(2))


U = np.zeros((k_res,k_res,2,stop),dtype=complex)

bs_vector = np.zeros((k_res,k_res,2),dtype=complex)



state_index = 0
for ii in range(k_res) : 
    for jj in range(k_res) : 
        
        U[ii,jj,:,state_index] = vs[ii,jj,:,0]
        bs_vector[ii,jj,:] = vs[ii,jj,:,0]
        state_index +=1

U = U.reshape(2*k_res**2,stop)  

if_unt = True
        
X_U = (mm.project(X,U)) #
Y_U = (mm.project(Y,U)) #

if if_unt : 
    X_U = mm.enforce_unitarity(X_U)
    Y_U = mm.enforce_unitarity(Y_U)

psx = np.exp(1j*dk*xs)
psy = np.exp(1j*dk*ys)

gaps = np.zeros((scan_res,scan_res))

for ix in range(scan_res) : 
    for iy in range(scan_res) : 

        obj_loc = lg.Localizer([X_U,Y_U],[np.array([psx[ix,iy]]),np.array([psy[ix,iy]])])
        
        gaps[ix,iy] = obj_loc.scan()[0]





hdf.save_numpy_to_hdf5(fname,X,"X_R")
hdf.save_numpy_to_hdf5(fname,X_rspace,"X")
hdf.save_numpy_to_hdf5(fname,Y,"Y_R")
hdf.save_numpy_to_hdf5(fname,Y_rspace,"Y")
hdf.save_numpy_to_hdf5(fname,xs,"xs_extract")
hdf.save_numpy_to_hdf5(fname,ys,"ys_extract")
hdf.save_numpy_to_hdf5(fname,U,"U")
hdf.save_numpy_to_hdf5(fname,gaps,"gaps")

os.makedirs("./figs/",exist_ok=True)

plt.imshow(gaps.T,cmap="magma_r",extent=2*[-0.5,0.5],norm=LogNorm())
cbar = plt.colorbar()
cbar.set_label(r"$\mu(\mathbf{r})$",rotation=270)
plt.title(f"$m=${m}")
plt.xlabel("x")
plt.ylabel("y")


plt.savefig(f"./figs/main_text_figure_4_ab_insets_LIF_N={k_res}_m={m}.png",bbox_inches="tight")




