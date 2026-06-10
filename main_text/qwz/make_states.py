import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import sys
import os

sys.path.insert(0,"./../../")

import toolkit_local.matrices as mm
import toolkit_local.hams as hh
import toolkit_local.localizer_general as lg 
import toolkit_local.hdf5 as hdf



if len(sys.argv) !=3 :
    k_res=8
    m=1
else : 
    k_res=int(sys.argv[1])
    m=float((sys.argv[2]))
    

dk = 2*np.pi/k_res
stop = k_res**2

folder = "./h5_files/"

os.makedirs(folder,exist_ok=True)

fname = folder + f"L={k_res}_m={m}.h5"

xs = np.array([0])
ys = np.array([0])


    

kxs, kys = np.linspace(0,1,k_res+1)[:-1], np.linspace(0,1,k_res+1)[:-1]

kx_mesh, ky_mesh = np.meshgrid(kxs,kys,indexing="ij")

four_T = np.kron(mm.ft_2D(k_res,k_res),np.eye(2))
four_T_inv = four_T.conj().T

def inv_op(N,offset=0,bond_or_site = "site") : 
    mat = np.zeros((N,N)) 
    translate = np.zeros((N,N))
    
    match bond_or_site : 
        
        case "site" :
            for i1 in range(N) : 

                mat[(-i1)% N,(i1) % N] = 1
                translate[(i1 + offset)% N,i1] = 1
                
            return np.conj(translate.T) @ mat @ translate
        
        case "bond" :
            for i1 in range(N) : 

                mat[(-1-i1)% N,(i1) % N] = 1
                translate[(i1 + offset)% N,i1] = 1
                
            return np.conj(translate.T) @ mat @ translate
        
        case _ : 
            raise ValueError("Invalid parameter provided")
        
        

# inv_op_ = np.kron(np.kron(inv_op(k_res),inv_op(k_res,)),mm.get_pauli("z"))
# inv_op_2 = np.kron(np.kron(inv_op(k_res,bond_or_site="bond",offset=-1),inv_op(k_res,bond_or_site="bond",offset=-1)),mm.get_pauli("z"))
# inv_op_2 = four_T @ inv_op_2 @ four_T_inv




hams = np.zeros((k_res,k_res,2,2),dtype=complex)



for ii in range(k_res) : 
    for jj in range(k_res) : 
        
        hams[ii,jj] = hh.getHam_QWZ_bloch(2*np.pi*kx_mesh[ii,jj],2*np.pi*ky_mesh[ii,jj],m=m) 
        
        

ws,vs = np.linalg.eigh(hams)




# print(np.min(ws[:,:,1] - ws[:,:,0]))

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

obj_loc = lg.Localizer([X_U,Y_U],[psx,psy])

obj_loc.build_locs()

locs = obj_loc.locs


ws_loc,vs_loc = np.linalg.eigh(locs)


mid_ind = len(ws_loc[0])//2

print(ws_loc[:,mid_ind-2:mid_ind+2])
print(ws_loc[:,mid_ind])

vec = vs_loc[:,:,mid_ind].reshape(1,len(X_U),4)

Us, Ss, Vs = np.linalg.svd(vec,full_matrices=False)

print("Ss :", Ss)

v1 = Us[0,:,0]

v1_rspace = four_T_inv @ U @ v1

hdf.save_numpy_to_hdf5(fname,X,"X_R")
hdf.save_numpy_to_hdf5(fname,X_rspace,"X")
hdf.save_numpy_to_hdf5(fname,Y,"Y_R")
hdf.save_numpy_to_hdf5(fname,Y_rspace,"Y")
hdf.save_numpy_to_hdf5(fname,xs,"xs_extract")
hdf.save_numpy_to_hdf5(fname,ys,"ys_extract")

hdf.save_numpy_to_hdf5(fname,vs_loc,"vs_loc")
hdf.save_numpy_to_hdf5(fname,U,"U")
hdf.save_numpy_to_hdf5(fname,v1,"p1_1_loc")


hdf.save_numpy_to_hdf5(fname,v1_rspace,"v1_rspace")

hdf.save_numpy_to_hdf5(fname,Us,"Us")
hdf.save_numpy_to_hdf5(fname,Ss,"Ss")





