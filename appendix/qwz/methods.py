import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.ticker import MultipleLocator

sys.path.insert(0,"./../../")

from toolkit_local import plotting as pp
from toolkit_local.cond_mat import get_K_vectors
from toolkit_local import matrices as mm
from toolkit_local import hams as hh
from toolkit_local.localizer_general import Localizer
from toolkit_local import hdf5 as hdf

def build_fourier_operator(kpoints_flat, rpoints_flat):
    """
    Build the inverse Fourier Transform matrix F.

    Args:
        kpoints_flat : array of shape (Nk_total, 2)
            Flattened k-points (kx, ky).
        rpoints_flat : array of shape (Npoints, 2)
            Flattened r-points (x, y).

    Returns:
        F : array of shape (Npoints, Nk_total)
            Fourier transform matrix.
    """

    Nk_total = kpoints_flat.shape[0]
    Npoints = rpoints_flat.shape[0]

    # Compute k · r for all (r, k) pairs
    k_dot_r = np.dot(rpoints_flat, kpoints_flat.T)  # shape (Npoints, Nk_total)

    # Construct Fourier operator
    F = np.exp(1j * k_dot_r) / np.sqrt(Nk_total)

    return F



def build_components(L,m) : 
    a1,a2 = np.array([1,0]), np.array([0,1])

    b_1 = 2*np.pi*np.array([1,0])
    b_2 = 2*np.pi*np.array([0,1])
    
    orb_dim = 2

    if True :
        ks_line = np.linspace(0,1,L+1)[:-1] # k discretization with correct spacing

        kxs_pre,kys_pre = np.meshgrid(ks_line,ks_line,indexing='ij')



        k_b1s = np.tensordot(kxs_pre,b_1,axes=0) 
        k_b2s = np.tensordot(kys_pre,b_2,axes=0) 

        ks = k_b1s + k_b2s

        kxs = k_b1s[:,:,0] + k_b2s[:,:,0]
        kys = k_b1s[:,:,1] + k_b2s[:,:,1]
        
        
        X_rspace, Y_rspace, _ = hh.getXYH_Chern(L,m,"pbc")

        rs = np.zeros((*X_rspace.diagonal()[::2].shape,2),dtype=complex)
        rs[...,0] = X_rspace.diagonal()[::2]
        rs[...,1] = Y_rspace.diagonal()[::2]
        
        four_T = build_fourier_operator(ks.reshape(L**2,2),rs)

        four_T = np.kron(four_T,np.eye(orb_dim,dtype=complex))
        
        
            
        hams = np.zeros((L,L,orb_dim,orb_dim),dtype=complex) 

        for ix in range(L) : 
            for iy in range(L) : 
                
                hams[ix,iy] = hh.getHam_QWZ_bloch(kxs[ix,iy], kys[ix,iy],m)


        ws, vs = np.linalg.eigh(hams)


        num_bands = 1
            
        U = np.zeros((L,L,orb_dim,L*L,num_bands), dtype=complex)
        # print(U.shape)

        ## still looking for a way to do this without a for loop. Perhaps np.split combined with sp.linalg.block_diag has the answer
        vec_index = 0
        for ii in range(L) : 
            for jj in range(L) :
                for kk in range(num_bands) : 
                    U[ii,jj,:,vec_index,kk] = vs[ii,jj,:,kk]
                vec_index += 1
                
        U = U.reshape((L*L*orb_dim,L*L*num_bands))


        hop = np.eye(L,k=-1,dtype=complex)
        hop[0,-1] = 1 
                
        eye = np.eye(L,dtype=complex)
                
        eye_orb = np.eye(orb_dim,dtype=complex)
                
        X_b1 = np.kron(hop,np.kron(eye,eye_orb))
        X_b2 = np.kron(eye,np.kron(hop,eye_orb))

   

        X_U1 = mm.enforce_unitarity(mm.project(X_b1,U))
        X_U2 = mm.enforce_unitarity(mm.project(X_b2,U))

        
        return X_U1, X_U2, U, ks, (X_rspace,Y_rspace), four_T


def extract_and_schmidt(L,r_ex,b_1,b_2,a1,a2,dk,X_U1,X_U2) :
    
    if True :
        
        r_a1 = np.array([(1/(2*np.pi))*np.dot(r_ex,b_1)])
        r_a2 = np.array([(1/(2*np.pi))*np.dot(r_ex,b_2)])
        cdim=4
        
        p_space1_pt = np.exp(1j*(r_a1*dk))   # np.exp(1j*dk1*xmesh) #

        p_space2_pt = np.exp(1j*(r_a2*dk))   # np.exp(1j*dk2*ymesh) #

        

        
        # print(rs.shape)
        # print(ks.shape)


        p_spaces = [p_space1_pt[0],p_space2_pt[0]] #[np.array([np.exp(-1j*(2*np.pi/L)*1/3)]), np.array([np.exp(1j*(2*np.pi/L)*2/3)]), np.array([np.exp(-1j*(2*np.pi/L)*1/3)])] #
        ops = [X_U1,X_U2]

        init_params = {
            "unitary_real_embed_style" : 'separate',
            'enforce_unitarity' : False
        }
        
        # sx, sy, sz = mm.get_pauli("xyz")


        obj = Localizer(ops[:], p_spaces[:],init_param=init_params) #  ,clifford_elements=[sz,sx,sy]
        
        obj.build_locs()
        
        # print(obj.locs.shape)
        ws_loc,vs_loc = np.linalg.eigh(obj.locs[0])
        
    vec_loc = vs_loc[:,len(ws_loc)//2]

    mu = ws_loc[len(ws_loc)//2]

    Us,Ss,Vs = np.linalg.svd(vec_loc.reshape(X_U1.shape[1],cdim),full_matrices=False)

    return (Us, Ss, Vs), p_spaces, mu, 


def build_and_save(L,scan_res,m) : 
    
    orb_dim=2
    
    dk = 2*np.pi/L
    
    X_U1,X_U2, U, ks, (X_rspace,Y_rspace), four_T = build_components(L,m)
    
    a1,a2 = np.array([1,0]), np.array([0,1])

    b_1 = 2*np.pi*np.array([1,0])
    b_2 = 2*np.pi*np.array([0,1])

    center = (L//2)*(a1 + a2)
    r_WC = center

    xs = np.linspace(-1,1,scan_res) + center[0]
    ys = np.linspace(-1,1,scan_res) + center[1]

    ysmesh,xsmesh = np.meshgrid(ys,xs,indexing="ij")
    
    gaps = np.zeros((scan_res,scan_res))
    Us_all = np.zeros((scan_res,scan_res,L**2,4),dtype=complex)
    Ss_all = np.zeros((scan_res,scan_res,4))
    Vs_all = np.zeros((scan_res,scan_res,4,4),dtype=complex)
    p_spaces_all = np.zeros((scan_res,scan_res,2),dtype=complex)
    
    for ix in range(scan_res) : 
        for iy in range(scan_res) : 
            
            r_ex = np.array([xsmesh[iy,ix],ysmesh[iy,ix]])
            (Us_all[iy,ix], Ss_all[iy,ix], Vs_all[iy,ix]), p_spaces_all[iy,ix], gaps[iy,ix] = extract_and_schmidt(L,r_ex,b_1,b_2,a1,a2,dk,X_U1,X_U2)
            
    (Us_WC, Ss_WC, Vs_WC), p_spaces_WC, gaps_WC = extract_and_schmidt(L,r_WC,b_1,b_2,a1,a2,dk,X_U1,X_U2)
            
            
    fname = f"./h5_files/qwz_bound_info_L={L}_m={m}_scan_res={scan_res}.h5"
    
    hdf.save_numpy_to_hdf5(fname,Us_all,"/Us_all")
    hdf.save_numpy_to_hdf5(fname,Ss_all,"/Ss_all")
    hdf.save_numpy_to_hdf5(fname,Vs_all,"/Vs_all")
    hdf.save_numpy_to_hdf5(fname,Us_WC,"/Us_WC")
    hdf.save_numpy_to_hdf5(fname,Ss_WC,"/Ss_WC")
    hdf.save_numpy_to_hdf5(fname,Vs_WC,"/Vs_WC")
    hdf.save_numpy_to_hdf5(fname,X_U1,"/X_U1")
    hdf.save_numpy_to_hdf5(fname,X_U2,"/X_U2")
    # hdf.save_numpy_to_hdf5(fname,X_U3,"/X_U3")
    hdf.save_numpy_to_hdf5(fname,gaps,"/gaps")    
    hdf.save_numpy_to_hdf5(fname,np.array([gaps_WC]),"/gaps_WC")
    hdf.save_numpy_to_hdf5(fname,xsmesh,"/xsmesh")
    hdf.save_numpy_to_hdf5(fname,ysmesh,"/ysmesh")
    hdf.save_numpy_to_hdf5(fname,r_WC,"/r_WC")
    hdf.save_numpy_to_hdf5(fname,four_T,"/four_T")
    
    hdf.save_numpy_to_hdf5(fname,U,"/U")
    hdf.save_numpy_to_hdf5(fname,X_rspace,"/X_rspace")
    hdf.save_numpy_to_hdf5(fname,Y_rspace,"/Y_rspace")
    hdf.save_numpy_to_hdf5(fname,ks,"/ks")
    
    
    hdf.save_numpy_to_hdf5(fname,p_spaces_all,"/p_spaces_all")
    hdf.save_numpy_to_hdf5(fname,np.array([p_spaces_WC]),"/p_spaces_WC")
        
            
def calc_bound_elements(L,scan_res,m) : 
    
    
    fname = f"./h5_files/qwz_bound_info_L={L}_m={m}_scan_res={scan_res}.h5"
    
    Us_all = hdf.load_hdf5_to_numpy(fname,"/Us_all")
    Us_WC = hdf.load_hdf5_to_numpy(fname,"/Us_WC")
    Ss_all = hdf.load_hdf5_to_numpy(fname,"/Ss_all")
    Vs_all = hdf.load_hdf5_to_numpy(fname,"/Vs_all")
    X_U1 = hdf.load_hdf5_to_numpy(fname,"/X_U1")
    X_U2 = hdf.load_hdf5_to_numpy(fname,"/X_U2")
    # X_U3 = hdf.load_hdf5_to_numpy(fname,"/X_U3")
    gaps = hdf.load_hdf5_to_numpy(fname,"/gaps")
    xsmesh = hdf.load_hdf5_to_numpy(fname,"/xsmesh")
    ysmesh = hdf.load_hdf5_to_numpy(fname,"/ysmesh")
    four_T = hdf.load_hdf5_to_numpy(fname,"/four_T")
    U = hdf.load_hdf5_to_numpy(fname,"/U")
    X_rspace = hdf.load_hdf5_to_numpy(fname,"/X_rspace")
    Y_rspace = hdf.load_hdf5_to_numpy(fname,"/Y_rspace")
    ks = hdf.load_hdf5_to_numpy(fname,"/ks")
    p_spaces_all = hdf.load_hdf5_to_numpy(fname,"/p_spaces_all")
    
    scan_res = gaps.shape[0] 
    
    
    comp1s_all = np.zeros((scan_res,scan_res))
    comp2s_all = np.zeros((scan_res,scan_res))
    comp3s_all = np.zeros((scan_res,scan_res))
    r_deviation_all = np.zeros((scan_res,scan_res))
    comms_upper = np.zeros((scan_res,scan_res))
    variances_rspace = np.zeros((scan_res,scan_res))
    variances_occupied = np.zeros((scan_res,scan_res))
    variances_wf = np.zeros((scan_res,scan_res))
    variances_wf_occupied = np.zeros((scan_res,scan_res))
    coherence_cartesian_wf = np.zeros((scan_res,scan_res))
    coherence_cartesian = np.zeros((scan_res,scan_res))
    deviations_rspace = np.zeros((scan_res,scan_res))
    
    deviations_from_WC_WF = np.zeros((scan_res,scan_res))
    deviations_from_WC_p1 = np.zeros((scan_res,scan_res))
    
    P_rspace = (four_T @ U ) @ (four_T @ U ).conj().T
    X_P = P_rspace @ X_rspace @ P_rspace
    Y_P = P_rspace @ Y_rspace @ P_rspace
    
    p1_WC = Us_WC[:,0]
    wf_WC = mm.normalize(p1_WC/np.abs(p1_WC)).copy()
    
    p1_WC_rspace = four_T @ U @ p1_WC
    wf_WC_rspace = four_T @ U @ wf_WC
    
    p1_WC_meanx, p1_WC_meany = mm.exp_val(X_P,p1_WC_rspace).real, mm.exp_val(Y_P,p1_WC_rspace).real
    wf_WC_meanx, wf_WC_meany = mm.exp_val(X_P,wf_WC_rspace).real, mm.exp_val(Y_P,wf_WC_rspace).real
    
    
    for ix in range(scan_res) : 
        for iy in range(scan_res) : 
            

            mu = gaps[iy,ix]
            p_spaces = p_spaces_all[iy,ix]
            
            Ss = Ss_all[iy,ix]
            Us = Us_all[iy,ix]
            
            p1 = Us[:,0]
            s1 = Ss[0]
            
            
            
            ops = []
            for iop, xmat in enumerate([X_U1,X_U2]) : 
                tmp_real,tmp_imag = mm.unt_to_real_imag(xmat*np.conj(p_spaces[iop]))
                ops.append(np.copy(tmp_real - np.eye(len(tmp_real),dtype=complex)))
                ops.append(np.copy(tmp_imag))
    
            cdim=4
            comms = []
            for ii, op1 in enumerate(ops) : 
                for jj, op2 in enumerate(ops) : 
                    # if ii==jj : 
                    #     pass
                    # else :
                    comms.append(mm.commutator(op1,op2))
                    

            b1 = 0 # absolute expval of commutators

            for comm in comms :
                b1 += np.abs(np.vdot(p1, comm @ p1))
                
            r_deviation_all[iy,ix] = 0

            for op in ops : 
                r_deviation_all[iy,ix] +=((np.vdot(p1, op @ p1)).real)**2


            b2 = 0 # expval of physical operators squared

            for op in ops  :  
                b2 += (np.vdot(p1, op @ op @ p1)).real #- ((np.vdot(p1, op @ p1)).real)**2

            # print(f"temp_b2 : {b2}")


            comms_upper[iy,ix] = mu**2 # mu

            # for op in ops  :  
            #     b3 += -((np.vdot(p1, op @ p1)).real)**2

            for comm in comms :
                for pindex in range(cdim-1) :
                    # print(f"b3 : {b3}")
                    
                    comms_upper[iy,ix] += np.abs(Ss[pindex+1]/s1)*np.abs(np.vdot(p1,comm @ Us[:,pindex+1]))

            # As[jj,ii] = b1
            # middles[jj,ii] = b2 
            # Bs[jj,ii] = b3 


            comp1s_all[iy,ix] = b2 - r_deviation_all[iy,ix] - b1/5

            comp2s_all[iy,ix] = b1 + mu**2 + comms_upper[iy,ix] - b2

            comm_norms = 0
            for comm in comms : 
                comm_norms += np.linalg.norm(comm,ord=2)

            comp3s_all[iy,ix] = cdim*(mu**2 + comm_norms) - b2
            
            wf_p1 = mm.normalize(p1/np.abs(p1))
            wf_rspace = (four_T @ U @ wf_p1) #np.roll((four_T @ U @ wf_p1).reshape(L,L,3),axis=(0,1),shift=(L//2,L//2)).reshape(3*L**2)
            
            p1_rspace = (four_T @ U @ p1) #np.roll((four_T @ U @ p1).reshape(L,L,3),axis=(0,1),shift=(L//2,L//2)).reshape(3*L**2)
            
            # print(p1_rspace.shape,X_rspace.shape)
            variances_rspace[iy,ix] = mm.variance(X_rspace,p1_rspace) + mm.variance(Y_rspace,p1_rspace)
            variances_occupied[iy,ix] = mm.variance(X_P,p1_rspace) + mm.variance(Y_P,p1_rspace)
            
            variances_wf[iy,ix] = mm.variance(X_rspace,wf_rspace) + mm.variance(Y_rspace,wf_rspace)
            variances_wf_occupied[iy,ix] = mm.variance(X_P,wf_rspace) + mm.variance(Y_P,wf_rspace)
            comm_cart = mm.commutator(X_P,Y_P)
            coherence_cartesian_wf[iy,ix] = variances_wf_occupied[iy,ix] - np.abs(np.vdot(wf_rspace, comm_cart @ wf_rspace))
            coherence_cartesian[iy,ix] = variances_occupied[iy,ix] - np.abs(np.vdot(p1_rspace, comm_cart @ p1_rspace))
            deviations_rspace[iy,ix] = (mm.exp_val(X_P,p1_rspace).real - xsmesh[iy,ix])**2 + (mm.exp_val(Y_P,p1_rspace).real - ysmesh[iy,ix])**2
            deviations_from_WC_p1[iy,ix] = (mm.exp_val(X_P,p1_rspace).real - p1_WC_meanx)**2 + (mm.exp_val(Y_P,p1_rspace).real - p1_WC_meany)**2
            deviations_from_WC_WF[iy,ix] = (mm.exp_val(X_P,wf_rspace).real - wf_WC_meanx)**2 + (mm.exp_val(Y_P,wf_rspace).real - wf_WC_meany)**2
            
            
            
            
    hdf.save_numpy_to_hdf5(fname,comp1s_all,"/comp1s_all") # testing lower bound
    hdf.save_numpy_to_hdf5(fname,comp2s_all,"/comp2s_all") # testing upper bound
    hdf.save_numpy_to_hdf5(fname,comp3s_all,"/comp3s_all") # testing loring bound
    hdf.save_numpy_to_hdf5(fname,r_deviation_all ,"/r_deviation_all")
    hdf.save_numpy_to_hdf5(fname,comms_upper     ,"/comms_upper")
    hdf.save_numpy_to_hdf5(fname,variances_rspace,"/variances_rspace")
    hdf.save_numpy_to_hdf5(fname,variances_occupied,"/variances_occupied")
    hdf.save_numpy_to_hdf5(fname,variances_wf,"/variances_wf")
    hdf.save_numpy_to_hdf5(fname,variances_wf_occupied,"/variances_wf_occupied")
    hdf.save_numpy_to_hdf5(fname,coherence_cartesian_wf,"/coherence_cartesian_wf")
    hdf.save_numpy_to_hdf5(fname,coherence_cartesian,"/coherence_cartesian")
    hdf.save_numpy_to_hdf5(fname,deviations_rspace,"/deviations_rspace")
    hdf.save_numpy_to_hdf5(fname,deviations_from_WC_p1,"/deviations_from_WC_p1")
    hdf.save_numpy_to_hdf5(fname,deviations_from_WC_WF,"/deviations_from_WC_WF")
    hdf.save_numpy_to_hdf5(fname,variances_wf_occupied - mm.variance(X_P,wf_WC_rspace) - mm.variance(Y_P,wf_WC_rspace),"/deviations_variance_from_WC_WF")
    
    print(mm.variance(X_P,wf_WC_rspace) + mm.variance(Y_P,wf_WC_rspace))
    

def calc_bound_elements_p1(L,scan_res,m) : 
    
    
    fname = f"./h5_files/qwz_bound_info_L={L}_m={m}_scan_res={scan_res}.h5"
    
    Us_all = hdf.load_hdf5_to_numpy(fname,"/Us_all")
    Us_WC = hdf.load_hdf5_to_numpy(fname,"/Us_WC")
    # X_U3 = hdf.load_hdf5_to_numpy(fname,"/X_U3")
    gaps = hdf.load_hdf5_to_numpy(fname,"/gaps")
    xsmesh = hdf.load_hdf5_to_numpy(fname,"/xsmesh")
    ysmesh = hdf.load_hdf5_to_numpy(fname,"/ysmesh")
    four_T = hdf.load_hdf5_to_numpy(fname,"/four_T")
    U = hdf.load_hdf5_to_numpy(fname,"/U")
    X_rspace = hdf.load_hdf5_to_numpy(fname,"/X_rspace")
    Y_rspace = hdf.load_hdf5_to_numpy(fname,"/Y_rspace")
    
    scan_res = gaps.shape[0] 
    
    

    variances_rspace_p1 = np.zeros((scan_res,scan_res))
    variances_occupied_p1 = np.zeros((scan_res,scan_res))
    variances_wf = np.zeros((scan_res,scan_res))
    variances_wf_occupied = np.zeros((scan_res,scan_res))
    coherence_cartesian_wf = np.zeros((scan_res,scan_res))
    coherence_cartesian_p1 = np.zeros((scan_res,scan_res))
    deviations_rspace_p1 = np.zeros((scan_res,scan_res))
    deviations_rspace_wf = np.zeros((scan_res,scan_res))
    # deviations_rspace_p1_v2 = np.zeros((scan_res,scan_res))
    
    deviations_from_WC_WF = np.zeros((scan_res,scan_res))
    deviations_from_WC_p1 = np.zeros((scan_res,scan_res))
    
    P_rspace = (four_T @ U ) @ (four_T @ U ).conj().T
    X_P = P_rspace @ X_rspace @ P_rspace
    Y_P = P_rspace @ Y_rspace @ P_rspace
    
    p1_WC = Us_WC[:,0]
    wf_WC = mm.normalize(p1_WC/np.abs(p1_WC)).copy()
    
    p1_WC_rspace = four_T @ U @ p1_WC
    wf_WC_rspace = four_T @ U @ wf_WC
    
    p1_WC_meanx, p1_WC_meany = mm.exp_val(X_rspace,p1_WC_rspace).real, mm.exp_val(Y_rspace,p1_WC_rspace).real
    wf_WC_meanx, wf_WC_meany = mm.exp_val(X_rspace,wf_WC_rspace).real, mm.exp_val(Y_rspace,wf_WC_rspace).real
    
    print(f"p1 mean : {p1_WC_meanx,p1_WC_meany}")
    
    variances_occupied_p1 = hdf.load_hdf5_to_numpy(fname,"variances_occupied")
    variances_wf_occupied = hdf.load_hdf5_to_numpy(fname,"variances_wf_occupied")
    
    
    for ix in range(scan_res) : 
        for iy in range(scan_res) :  
            
            
            Us = Us_all[iy,ix]
            
            p1 = Us[:,0]
   

            
            wf_p1 = mm.normalize(p1/np.abs(p1))
            wf_rspace = (four_T @ U @ wf_p1) #np.roll((four_T @ U @ wf_p1).reshape(L,L,3),axis=(0,1),shift=(L//2,L//2)).reshape(3*L**2)
            
            p1_rspace = (four_T @ U @ p1) #np.roll((four_T @ U @ p1).reshape(L,L,3),axis=(0,1),shift=(L//2,L//2)).reshape(3*L**2)
            
            meanx_p1 = mm.exp_val(X_rspace,p1_rspace).real
            meany_p1 = mm.exp_val(Y_rspace,p1_rspace).real
            
            meanx_wf = mm.exp_val(X_rspace,wf_rspace).real
            meany_wf = mm.exp_val(Y_rspace,wf_rspace).real
            
            r_x = xsmesh[iy,ix]
            r_y = ysmesh[iy,ix]
            
            # print(p1_rspace.shape,X_rspace.shape)
            # variances_rspace_p1[iy,ix] = mm.variance(X_rspace,p1_rspace) + mm.variance(Y_rspace,p1_rspace)
            # variances_occupied_p1[iy,ix] = mm.variance(X_P,p1_rspace) + mm.variance(Y_P,p1_rspace)
            
            # variances_wf[iy,ix] = mm.variance(X_rspace,wf_rspace) + mm.variance(Y_rspace,wf_rspace)
            # variances_wf_occupied[iy,ix] = mm.variance(X_P,wf_rspace) + mm.variance(Y_P,wf_rspace)
            comm_cart = mm.commutator(X_P,Y_P)
            coherence_cartesian_wf[iy,ix] = variances_wf_occupied[iy,ix] - np.abs(np.vdot(wf_rspace, comm_cart @ wf_rspace))
            coherence_cartesian_p1[iy,ix] = variances_occupied_p1[iy,ix] - np.abs(np.vdot(p1_rspace, comm_cart @ p1_rspace))
            deviations_rspace_p1[iy,ix] = np.sqrt((meanx_p1 - r_x)**2 + (meany_p1 - r_y)**2)
            # deviations_rspace_p1_v2[iy,ix] = np.sqrt((meanx_p1 - (24 - r_x))**2 + (meany_p1 - (24 - r_y))**2)
            deviations_rspace_wf[iy,ix] = np.sqrt((meanx_p1 - r_x)**2 + (meany_p1 - r_y)**2)
            deviations_from_WC_p1[iy,ix] = np.sqrt((meanx_p1 - p1_WC_meanx)**2 + (meany_p1 - p1_WC_meany)**2)
            deviations_from_WC_WF[iy,ix] = np.sqrt((meanx_wf - wf_WC_meanx)**2 + (meany_wf - wf_WC_meany)**2)
            
            
    fname_new = fname[:-3] + "_NEW.h5"
            

    # hdf.save_numpy_to_hdf5(fname_new,variances_rspace_p1,"/variances_rspace_p1")
    hdf.save_numpy_to_hdf5(fname_new,variances_occupied_p1,"/variances_occupied_p1")
    # hdf.save_numpy_to_hdf5(fname_new,variances_wf,"/variances_wf")
    hdf.save_numpy_to_hdf5(fname_new,variances_wf_occupied,"/variances_wf_occupied")
    hdf.save_numpy_to_hdf5(fname_new,coherence_cartesian_wf,"/coherence_cartesian_wf")
    hdf.save_numpy_to_hdf5(fname_new,coherence_cartesian_p1,"/coherence_cartesian_p1")

    hdf.save_numpy_to_hdf5(fname_new,deviations_rspace_p1,"/deviations_rspace_p1")
    # hdf.save_numpy_to_hdf5(fname_new,deviations_rspace_p1_v2,"/deviations_rspace_p1_v2")
    hdf.save_numpy_to_hdf5(fname_new,deviations_from_WC_p1,"/deviations_from_WC_p1")
    hdf.save_numpy_to_hdf5(fname_new,deviations_from_WC_WF,"/deviations_from_WC_WF")
    hdf.save_numpy_to_hdf5(fname_new,variances_wf_occupied - mm.variance(X_P,wf_WC_rspace) - mm.variance(Y_P,wf_WC_rspace),"/deviations_variance_from_WC_WF")
    
    print(mm.variance(X_P,wf_WC_rspace) + mm.variance(Y_P,wf_WC_rspace))
    

    
def parallel_exp_val(states,op) : 
    
    
    op_state = np.tensordot(states,op,axes=[(-1),(-1)])
    
    state_op_state = np.conj(states) * op_state 
    
    return np.sum(state_op_state,axis=-1)
    
def parallel_mmult(states,op) : 
    
    
    return np.tensordot(states,op,axes=[(-1),(-1)])
    
def parallel_normalization(states) :
    
    state_dim = states.shape[-1]
    
    norms = np.tensordot(np.linalg.norm(states,axis=-1),np.ones(state_dim),axes=0)
    
    return states/norms 
    
    
def calc_bound_elements_p1_parallel_trivial(L,scan_res,m,speed='fast') : 
    
    
    fname = f"./h5_files/qwz_bound_info_L={L}_m={m}_scan_res={scan_res}.h5"
    
    Us_all = hdf.load_hdf5_to_numpy(fname,"/Us_all")
    Us_WC = hdf.load_hdf5_to_numpy(fname,"/Us_WC")
    # X_U3 = hdf.load_hdf5_to_numpy(fname,"/X_U3")
    gaps = hdf.load_hdf5_to_numpy(fname,"/gaps")
    xsmesh = hdf.load_hdf5_to_numpy(fname,"/xsmesh")
    ysmesh = hdf.load_hdf5_to_numpy(fname,"/ysmesh")
    four_T = hdf.load_hdf5_to_numpy(fname,"/four_T")
    U = hdf.load_hdf5_to_numpy(fname,"/U")
    X_rspace = hdf.load_hdf5_to_numpy(fname,"/X_rspace")
    Y_rspace = hdf.load_hdf5_to_numpy(fname,"/Y_rspace")
    
    scan_res = gaps.shape[0] 
    
    

    # variances_rspace_p1 = np.zeros((scan_res,scan_res))
    # variances_occupied_p1 = np.zeros((scan_res,scan_res))
    # variances_wf = np.zeros((scan_res,scan_res))
    # variances_wf_occupied = np.zeros((scan_res,scan_res))
    # coherence_cartesian_wf = np.zeros((scan_res,scan_res))
    # coherence_cartesian_p1 = np.zeros((scan_res,scan_res))
    deviations_rspace_p1 = np.zeros((scan_res,scan_res))
    # deviations_rspace_wf = np.zeros((scan_res,scan_res))
    # deviations_rspace_p1_v2 = np.zeros((scan_res,scan_res))
    
    # deviations_from_WC_WF = np.zeros((scan_res,scan_res))
    # deviations_from_WC_p1 = np.zeros((scan_res,scan_res))
    
    P_rspace = (four_T @ U ) @ (four_T @ U ).conj().T
    
    U_rspace = four_T @ U
    X_P = P_rspace @ X_rspace @ P_rspace
    Y_P = P_rspace @ Y_rspace @ P_rspace
    
    comm_cart = mm.commutator(X_P,Y_P)
    
    print("norm")
    print(np.linalg.norm(comm_cart,ord=2))
    
    p1_WC = Us_WC[:,0]
    # wf_WC = mm.normalize(p1_WC/np.abs(p1_WC)).copy()
    
    p1_WC_rspace = four_T @ U @ p1_WC
    # wf_WC_rspace = four_T @ U @ wf_WC
    
    p1_WC_meanx, p1_WC_meany = mm.exp_val(X_rspace,p1_WC_rspace).real, mm.exp_val(Y_rspace,p1_WC_rspace).real
    # wf_WC_meanx, wf_WC_meany = mm.exp_val(X_rspace,wf_WC_rspace).real, mm.exp_val(Y_rspace,wf_WC_rspace).real
    
    print(f"p1 mean : {p1_WC_meanx,p1_WC_meany}")
    
    # variances_occupied_p1 = hdf.load_hdf5_to_numpy(fname,"variances_occupied")
    # variances_wf_occupied = hdf.load_hdf5_to_numpy(fname,"variances_wf_occupied")
    
    if False : 
        for ix in range(scan_res) : 
            for iy in range(scan_res) :  
                
                
                # Us = Us_all[iy,ix]
                
                p1 = Us_all[iy,ix,:,0]
    

                p1_rspace = (four_T @ U @ p1) #np.roll((four_T @ U @ p1).reshape(L,L,3),axis=(0,1),shift=(L//2,L//2)).reshape(3*L**2)
                
                meanx_p1 = mm.exp_val(X_rspace,p1_rspace).real
                meany_p1 = mm.exp_val(Y_rspace,p1_rspace).real
                
                r_x = xsmesh[iy,ix]
                r_y = ysmesh[iy,ix]
                
                deviations_rspace_p1[iy,ix] = np.sqrt((meanx_p1 - r_x)**2 + (meany_p1 - r_y)**2)
                
                
                
                # print(p1_rspace.shape,X_rspace.shape)
                # variances_rspace_p1[iy,ix] = mm.variance(X_rspace,p1_rspace) + mm.variance(Y_rspace,p1_rspace)
                # variances_occupied_p1[iy,ix] = mm.variance(X_P,p1_rspace) + mm.variance(Y_P,p1_rspace)
                
                # variances_wf[iy,ix] = mm.variance(X_rspace,wf_rspace) + mm.variance(Y_rspace,wf_rspace)
                # variances_wf_occupied[iy,ix] = mm.variance(X_P,wf_rspace) + mm.variance(Y_P,wf_rspace)

                # coherence_cartesian_p1[iy,ix] = variances_occupied_p1[iy,ix] - np.abs(np.vdot(p1_rspace, comm_cart @ p1_rspace))
                
                # deviations_rspace_p1_v2[iy,ix] = np.sqrt((meanx_p1 - (24 - r_x))**2 + (meany_p1 - (24 - r_y))**2)
                # deviations_rspace_wf[iy,ix] = np.sqrt((meanx_p1 - r_x)**2 + (meany_p1 - r_y)**2)
                # deviations_from_WC_p1[iy,ix] = np.sqrt((meanx_p1 - p1_WC_meanx)**2 + (meany_p1 - p1_WC_meany)**2)

            
    
    if True :  
        
        p1s_all = Us_all[:,:,:,0]
      
        wfs_all = p1s_all/np.abs(p1s_all)
        
        wfs_all = parallel_normalization(parallel_mmult(wfs_all,U_rspace))
        
        
        
        
        
        p1s_all = parallel_mmult(p1s_all,U_rspace)
        
        mean_xs_all = parallel_exp_val(p1s_all, X_rspace ).real
        
        mean_ys_all = parallel_exp_val(p1s_all, Y_rspace ).real
        
        
        mean_x2s_all = parallel_exp_val(p1s_all, X_rspace @ X_rspace ).real
        
        mean_y2s_all = parallel_exp_val(p1s_all, Y_rspace @ Y_rspace ).real
        
        mean_x2Ps_all = parallel_exp_val(p1s_all, X_P @ X_P).real
        
        mean_y2Ps_all = parallel_exp_val(p1s_all, Y_P @ Y_P).real
        
        comm_exp_vals = parallel_exp_val(p1s_all, comm_cart)
        
        
        
        var_occ_x = (mean_x2Ps_all - mean_xs_all**2) 
        var_occ_y = (mean_y2Ps_all - mean_ys_all**2)
        
        
        deviations_from_extract = np.sqrt((mean_xs_all - xsmesh)**2 + (mean_ys_all - ysmesh)**2)
        
        deviations_from_WC = np.sqrt((mean_xs_all - p1_WC_meanx)**2 + (mean_ys_all - p1_WC_meany)**2)
        
        var_x = (mean_x2s_all - mean_xs_all**2) 
        var_y = (mean_y2s_all - mean_ys_all**2)
        
        coherence = np.sqrt(var_occ_x + var_occ_y) - np.sqrt(np.abs(comm_exp_vals))
        

        
        wfs_mean_xs_all = parallel_exp_val(wfs_all, X_rspace ).real
        
        wfs_mean_ys_all = parallel_exp_val(wfs_all, Y_rspace ).real
        
        
        wfs_mean_x2s_all = parallel_exp_val(wfs_all, X_rspace @ X_rspace ).real
        
        wfs_mean_y2s_all = parallel_exp_val(wfs_all, Y_rspace @ Y_rspace ).real
        
        wfs_mean_x2Ps_all = parallel_exp_val(wfs_all, X_P @ X_P).real
        
        wfs_mean_y2Ps_all = parallel_exp_val(wfs_all, Y_P @ Y_P).real
        
        wfs_comm_exp_vals = parallel_exp_val(wfs_all, comm_cart)
        
        
        
        wfs_var_occ_x = (wfs_mean_x2Ps_all - wfs_mean_xs_all**2) 
        wfs_var_occ_y = (wfs_mean_y2Ps_all - wfs_mean_ys_all**2)
        
        
        wfs_deviations_from_extract = np.sqrt((wfs_mean_xs_all - xsmesh)**2 + (wfs_mean_ys_all - ysmesh)**2)
        
        wfs_deviations_from_WC = np.sqrt((wfs_mean_xs_all - p1_WC_meanx)**2 + (wfs_mean_ys_all - p1_WC_meany)**2)
        
        wfs_var_x = (wfs_mean_x2s_all - wfs_mean_xs_all**2) 
        wfs_var_y = (wfs_mean_y2s_all - wfs_mean_ys_all**2)
        
        wfs_coherence = np.sqrt(wfs_var_occ_x + wfs_var_occ_y) - np.sqrt(np.abs(wfs_comm_exp_vals))
        
        
        
        
        
        
        
        
        
    
    fname_new = fname#[:-3] + "_NEW2_p1_data.h5"
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_occ_x,"/wfs_var_occ_x")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_occ_y,"/wfs_var_occ_y")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_occ_x + wfs_var_occ_y,"/wfs_var_occ_both")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_occ_x - wfs_var_occ_y,"/wfs_var_occ_diff")
    hdf.save_numpy_to_hdf5(fname_new,(wfs_var_occ_x - wfs_var_occ_y).__abs__(),"/wfs_var_occ_diff_abs")
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_deviations_from_extract,"/wfs_deviations_from_extract")
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_deviations_from_WC,"/wfs_deviations_from_WC")
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_x,"/wfs_var_x")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_y,"/wfs_var_y")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_x + wfs_var_y,"/wfs_var_both")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_x - wfs_var_y,"/wfs_var_diff")
    hdf.save_numpy_to_hdf5(fname_new,(wfs_var_x - wfs_var_y).__abs__(),"/wfs_var_diff_abs")
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_coherence,"/wfs_coherence")
    hdf.save_numpy_to_hdf5(fname_new,wfs_comm_exp_vals,"/wfs_comm_exp_vals")
        
        
        
        
        
        
        
    
    fname_new = fname#[:-3] + "_NEW2_p1_data.h5"
    
    hdf.save_numpy_to_hdf5(fname_new,var_occ_x,"/var_occ_x")
    hdf.save_numpy_to_hdf5(fname_new,var_occ_y,"/var_occ_y")
    hdf.save_numpy_to_hdf5(fname_new,var_occ_x + var_occ_y,"/var_occ_both")
    hdf.save_numpy_to_hdf5(fname_new,var_occ_x - var_occ_y,"/var_occ_diff")
    hdf.save_numpy_to_hdf5(fname_new,(var_occ_x - var_occ_y).__abs__(),"/var_occ_diff_abs")
    
    hdf.save_numpy_to_hdf5(fname_new,deviations_from_extract,"/deviations_from_extract")
    
    hdf.save_numpy_to_hdf5(fname_new,deviations_from_WC,"/deviations_from_WC")
    
    hdf.save_numpy_to_hdf5(fname_new,var_x,"/var_x")
    hdf.save_numpy_to_hdf5(fname_new,var_y,"/var_y")
    hdf.save_numpy_to_hdf5(fname_new,var_x + var_y,"/var_both")
    hdf.save_numpy_to_hdf5(fname_new,var_x - var_y,"/var_diff")
    hdf.save_numpy_to_hdf5(fname_new,(var_x - var_y).__abs__(),"/var_diff_abs")
    
    hdf.save_numpy_to_hdf5(fname_new,coherence,"/coherence")
    hdf.save_numpy_to_hdf5(fname_new,comm_exp_vals,"/comm_exp_vals")
    
    

    # # hdf.save_numpy_to_hdf5(fname_new,variances_rspace_p1,"/variances_rspace_p1")
    # hdf.save_numpy_to_hdf5(fname_new,variances_occupied_p1,"/variances_occupied_p1")
    # # hdf.save_numpy_to_hdf5(fname_new,variances_wf,"/variances_wf")
    # hdf.save_numpy_to_hdf5(fname_new,variances_wf_occupied,"/variances_wf_occupied")
    # hdf.save_numpy_to_hdf5(fname_new,coherence_cartesian_wf,"/coherence_cartesian_wf")
    # hdf.save_numpy_to_hdf5(fname_new,coherence_cartesian_p1,"/coherence_cartesian_p1")

    # hdf.save_numpy_to_hdf5(fname_new,deviations_rspace_p1,"/deviations_rspace_p1")
    # # hdf.save_numpy_to_hdf5(fname_new,deviations_rspace_p1_v2,"/deviations_rspace_p1_v2")
    # hdf.save_numpy_to_hdf5(fname_new,deviations_from_WC_p1,"/deviations_from_WC_p1")
    # hdf.save_numpy_to_hdf5(fname_new,deviations_from_WC_WF,"/deviations_from_WC_WF")
    # hdf.save_numpy_to_hdf5(fname_new,variances_wf_occupied - mm.variance(X_P,wf_WC_rspace) - mm.variance(Y_P,wf_WC_rspace),"/deviations_variance_from_WC_WF")
    
    # print(mm.variance(X_P,wf_WC_rspace) + mm.variance(Y_P,wf_WC_rspace))
    

def build_and_save_scaling(L,m) : 
    
    orb_dim=2
    
    dk = 2*np.pi/L
    
    X_U1,X_U2, U, ks, (X_rspace,Y_rspace), four_T = build_components(L,m)
    
    a1,a2 = np.array([1,0]), np.array([0,1])

    b_1 = 2*np.pi*np.array([1,0])
    b_2 = 2*np.pi*np.array([0,1])

    center = (L//2)*(a1 + a2)
    r_WC = center

    xsmesh = np.array([0,0.5,0.25,0.25,0.12]) + center[0]
    ysmesh = np.array([0,0.5,0.25,0.00,0.-0.43]) + center[1]

    num_points = len(xsmesh)
    
    gaps = np.zeros((num_points))
    Us_all = np.zeros((num_points,L**2,4),dtype=complex)
    Ss_all = np.zeros((num_points,4))
    Vs_all = np.zeros((num_points,4,4),dtype=complex)
    p_spaces_all = np.zeros((num_points,2),dtype=complex)
    
    for ix in range(len(xsmesh)) : 

            
            r_ex = np.array([xsmesh[ix],ysmesh[ix]])
            (Us_all[ix], Ss_all[ix], Vs_all[ix]), p_spaces_all[ix], gaps[ix] = extract_and_schmidt(L,r_ex,b_1,b_2,a1,a2,dk,X_U1,X_U2)
            
    (Us_WC, Ss_WC, Vs_WC), p_spaces_WC, gaps_WC = extract_and_schmidt(L,r_WC,b_1,b_2,a1,a2,dk,X_U1,X_U2)
            
            
    fname = f"./h5_files_scaling/qwz_scaling_info_L={L}_m={m}.h5"
    
    hdf.save_numpy_to_hdf5(fname,Us_all,"/Us_all")
    hdf.save_numpy_to_hdf5(fname,Ss_all,"/Ss_all")
    hdf.save_numpy_to_hdf5(fname,Vs_all,"/Vs_all")
    hdf.save_numpy_to_hdf5(fname,Us_WC,"/Us_WC")
    hdf.save_numpy_to_hdf5(fname,Ss_WC,"/Ss_WC")
    hdf.save_numpy_to_hdf5(fname,Vs_WC,"/Vs_WC")
    hdf.save_numpy_to_hdf5(fname,X_U1,"/X_U1")
    hdf.save_numpy_to_hdf5(fname,X_U2,"/X_U2")
    # hdf.save_numpy_to_hdf5(fname,X_U3,"/X_U3")
    hdf.save_numpy_to_hdf5(fname,gaps,"/gaps")    
    hdf.save_numpy_to_hdf5(fname,np.array([gaps_WC]),"/gaps_WC")
    hdf.save_numpy_to_hdf5(fname,xsmesh,"/xsmesh")
    hdf.save_numpy_to_hdf5(fname,ysmesh,"/ysmesh")
    hdf.save_numpy_to_hdf5(fname,r_WC,"/r_WC")
    hdf.save_numpy_to_hdf5(fname,four_T,"/four_T")
    
    hdf.save_numpy_to_hdf5(fname,U,"/U")
    hdf.save_numpy_to_hdf5(fname,X_rspace,"/X_rspace")
    hdf.save_numpy_to_hdf5(fname,Y_rspace,"/Y_rspace")
    hdf.save_numpy_to_hdf5(fname,ks,"/ks")
    
    
    hdf.save_numpy_to_hdf5(fname,p_spaces_all,"/p_spaces_all")
    hdf.save_numpy_to_hdf5(fname,np.array([p_spaces_WC]),"/p_spaces_WC")