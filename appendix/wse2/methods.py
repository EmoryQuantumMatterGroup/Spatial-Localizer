import numpy as np
import matplotlib.pyplot as plt
import sys
import os


sys.path.insert(0,"./../../")

from toolkit_local.cond_mat import get_K_vectors
from toolkit_local import matrices as mm
from toolkit_local.localizer_general import Localizer
from toolkit_local import hdf5 as hdf



if True :
    
    def build_C3_operator_triangle_a1_a1minusa2(Nx, Ny, m0=0, n0=0):
        """
        Rotate around the centroid of the triangle formed by:
        (m0, n0), (m0+1, n0), (m0+1, n0-1)
        """
        C3_4D = np.zeros((Nx, Ny, Nx, Ny), dtype=complex)

        # Centroid of the triangle
        m_c = m0 + 2/3
        n_c = n0 - 1/3

        for n in range(Ny):
            for m in range(Nx):
                # Shift relative to centroid
                m_rel = m - m_c
                n_rel = n - n_c

                # Apply CCW C3 rotation
                m_rot_rel = -m_rel - n_rel
                n_rot_rel = m_rel

                # Shift back
                m_rot = m_rot_rel + m_c
                n_rot = n_rot_rel + n_c

                # Round and wrap with PBC
                m_rot_idx = int(np.round(m_rot)) % Nx
                n_rot_idx = int(np.round(n_rot)) % Ny

                C3_4D[m_rot_idx, n_rot_idx, m, n] = 1.0

        return C3_4D.reshape(Nx * Ny, Nx * Ny)

    def build_C3_operator_triangle_a2_a2minusa1(Nx, Ny, m0, n0):
        """
        Rotate around the centroid of the triangle formed by:
        (m0, n0), (m0, n0+1), (m0-1, n0+1)
        """
        C3_4D = np.zeros((Nx, Ny, Nx, Ny), dtype=complex)

        # Centroid of the triangle
        m_c = m0 - 1/3
        n_c = n0 + 2/3

        for n in range(Ny):
            for m in range(Nx):
                # Shift relative to centroid
                m_rel = m - m_c
                n_rel = n - n_c

                # Apply CCW C3 rotation
                m_rot_rel = -m_rel - n_rel
                n_rot_rel = m_rel

                # Shift back
                m_rot = m_rot_rel + m_c
                n_rot = n_rot_rel + n_c

                # Round and wrap with PBC
                m_rot_idx = int(np.round(m_rot)) % Nx
                n_rot_idx = int(np.round(n_rot)) % Ny

                C3_4D[m_rot_idx, n_rot_idx, m, n] = 1.0

        return C3_4D.reshape(Nx * Ny, Nx * Ny)


    def get_lat_vecs(a) : 
        a1 = 0.5*a*np.array([2,0])
        a2 = 0.5*a*np.array([1,np.sqrt(3)])
        
        return a1,a2


    def get_xy_WSe2(L,a1,a2,offs=np.array([[0,0], [0,0], [0,0]]), no_orbs=False) : 
        
        a1_mat = np.diag(np.array(range(0,L),dtype=complex))
        
        a2_mat = np.diag(np.array(range(0,L),dtype=complex))
        
        eye = np.eye(L,dtype=complex)
        
        a1_mat = np.kron(a1_mat,eye)
        
        a2_mat = np.kron(eye,a2_mat)
        
        X = a1[0]*a1_mat + a2[0]*a2_mat
        Y = a1[1]*a1_mat + a2[1]*a2_mat
        
        if no_orbs : 
            return X,Y
        
        x_full = np.zeros((X.shape[0],3,X.shape[1],3),dtype=complex)
        y_full = np.zeros((Y.shape[0],3,X.shape[1],3),dtype=complex)
        
        for ii in range(3) : 
            offs_curr = offs[ii]
            
            x_full[:,ii,:,ii] = X + np.eye(L**2,dtype=complex)*offs_curr[0]
            y_full[:,ii,:,ii] = Y + np.eye(L**2,dtype=complex)*offs_curr[1]
            
        return x_full.reshape((3*L**2,3*L**2)),y_full.reshape((3*L**2,3*L**2))  
        

    def getHam_WSe2(kx,ky, param_mode="og_paper_lda", ham_mode='') : 
        
        
        # parameters 
        match param_mode :
            
            case "newer_paper" :
                
                a = 1 # lattice spacing
                
                t0 = -0.146 # hoppings
                t1 = -0.124
                t2 = 0.507
                t11 = 0.117
                t12 = 0.127
                t22 = 0.015
                
                eps1 = 0.728 # energies
                eps2 = 1.655
                
                alpha = kx*a/2
                beta = ky*np.sqrt(3)*a/2
            
            case "og_paper_gga" :
                
                a = 1 # lattice spacing
                
                t0 = -0.207 # hoppings
                t1 = 0.457
                t2 = 0.486
                t11 = 0.263
                t12 = 0.329
                t22 = 0.034
                
                eps1 = 0.943 # energies
                eps2 = 2.179
                
                alpha = kx*a/2
                beta = ky*np.sqrt(3)*a/2
                
            case "og_paper_lda" :
                
                a = 1 # lattice spacing
                
                t0 = -0.184 # hoppings
                t1 = 0.506
                t2 = 0.514
                t11 = 0.305
                t12 = 0.353
                t22 = 0.025
                
                eps1 = 1.124 # energies
                eps2 = 2.447
                
                alpha = kx*a/2
                beta = ky*np.sqrt(3)*a/2
            
            case _ :
                raise ValueError("why?")
        
        
        if False :
            ham = np.zeros((3,3),dtype=complex) 
            
            
            # h1 term 
            
            ham[0,1] = -2*np.sqrt(3)*t2*np.sin(alpha)*np.sin(beta) + 2j*t1*(np.sin(alpha)*np.cos(beta))
            
            ham[0,2] = 2*t2*(- np.cos(alpha)*np.cos(beta)) + 2*np.sqrt(3)*1j*t1*np.cos(alpha)*np.sin(beta)
            
            ham[1,2] = np.sqrt(3)*(t22 - t11)*np.sin(alpha)*np.sin(beta) + 4j*t12*np.sin(alpha)*(np.cos(alpha) - np.cos(beta))
            
            ham += np.conj(ham.T)
            
            
            ham[0,0] = 2*t0*(2*np.cos(alpha)*np.cos(beta)) + eps1 
            
            ham[1,1] = (t11 + 3*t22)*np.cos(alpha)*np.cos(beta) + eps2
            
            ham[2,2] = (3*t11 + t22)*np.cos(alpha)*np.cos(beta) + eps2 
            
        elif ham_mode == 'TNN' : 
            
            
            r0 = 0.036
            r1 = -0.234
            r2 = 0.107
            r11 = 0.044 
            r12 = 0.075
            u0 = -0.061
            u1 = 0.032
            u2 = 0.007
            u11 = 0.329
            u12 = -0.202
            u22 = -0.164
            
            
            ham = np.zeros((3,3),dtype=complex) 
            
            
            # h1 term 
            
            # real part
            h1 = -2*np.sqrt(3)*t2*np.sin(alpha)*np.sin(beta) + 2*(r1 + r2)*np.sin(3*alpha)*np.sin(beta) - 2*np.sqrt(3)*u2*np.sin(2*alpha)*np.sin(2*beta)
            
            # imag part
            h1 += 1j*( 2*t1*np.sin(alpha)*(2*np.cos(alpha) + np.cos(beta)) + 2*(r1 - r2)*np.sin(3*alpha)*np.cos(beta) + 2*u1*np.sin(2*alpha)*(2*np.cos(2*alpha) + np.cos(2*beta))  )
            
            
            # real part
            h2 = 2*t2*(np.cos(2*alpha) - np.cos(alpha)*np.cos(beta)) - (2/np.sqrt(3))*(r1 + r2)*(np.cos(3*alpha)*np.cos(beta) - np.cos(2*beta)) + 2*u2*(np.cos(4*alpha) - np.cos(2*alpha)*np.cos(2*beta))
            
            h2 += 1j* ( 2*np.sqrt(3)*t1*np.cos(alpha)*np.sin(beta) + (2/np.sqrt(3))*np.sin(beta)*(r1-r2)*(np.cos(3*alpha) + 2*np.cos(beta)) + 2*np.sqrt(3)*u1*np.cos(2*alpha)*np.sin(2*beta))
            
            # real part
            h12 = np.sqrt(3)*(t22 - t11)*np.sin(alpha)*np.sin(beta) + 4*r12*np.sin(3*alpha)*np.sin(beta) + np.sqrt(3)*(u22-u11)*np.sin(2*alpha)*np.sin(2*beta)
            
            h12 += 1j* ( 4*t12*np.sin(alpha)*(np.cos(alpha) - np.cos(beta)) + 4*u12*np.sin(2*alpha)*(np.cos(2*alpha) - np.cos(2*beta)))
            
            ham[0,1] = h1 
            
            ham[1,0] = np.conj(h1)
            
            ham[0,2] = h2
            ham[2,0] = np.conj(h2)
            
            ham[1,2] = h12
            ham[2,1] = np.conj(h12)
            
            # ham += np.conj(ham.T)
            
            
            ham[0,0] = 2*t0*(np.cos(2*alpha) + 2*np.cos(alpha)*np.cos(beta)) + eps1 + 2*r0*(2*np.cos(3*alpha)*np.cos(beta) + np.cos(2*beta)) + 2*u0*(2*np.cos(2*alpha)*np.cos(2*beta) + np.cos(4*alpha))
                        
            
            ham[1,1] = 2*t11*np.cos(2*alpha) + (t11 + 3*t22)*np.cos(alpha)*np.cos(beta) + eps2 + 4*r11*np.cos(3*alpha)*np.cos(beta) + 2*(r11 + np.sqrt(3)*r12)*np.cos(2*beta) + (u11 + 3*u22)*np.cos(2*alpha)*np.cos(2*beta) + 2*u11*np.cos(4*alpha)                     
            
            ham[2,2] = 2*t22*np.cos(2*alpha) + (3*t11 + t22)*np.cos(alpha)*np.cos(beta) + eps2 + 2*r11*(2*np.cos(3*alpha)*np.cos(beta) + np.cos(2*beta)) + (2/np.sqrt(3))*r12*(4*np.cos(3*alpha)*np.cos(beta) - np.cos(2*beta)) + (3*u11 + u22)*np.cos(2*alpha)*np.cos(2*beta) + 2*u22*np.cos(4*alpha)                  
            
        else :
            
            ham = np.zeros((3,3),dtype=complex) 
            
            
            # h1 term 
            
            h1 = -2*np.sqrt(3)*t2*np.sin(alpha)*np.sin(beta) + 2j*t1*(np.sin(2*alpha) + np.sin(alpha)*np.cos(beta))
            
            h2 = 2*t2*(np.cos(2*alpha) - np.cos(alpha)*np.cos(beta)) + 2*np.sqrt(3)*1j*t1*np.cos(alpha)*np.sin(beta)
            
            h12 = np.sqrt(3)*(t22 - t11)*np.sin(alpha)*np.sin(beta) + 4j*t12*np.sin(alpha)*(np.cos(alpha) - np.cos(beta))
            
            ham[0,1] = h1 
            
            ham[1,0] = np.conj(h1)
            
            ham[0,2] = h2
            ham[2,0] = np.conj(h2)
            
            ham[1,2] = h12
            ham[2,1] = np.conj(h12)
            
            # ham += np.conj(ham.T)
            
            
            ham[0,0] = 2*t0*(np.cos(2*alpha) + 2*np.cos(alpha)*np.cos(beta)) + eps1 
            
            ham[1,1] = 2*t11*np.cos(2*alpha) + (t11 + 3*t22)*np.cos(alpha)*np.cos(beta) + eps2
            
            ham[2,2] = 2*t22*np.cos(2*alpha) + (3*t11 + t22)*np.cos(alpha)*np.cos(beta) + eps2 
        
        return ham 


    def get_xy_kspace_WSe2(L) : 
        
        hop = np.eye(L,k=-1,dtype=complex)
        hop[0,-1] = 1 
        
        # print(hop.real)
        
        eye = np.eye(L,dtype=complex)
        
        eye_orb = np.eye(3,dtype=complex)
        
        X = np.kron(hop,np.kron(eye,eye_orb))
        Y = np.kron(eye,np.kron(hop,eye_orb))
        
        return X,Y 



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
    
    

def build_components(L) : 
    a1,a2 = get_lat_vecs(1)
    bb1,bb2,_ = get_K_vectors(np.array([*a1,0]),np.array([*a2,0]),np.array([0,0,1]))
    b_1 = bb1[:2]
    b_2 = bb2[:2]

    if True :
        ks_line = np.linspace(0,1,L+1)[:-1] # k discretization with correct spacing

        kxs_pre,kys_pre = np.meshgrid(ks_line,ks_line,indexing='ij')



        k_b1s = np.tensordot(kxs_pre,b_1,axes=0) 
        k_b2s = np.tensordot(kys_pre,b_2,axes=0) 

        ks = k_b1s + k_b2s

        kxs = k_b1s[:,:,0] + k_b2s[:,:,0]
        kys = k_b1s[:,:,1] + k_b2s[:,:,1]
        
        
        X_rspace, Y_rspace = get_xy_WSe2(L,a1,a2,no_orbs=True) 

        rs = np.zeros((*X_rspace.diagonal().shape,2),dtype=complex)
        rs[...,0] = X_rspace.diagonal()
        rs[...,1] = Y_rspace.diagonal()
        
        four_T = build_fourier_operator(ks.reshape(L**2,2),rs)

        four_T = np.kron(four_T,np.eye(3,dtype=complex))
        
        X_rspace = np.kron(X_rspace,np.eye(3))
        Y_rspace = np.kron(Y_rspace,np.eye(3))


        # ks = gamma_center_mesh(b_1,b_2,L,L)

        # kxs, kys = ks[...,0], ks[...,1]
        if False : # plotting checks
            
            plt.scatter(kxs,kys)

            plt.scatter(*b_1,c='g')
            plt.scatter(*b_2,c='r')
            plt.gca().set_aspect('equal')
            plt.scatter(*(K),c='k')
            plt.show()

            # check that kxs[:,0] and kys[:,0] go along b_1
            plt.scatter(kxs[:,0],kys[:,0],c='b')
            plt.gca().set_aspect("equal")

            # and same for b_2 

            # check that kxs[0,:] and kys[0,:] go along b_2
            plt.scatter(kxs[0,:],kys[0,:],c='k')
            plt.gca().set_aspect("equal")
            
            
        hams = np.zeros((L,L,3,3),dtype=complex) 
        param_mode = 'newer_paper'
        ham_mode = 'TNN'



        for ix in range(L) : 
            for iy in range(L) : 
                
                hams[ix,iy] = getHam_WSe2(kxs[ix,iy], kys[ix,iy],param_mode=param_mode, ham_mode=ham_mode)


        ws, vs = np.linalg.eigh(hams)


        num_bands = 1
            


        U = np.zeros((L,L,3,L*L,num_bands), dtype=complex)
        # print(U.shape)

        ## still looking for a way to do this without a for loop. Perhaps np.split combined with sp.linalg.block_diag has the answer
        vec_index = 0
        for ii in range(L) : 
            for jj in range(L) :
                for kk in range(num_bands) : 
                    U[ii,jj,:,vec_index,kk] = vs[ii,jj,:,kk]
                vec_index += 1
                
        U = U.reshape((L*L*3,L*L*num_bands))


        X_b1,X_b2 = get_xy_kspace_WSe2(L)

        X_b3 = X_b1.T @ X_b2.T

        X_U1 = mm.enforce_unitarity(mm.project(X_b1,U))
        X_U2 = mm.enforce_unitarity(mm.project(X_b2,U))
        X_U3 = mm.enforce_unitarity(mm.project(X_b3,U)) #  X_U1 @ X_U2.T #



        # dk=2*np.pi/L

        # points_cart = np.array([(ii+2/3)*a1 + (jj+2/3)*a2 for ii in range(L) for jj in range(L)])

        # points_ion_cart = np.array([(ii)*a1 + (jj)*a2 for ii in range(L) for jj in range(L)])

        # points_lat = np.array([(ii+2/3, jj+2/3) for ii in range(L) for jj in range(L)])
        # points_ion_lat = np.array([(ii, jj) for ii in range(L) for jj in range(L)])
        
        return X_U1, X_U2, X_U3, U, ks, (X_rspace,Y_rspace), four_T


def extract_and_schmidt(L,r_ex,b_1,b_2,a1,a2,dk,X_U1,X_U2,X_U3) :
    
    if True :
        
        r_a1 = np.array([(1/(2*np.pi))*np.dot(r_ex,b_1)])
        r_a2 = np.array([(1/(2*np.pi))*np.dot(r_ex,b_2)])
        
        p_space1_pt = np.exp(1j*(r_a1*dk))   # np.exp(1j*dk1*xmesh) #

        p_space2_pt = np.exp(1j*(r_a2*dk))   # np.exp(1j*dk2*ymesh) #

        p_space3_pt = np.conj(p_space1_pt)*np.conj(p_space2_pt)

        
        # print(rs.shape)
        # print(ks.shape)


        p_spaces = [p_space1_pt[0],p_space2_pt[0],(p_space3_pt[0])] #[np.array([np.exp(-1j*(2*np.pi/L)*1/3)]), np.array([np.exp(1j*(2*np.pi/L)*2/3)]), np.array([np.exp(-1j*(2*np.pi/L)*1/3)])] #
        ops = [X_U1,X_U2,X_U3]

        init_params = {
            "unitary_real_embed_style" : 'separate',
            'enforce_unitarity' : False
        }
        
        # sx, sy, sz = mm.get_pauli("xyz")
        
        Gammas = mm.cliff_generators(3)

        obj = Localizer(ops[:], p_spaces[:],init_param=init_params,clifford_elements=Gammas) #  ,clifford_elements=[sz,sx,sy]
        cdim = len(Gammas[0])
        
        obj.build_locs()
        
        # print(obj.locs.shape)
        ws_loc,vs_loc = np.linalg.eigh(obj.locs[0])
        
    vec_loc = vs_loc[:,len(ws_loc)//2]

    mu = ws_loc[len(ws_loc)//2]

    Us,Ss,Vs = np.linalg.svd(vec_loc.reshape(X_U1.shape[1],cdim),full_matrices=False)



    return (Us, Ss, Vs), p_spaces, mu, 
        


def build_and_save(L,scan_res) : 
    
    dk = 2*np.pi/L
    
    X_U1,X_U2,X_U3, U, ks, (X_rspace,Y_rspace), four_T = build_components(L)
    
    a1,a2 = get_lat_vecs(1)
    bb1,bb2,_ = get_K_vectors(np.array([*a1,0]),np.array([*a2,0]),np.array([0,0,1]))
    b_1 = bb1[:2]
    b_2 = bb2[:2]

    center = (L//2)*(a1 + a2)
    r_WC = center - a1 - a2 + (a1 + a2)*2/3

    xs = np.linspace(-1,1,scan_res) + center[0]
    ys = np.linspace(-1,1,scan_res) + center[1]

    ysmesh,xsmesh = np.meshgrid(ys,xs,indexing="ij")
    
    dist_x, dist_y = r_WC[0] - xsmesh, r_WC[1] - ysmesh 
    
    dist_r = np.sqrt(dist_x**2 + dist_y**2)
    
    min_ind = np.argmin(np.abs(dist_r))
    
    xsmesh += dist_x.flatten()[min_ind]
    ysmesh += dist_y.flatten()[min_ind]
    
    gaps = np.zeros((scan_res,scan_res))
    Us_all = np.zeros((scan_res,scan_res,L**2,8),dtype=complex)
    Ss_all = np.zeros((scan_res,scan_res,8))
    Vs_all = np.zeros((scan_res,scan_res,8,8),dtype=complex)
    p_spaces_all = np.zeros((scan_res,scan_res,3),dtype=complex)
    
    for ix in range(scan_res) : 
        for iy in range(scan_res) : 
            
            
            r_ex = np.array([xsmesh[iy,ix],ysmesh[iy,ix]])
            (Us_all[iy,ix], Ss_all[iy,ix], Vs_all[iy,ix]), p_spaces_all[iy,ix], gaps[iy,ix] = extract_and_schmidt(L,r_ex,b_1,b_2,a1,a2,dk,X_U1,X_U2,X_U3)
            
    (Us_WC, Ss_WC, Vs_WC), p_spaces_WC, gaps_WC = extract_and_schmidt(L,r_WC,b_1,b_2,a1,a2,dk,X_U1,X_U2,X_U3)
            
            
    fname = f"./h5_files/wse2_bound_info_L={L}_scan_res={scan_res}.h5"
    
    hdf.save_numpy_to_hdf5(fname,Us_all,"/Us_all")
    hdf.save_numpy_to_hdf5(fname,Ss_all,"/Ss_all")
    hdf.save_numpy_to_hdf5(fname,Vs_all,"/Vs_all")
    hdf.save_numpy_to_hdf5(fname,Us_WC,"/Us_WC")
    hdf.save_numpy_to_hdf5(fname,Ss_WC,"/Ss_WC")
    hdf.save_numpy_to_hdf5(fname,Vs_WC,"/Vs_WC")
    hdf.save_numpy_to_hdf5(fname,X_U1,"/X_U1")
    hdf.save_numpy_to_hdf5(fname,X_U2,"/X_U2")
    hdf.save_numpy_to_hdf5(fname,X_U3,"/X_U3")
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
           

def calc_bound_elements(L,scan_res) : 
    
    
    fname = f"./h5_files/wse2_bound_info_L={L}_scan_res={scan_res}.h5"
    
    Us_all = hdf.load_hdf5_to_numpy(fname,"/Us_all")
    Us_WC = hdf.load_hdf5_to_numpy(fname,"/Us_WC")
    Ss_all = hdf.load_hdf5_to_numpy(fname,"/Ss_all")
    Vs_all = hdf.load_hdf5_to_numpy(fname,"/Vs_all")
    X_U1 = hdf.load_hdf5_to_numpy(fname,"/X_U1")
    X_U2 = hdf.load_hdf5_to_numpy(fname,"/X_U2")
    X_U3 = hdf.load_hdf5_to_numpy(fname,"/X_U3")
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
        print(f"ix : {ix}")
        for iy in range(scan_res) : 
            
            
            
            mu = gaps[iy,ix]
            p_spaces = p_spaces_all[iy,ix]
            
            Ss = Ss_all[iy,ix]
            Us = Us_all[iy,ix]
            
            p1 = Us[:,0]
            s1 = Ss[0]
            
            
            
            ops = []
            for iop, xmat in enumerate([X_U1,X_U2,X_U3]) : 
                tmp_real,tmp_imag = mm.unt_to_real_imag(xmat*np.conj(p_spaces[iop]))
                ops.append(np.copy(tmp_real - np.eye(len(tmp_real),dtype=complex)))
                ops.append(np.copy(tmp_imag))
    
            cdim=8
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
                r_deviation_all[iy,ix] +=((np.vdot(p1, op @ op @ p1)).real)**2


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
            
            
            
            
    hdf.save_numpy_to_hdf5(fname,comp1s_all,"/comp1s_all")
    hdf.save_numpy_to_hdf5(fname,comp2s_all,"/comp2s_all")
    hdf.save_numpy_to_hdf5(fname,comp3s_all,"/comp3s_all")
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
    hdf.save_numpy_to_hdf5(fname,(variances_wf_occupied - mm.variance(X_P,wf_WC_rspace) - mm.variance(Y_P,wf_WC_rspace)),"/deviations_variance_from_WC_WF")
    hdf.save_numpy_to_hdf5(fname,(variances_occupied - mm.variance(X_P,p1_WC_rspace) - mm.variance(Y_P,p1_WC_rspace)),"/deviations_variance_from_WC_p1")
    
    print(mm.variance(X_P,wf_WC_rspace) + mm.variance(Y_P,wf_WC_rspace))
    
