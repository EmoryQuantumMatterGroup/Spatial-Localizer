import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.ticker import MultipleLocator

sys.path.insert(0,"./../../")

from toolkit_local import plotting as pp
from toolkit_local.cond_mat import get_K_vectors
from toolkit_local import matrices as mm
from toolkit_local.localizer_general import Localizer
from toolkit_local import hdf5 as hdf





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
            
            
        hams = np.zeros((L,L,3,3),dtype=complex) 
        param_mode = 'newer_paper'
        ham_mode = 'TNN'



        for ix in range(L) : 
            for iy in range(L) : 
                
                hams[ix,iy] = getHam_WSe2(kxs[ix,iy], kys[ix,iy],param_mode=param_mode, ham_mode=ham_mode)


        ws, vs = np.linalg.eigh(hams)


        num_bands = 1
            


        U = np.zeros((L,L,3,L*L,num_bands), dtype=complex)

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
        X_U3 = mm.enforce_unitarity(mm.project(X_b3,U)) 

        
        return X_U1, X_U2, X_U3, U, ks, (X_rspace,Y_rspace), four_T


