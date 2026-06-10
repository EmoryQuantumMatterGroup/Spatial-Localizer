import numpy as np
import matplotlib.pyplot as plt
import sys

import matplotlib.gridspec as gridspec
from scipy.special import genlaguerre
import math

import os 
os.makedirs("./h5_files/",exist_ok=True)

sys.path.insert(0,"./../../")

from toolkit_local import plotting as pp
from toolkit_local.cond_mat import get_K_vectors
from toolkit_local import matrices as mm
from toolkit_local.localizer_general import Localizer



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
    

def getHam_WSe2(kx,ky, param_mode="newer_paper", ham_mode='') : 
    
    
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
        
        case _ :
            raise ValueError("why?")
    
    
    if ham_mode == 'TNN' : 
        
        
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
        
        
   
        h1 = -2*np.sqrt(3)*t2*np.sin(alpha)*np.sin(beta) + 2*(r1 + r2)*np.sin(3*alpha)*np.sin(beta) - 2*np.sqrt(3)*u2*np.sin(2*alpha)*np.sin(2*beta)
        
 
        h1 += 1j*( 2*t1*np.sin(alpha)*(2*np.cos(alpha) + np.cos(beta)) + 2*(r1 - r2)*np.sin(3*alpha)*np.cos(beta) + 2*u1*np.sin(2*alpha)*(2*np.cos(2*alpha) + np.cos(2*beta))  )
        
        
  
        h2 = 2*t2*(np.cos(2*alpha) - np.cos(alpha)*np.cos(beta)) - (2/np.sqrt(3))*(r1 + r2)*(np.cos(3*alpha)*np.cos(beta) - np.cos(2*beta)) + 2*u2*(np.cos(4*alpha) - np.cos(2*alpha)*np.cos(2*beta))
        
        h2 += 1j* ( 2*np.sqrt(3)*t1*np.cos(alpha)*np.sin(beta) + (2/np.sqrt(3))*np.sin(beta)*(r1-r2)*(np.cos(3*alpha) + 2*np.cos(beta)) + 2*np.sqrt(3)*u1*np.cos(2*alpha)*np.sin(2*beta))
        
  
        h12 = np.sqrt(3)*(t22 - t11)*np.sin(alpha)*np.sin(beta) + 4*r12*np.sin(3*alpha)*np.sin(beta) + np.sqrt(3)*(u22-u11)*np.sin(2*alpha)*np.sin(2*beta)
        
        h12 += 1j* ( 4*t12*np.sin(alpha)*(np.cos(alpha) - np.cos(beta)) + 4*u12*np.sin(2*alpha)*(np.cos(2*alpha) - np.cos(2*beta)))
        
        ham[0,1] = h1 
        
        ham[1,0] = np.conj(h1)
        
        ham[0,2] = h2
        ham[2,0] = np.conj(h2)
        
        ham[1,2] = h12
        ham[2,1] = np.conj(h12)
        

        
        
        ham[0,0] = 2*t0*(np.cos(2*alpha) + 2*np.cos(alpha)*np.cos(beta)) + eps1 + 2*r0*(2*np.cos(3*alpha)*np.cos(beta) + np.cos(2*beta)) + 2*u0*(2*np.cos(2*alpha)*np.cos(2*beta) + np.cos(4*alpha))
                    
        
        ham[1,1] = 2*t11*np.cos(2*alpha) + (t11 + 3*t22)*np.cos(alpha)*np.cos(beta) + eps2 + 4*r11*np.cos(3*alpha)*np.cos(beta) + 2*(r11 + np.sqrt(3)*r12)*np.cos(2*beta) + (u11 + 3*u22)*np.cos(2*alpha)*np.cos(2*beta) + 2*u11*np.cos(4*alpha)                     
        
        ham[2,2] = 2*t22*np.cos(2*alpha) + (3*t11 + t22)*np.cos(alpha)*np.cos(beta) + eps2 + 2*r11*(2*np.cos(3*alpha)*np.cos(beta) + np.cos(2*beta)) + (2/np.sqrt(3))*r12*(4*np.cos(3*alpha)*np.cos(beta) - np.cos(2*beta)) + (3*u11 + u22)*np.cos(2*alpha)*np.cos(2*beta) + 2*u22*np.cos(4*alpha)                  
        
    else :
        
        ham = np.zeros((3,3),dtype=complex) 
        

        
        h1 = -2*np.sqrt(3)*t2*np.sin(alpha)*np.sin(beta) + 2j*t1*(np.sin(2*alpha) + np.sin(alpha)*np.cos(beta))
        
        h2 = 2*t2*(np.cos(2*alpha) - np.cos(alpha)*np.cos(beta)) + 2*np.sqrt(3)*1j*t1*np.cos(alpha)*np.sin(beta)
        
        h12 = np.sqrt(3)*(t22 - t11)*np.sin(alpha)*np.sin(beta) + 4j*t12*np.sin(alpha)*(np.cos(alpha) - np.cos(beta))
        
        ham[0,1] = h1 
        
        ham[1,0] = np.conj(h1)
        
        ham[0,2] = h2
        ham[2,0] = np.conj(h2)
        
        ham[1,2] = h12
        ham[2,1] = np.conj(h12)

        
        
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


# Normalized hydrogenic radial function R_{n,l}(r)
def hydrogenic_radial(r, n=5, l=2, Z=1.0, a0=1.0):
    """
    Normalized radial part for hydrogenic orbitals:
      R_{n,l}(r) = N_{n,l} * rho^l * e^{-rho/2} * L_{n-l-1}^{2l+1}(rho),
    where rho = 2Zr/(n a0),
    and N_{n,l} = sqrt((2Z/(n a0))^3 * (n-l-1)! / (2n * (n+l)!)).
    """
    rho = 2 * Z * r / (n * a0)
    L   = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    N_nl = np.sqrt((2*Z/(n*a0))**3 * math.factorial(n - l - 1) / (2*n * math.factorial(n + l)))
    return N_nl * (rho**l) * np.exp(-rho / 2) * L

# Real d-orbital angular parts Y_l^m(θ,φ) 
def psi_dz2(r, theta, phi, n=5, Z=1.0, a0=1.0):
    """Normalized 5d_{z^2} orbital"""
    R = hydrogenic_radial(r, n=n, l=2, Z=Z, a0=a0)
    Y = np.sqrt(5/(16*np.pi)) * (3*np.cos(theta)**2 - 1)
    return R * Y

def psi_dxy(r, theta, phi, n=5, Z=1.0, a0=1.0):
    """Normalized 5d_{xy} orbital"""
    R = hydrogenic_radial(r, n=n, l=2, Z=Z, a0=a0)
    Y = np.sqrt(15/(4*np.pi)) * np.sin(theta)**2 * np.sin(2*phi)
    return R * Y

def psi_dx2y2(r, theta, phi, n=5, Z=1.0, a0=1.0):
    """Normalized 5d_{x^2-y^2} orbital"""
    R = hydrogenic_radial(r, n=n, l=2, Z=Z, a0=a0)
    Y = np.sqrt(15/(4*np.pi)) * np.sin(theta)**2 * np.cos(2*phi)
    return R * Y

# Orbital by index 
def psi_orbital(r, theta, phi, idx, Z_eff, a0):
    """
    Selects the real d-orbital wavefunction:
      idx 0 -> d_z^2
      idx 1 -> d_xy
      idx 2 -> d_{x^2-y^2}
    """
    if idx == 0:
        return psi_dz2(r, theta, phi, Z=Z_eff, a0=a0)
    elif idx == 1:
        return psi_dxy(r, theta, phi, Z=Z_eff, a0=a0)
    elif idx == 2:
        return psi_dx2y2(r, theta, phi, Z=Z_eff, a0=a0)
    else:
        raise ValueError("idx must be 0, 1, or 2")


def make_triangular_positions_origin(Nx, Ny, a):
    """
    Build an Nx x Ny triangular lattice starting at (0,0):
      a1 = (1,0)*a, a2 = (0.5,sqrt3/2)*a.
    """
    a1 = np.array([1.0, 0.0]) * a
    a2 = np.array([0.5, np.sqrt(3)/2]) * a
    pos = np.array([i*a1 + j*a2 for i in range(Nx) for j in range(Ny)])
    return pos.reshape(Nx, Ny, 2)


def psi_superposed_multi(X, Y, Z, coeffs, positions, Z_eff, a0):
    """
    Build total wavefunction
    """
    Nx, Ny, Norb = coeffs.shape
    psi = np.zeros_like(X, dtype=np.complex128)
    for i in range(Nx):
        for j in range(Ny):
            x0, y0 = positions[i, j]
            dx, dy = X - x0, Y - y0
            r       = np.sqrt(dx*dx + dy*dy + Z*Z)
            theta   = np.arccos(np.divide(Z, r, out=np.zeros_like(r), where=r!=0))
            phi     = np.arctan2(dy, dx)
            for k in range(Norb):
                psi += coeffs[i, j, k] * psi_orbital(r, theta, phi, k, Z_eff, a0)
    return psi


def plot_heatmap_state_multi_origin(coeffs, a, Z_eff, a0,
                                    padding=1.0, grid_N=100,
                                    center_on_mean=False, padding_mean=2.0, ax=None, ifScatter=True, scatter_size=50, xShift=-0.015,center=(0,0)):

    Nx, Ny, _ = coeffs.shape
    positions = make_triangular_positions_origin(Nx, Ny, a)
    max_x, max_y = positions[:,:,0].max(), positions[:,:,1].max()
    x_min_full, x_max_full = -padding*a, max_x + padding*a
    y_min_full, y_max_full = -padding*a, max_y + padding*a

    # determine mesh limits
    if center_on_mean:
        xs_full = np.linspace(x_min_full, x_max_full, grid_N)
        ys_full = np.linspace(y_min_full, y_max_full, grid_N)
        Xf, Yf = np.meshgrid(xs_full, ys_full)
        psi_full = psi_superposed_multi(Xf, Yf, np.zeros_like(Xf),
                                        coeffs, positions, Z_eff, a0)
        dx_full = xs_full[1] - xs_full[0]
        dy_full = ys_full[1] - ys_full[0]
        P = np.abs(psi_full)**2
        total = np.sum(P) * dx_full * dy_full
        x_mean = np.sum(Xf * P) * dx_full * dy_full / total
        y_mean = np.sum(Yf * P) * dx_full * dy_full / total
        x_min, x_max = x_mean - padding_mean*a, x_mean + padding_mean*a
        y_min, y_max = y_mean - padding_mean*a, y_mean + padding_mean*a
    else:
        x_mean, y_mean = center
        x_min, x_max = x_mean - padding_mean*a, x_mean + padding_mean*a
        y_min, y_max = y_mean - padding_mean*a, y_mean + padding_mean*a

    # mesh over chosen region
    xs = np.linspace(x_min, x_max, grid_N)
    ys = np.linspace(y_min, y_max, grid_N)
    X, Y = np.meshgrid(xs, ys)
    psi2d = psi_superposed_multi(X, Y, np.zeros_like(X),
                                 coeffs, positions, Z_eff, a0)

    # normalize on slice
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    psi2d /= np.sqrt(np.sum(np.abs(psi2d)**2) )
    psi_plot = psi2d.real

    # plot with square axes and colorbar
    vlim = np.max(np.abs(psi_plot))
    
    if type(ax) == type(None) :
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(1, 10, figure=fig)
        ax  = fig.add_subplot(gs[:, :-1])
        cax = fig.add_subplot(gs[:, -1])
        
        print(np.linalg.norm(psi_plot.flatten()))
        im = ax.imshow(psi_plot, extent=[x_min, x_max, y_min, y_max],
                    origin='lower', cmap='seismic', vmin=-vlim, vmax=vlim,
                    aspect='equal')
        # overlay lattice centers in view
        px = positions[:,:,0].ravel()
        py = positions[:,:,1].ravel()
        mask = (px>=x_min)&(px<=x_max)&(py>=y_min)&(py<=y_max)
        # ax.scatter(px[mask], py[mask], c='white', edgecolors='black',
        #         s=50, linewidths=1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Heatmap' + (' (Centered)' if center_on_mean else ''))
        fig.colorbar(im, cax=cax, label=r'$\Re[\psi(x,y,0)]$')
        plt.tight_layout(pad=0)
        plt.show()
        
    else : 
        
        print("psi_norm : ",np.linalg.norm(psi_plot.flatten()))
        print("vlim : ", vlim)
        # im = ax.imshow(psi_plot, extent=[x_min, x_max, y_min, y_max],
        #             origin='lower', cmap='seismic', vmin=-vlim, vmax=vlim,
        #             aspect='equal',interpolation=None)
        im = ax.pcolormesh(xs, ys, psi_plot, shading='auto',
                        cmap='seismic', vmin=-vlim, vmax=vlim)
        # overlay lattice centers in view
        px = positions[:,:,0].ravel()
        py = positions[:,:,1].ravel()
        mask = (px>=x_min)&(px<=x_max)&(py>=y_min)&(py<=y_max)
        
        if ifScatter :
            ax.scatter(px+xShift, py, c='white', edgecolors='black',
                    s=scatter_size, linewidths=1,zorder=20)
        
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"Plot with Orbitals, a={lattice_a}")
        return im




L = int(sys.argv[1])

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



    dk=2*np.pi/L

    points_cart = np.array([(ii+2/3)*a1 + (jj+2/3)*a2 for ii in range(L) for jj in range(L)])

    points_ion_cart = np.array([(ii)*a1 + (jj)*a2 for ii in range(L) for jj in range(L)])

    points_lat = np.array([(ii+2/3, jj+2/3) for ii in range(L) for jj in range(L)])
    points_ion_lat = np.array([(ii, jj) for ii in range(L) for jj in range(L)])

if True :
    pspace1_pt = np.exp(1j*(points_lat[:,0]*dk))   # np.exp(1j*dk1*xmesh) #

    pspace2_pt = np.exp(1j*(points_lat[:,1]*dk))   # np.exp(1j*dk2*ymesh) #

    pspace3_pt = np.conj(pspace1_pt)*np.conj(pspace2_pt)

    x_points, y_points = get_xy_WSe2(L,a1,a2,no_orbs=True) 

    rs = np.zeros((*x_points.diagonal().shape,2),dtype=complex)
    rs[...,0] = x_points.diagonal()
    rs[...,1] = y_points.diagonal()


    four_T = build_fourier_operator(ks.reshape(L**2,2),rs)

    four_T = np.kron(four_T,np.eye(3,dtype=complex))

    p_spaces = [pspace1_pt[0],pspace2_pt[0],(pspace3_pt[0])] #[np.array([np.exp(-1j*(2*np.pi/L)*1/3)]), np.array([np.exp(1j*(2*np.pi/L)*2/3)]), np.array([np.exp(-1j*(2*np.pi/L)*1/3)])] #
    ops = [X_U1,X_U2,X_U3]

    init_params = {
        "unitary_real_embed_style" : 'separate',
        'enforce_unitarity' : False
    }
    
    sx, sy, sz = mm.get_pauli("xyz")
    
    Gammas = mm.cliff_generators(3)

    obj = Localizer(ops[:], p_spaces[:],init_param=init_params,clifford_elements=Gammas) #  ,clifford_elements=[sz,sx,sy]
    cdim = len(Gammas[0])
    
    obj.build_locs()
    

    ws_loc,vs_loc = np.linalg.eigh(obj.locs[0])
    
    
    
vec_loc = vs_loc[:,len(ws_loc)//2]

Us,Ss,Vs = np.linalg.svd(vec_loc.reshape(U.shape[1],cdim),full_matrices=False)

vec_single = Us[:,0]

wf_kspace = vec_single/np.abs(vec_single)


hvec_single_rspace = (four_T @ U @ wf_kspace).reshape(L,L,3)

hvec_single_rspace = np.roll(hvec_single_rspace,(L//2,L//2),axis=(0,1))

L=24
coeffs_me = hvec_single_rspace.reshape(L,L,3)

# Physical parameters for Tungsten (W) 
Z_atomic     = 74                       # Atomic number for Tungsten
sigma        = 60 + 3*0.35              # Screening constant (Slater's rules)
Z_eff        = Z_atomic - sigma         # Effective nuclear charge for 5d electrons
a0_angstrom  = 0.529177                 # Bohr radius in Angstrom
lattice_a    = 3.325                    # W lattice constant in Angstrom from PHYSICAL REVIEW B 88, 085433 (2013) Table  II

fig,ax = plt.subplots(1)

mean_pad = 2
ifScatter=True
scatter_size=50

im = plot_heatmap_state_multi_origin(coeffs_me, lattice_a, Z_eff, a0_angstrom,
                                padding=0.5, grid_N=200,
                                center_on_mean=True, padding_mean=mean_pad,ax=ax,ifScatter=ifScatter,scatter_size=scatter_size)

ax.set_aspect("equal")
ax.set(xticks=[],yticks=[])
pp.add_cbar(im,fig,ax,width=0.03,title=r"$ \psi(x,y) $",ticks=[0],tick_labels=[0],title_pad=12)


fig.savefig(f"./figs/main_text_figure_1_c_WF_continuum_plot_a={lattice_a}_mean_pad={mean_pad}.png",bbox_inches="tight")
    



