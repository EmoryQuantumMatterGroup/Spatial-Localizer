"""Module for generic matrices and operations"""

import numpy as np
import scipy as sp

def direct_sum(A,B, dtype=complex) :
    
    """Direct Sum for two matrices of equal size

    Args:
        A (ndarray): Matrix of same size as B
        B (ndarray): Matrix of same size as A
        dtype :
    """
    
    return np.kron(np.array([[1,0],[0,0]],dtype=dtype),A) + np.kron(np.array([[0,0],[0,1]],dtype=dtype),B)
    

def diag_subspace(O,vs,return_ws=False, if_Hermitian=False) : 
    
    sub_O = np.conj(vs.T) @ O @ vs 
    
    
    if if_Hermitian :
        ws_sub, vs_sub = np.linalg.eigh(sub_O)
        
        vs_full = vs @ vs_sub
    
    
        if return_ws : 
            return ws_sub,vs_full
        else : 
            return vs_full
    
    else :
        ws_sub, vs_sub = np.linalg.eig(sub_O)
        
        vs_full = vs @ vs_sub
    
    
        if return_ws : 
            return ws_sub,vs_full
        else : 
            return vs_full

def get_pauli(choice) :
    
    """Returns a given choice of Pauli matrices. Choices are 'x', 'y', 'z', 'xyz', and 'I'. 
    'xyz' returns a tuple of Pauli matrices."""
    
    match choice :
        
        case "x" :
            return np.array([[0,1],[1,0]], dtype=complex)
        
        case "y" :
            return np.array([[0,-1j],[1j,0]], dtype=complex)
        
        case "z" :
            return np.array([[1,0],[0,-1]], dtype=complex)
        
        case "I" :
            return np.eye(2,dtype=complex)
        
        case 'xyz' :
            
            return [get_pauli('x'), get_pauli('y'), get_pauli('z')]
        
        case None : 
            return None
        
        case _ :
            raise RuntimeError("Invalid Pauli matrix choice given.")
        
def get_gamma(choice) :
    
    """Returns a given choice of gamma matrices"""
    
    match choice :
        
        case '0' : 
            return np.kron(get_pauli('z'),np.eye(2,dtype=complex))
        
        case '1' :
            return np.kron(-1*get_pauli('y'),get_pauli('x'))
        
        case '2' :
            return np.kron(-1*get_pauli('y'),get_pauli('y'))
        
        case '3' :
            return np.kron(-1*get_pauli('y'),get_pauli('z'))
        
        case '4' :
            return np.kron(get_pauli('x'), np.eye(2,dtype=complex))
        
        case 'all' : 
            return [ get_gamma('1'), get_gamma('2'), get_gamma('3'), get_gamma('4'), get_gamma('0')]
        
        case _ : 
            raise RuntimeError("Invalid gamma matrix choice given.")
        
def get_gamma_pbc_nice(choice) :
    
    """Returns a given choice of gamma matrices"""
    
    match choice :
        
        case '2' :
            return np.kron(get_pauli('I'),get_pauli('x'))
        
        case '1' :
            return np.kron(get_pauli('x'),get_pauli('z'))
        
        case '4' :
            return np.kron(get_pauli('I'),get_pauli('y'))
        
        case '3' :
            return np.kron(get_pauli('y'),get_pauli('z'))
        
        case '5' : 
            return np.kron(get_pauli('z'),get_pauli('z'))
        
        case 'all' : 
            return [ get_gamma_pbc_nice('1'), get_gamma_pbc_nice('2'), get_gamma_pbc_nice('3'), get_gamma_pbc_nice('4'), get_gamma_pbc_nice('5')]
        
        case _ : 
            raise RuntimeError("Invalid gamma matrix choice given.")


def get_gamma_pbc_nice_alt(choice) :
    
    """Returns a given choice of gamma matrices"""
    
    match choice :
        
        case '1' :
            return np.kron(get_pauli('x'),get_pauli('I'))
        
        case '2' :
            return np.kron(get_pauli('z'),get_pauli('x'))
        
        case '3' :
            return np.kron(get_pauli('y'),get_pauli('I'))
        
        case '4' :
            return np.kron(get_pauli('z'), get_pauli('y'))
        
        case '5' : 
            return np.kron(get_pauli('z'),get_pauli('z'))
        
        case 'all' : 
            return [ get_gamma_pbc_nice('1'), get_gamma_pbc_nice('2'), get_gamma_pbc_nice('3'), get_gamma_pbc_nice('4'), get_gamma_pbc_nice('5')]
        
        case _ : 
            raise RuntimeError("Invalid gamma matrix choice given.")

def get_gamma_Sol_Vand(choice) :
    
    """Returns a given choice of gamma matrices"""
    
    match choice :
        
        case '1' : 
            return np.kron(np.eye(2,dtype=complex), get_pauli('x'))
        
        case '2' :
            return np.kron(np.eye(2, dtype=complex),get_pauli('z'))
        
        case '3' :
            return np.kron(get_pauli('x'),get_pauli('y'))
        
        case '4' :
            return np.kron(get_pauli('y'),get_pauli('y'))
        
        case '5' :
            return np.kron(get_pauli('z'), get_pauli('y'))
        
        case _ : 
            raise RuntimeError("Invalid gamma matrix choice given.")
        
def cliff_generators(k) :
    """Function to create Clifford generators as complex matrices for a Euclidean Clifford algebra of dimension 2k (i.e., Cl_{2k,0}) via the Weyl-Brauer matrices.
    The matrices will be of dimension 

    Args:
        k (int): Specifies dimension of the Clifford generators. 
        
    Returns : 
        Gammas (list) : List of Clifford generators in the order [P_1,Q_1,P_2,Q_2,...,P_k,Q_k]
    """
     
    
    s0 = get_pauli("I")
    sx,sy,sz = get_pauli("xyz")
    
    
    generators = []
    used_P = 0
    for ip in range(k) : 
        for i_scan in range(k) :
            if i_scan == 0 : # create first slot
                
                if ip == 0 :
                    used_P += 1
                    
                    # print(f"filling {ip} generator")
                    P_i = np.copy(sx)
                    Q_i = np.copy(-1*sy)
                else :
                    P_i = np.copy(sz)
                    Q_i = np.copy(sz)
                    
            else : # kronecker with everything else
                
                if ip == i_scan :
                    used_P += 1
                    
                    # print(f"filling {ip} generator")
                    P_i = np.kron(P_i,sx   )
                    Q_i = np.kron(Q_i,-1*sy)
                    
                elif i_scan < ip :

                    P_i = np.kron(P_i,sz)
                    Q_i = np.kron(Q_i,sz)
                
                elif i_scan > ip :

                    P_i = np.kron(P_i,s0)
                    Q_i = np.kron(Q_i,s0)
                
                else : 
                    raise ValueError("Something went wrong")
        
        generators.append(np.copy(P_i))
        generators.append(np.copy(Q_i))
    
    # print(f"Used_p : {used_P}")
        
    return generators
        
def project(X,U) :

    """Returns a projected copy of X"""
    
    arr = np.matmul(X,U)
    arr = np.matmul(np.conj(np.transpose(U)),arr)

    return arr

def exp_val(O,vec) : 
    """Returns the expectation value of operator O w.r.t. vector vec"""
    
    Ovec = np.matmul(O,vec)
    
    return np.dot(np.conj(vec),Ovec)

def variance(O,vec,ifReal=True) : 
    return exp_val(O@O,vec).real - (exp_val(O,vec).real)**2 if ifReal else exp_val(O@O,vec) - (exp_val(O,vec))**2


def exp_diags(X,coeff) : 
    
    """Always hand me a copy if you want to retain original array! Equivalent to the matrix exponential of a diagonal matrix"""   
    
    for ii in range(len(X)) :
        X[ii,ii] = np.exp(coeff*X[ii,ii])
    
    return X
        
def cos_diags(X,coeff) :    
    """Always hand me a copy if you want to retain original array! Equivalent to the matrix cosine of a diagonal matrix""" 
    
    for ii in range(len(X)) :
        X[ii,ii] = np.cos(coeff*X[ii,ii])
    
    return X

def sin_diags(X,coeff) :   
    """Always hand me a copy if you want to retain original array! Equivalent to the matrix sin of a diagonal matrix""" 
     
    for ii in range(len(X)) :
        X[ii,ii] = np.sin(coeff*X[ii,ii])
    
    return X

def normalize(vec) :
    """Give me a copy if you want to retain the original vector!"""
    vec = vec/np.linalg.norm(vec)
    
    return vec

def commutator(A,B) :
    return A @ B - B @ A
def anticommutator(A,B) :
    return A @ B + B @ A


def sparse_diag(dense_matrix, N, return_eigenvectors=True, clean=True,tol=1e-8, sparse_type=sp.sparse.csr_array):
    """
    Converts a dense matrix to a sparse matrix, and computes the N smallest eigenvalues
    and optionally their associated eigenvectors.

    Parameters:
    dense_matrix (np.ndarray): The dense matrix to convert.
    N (int): The number of smallest eigenvalues and eigenvectors to compute.
    return_eigenvectors (bool): Whether to return the eigenvectors. Default is True.

    Returns:
    tuple: A tuple containing:
        - eigenvalues (np.ndarray): The N smallest eigenvalues of the sparse matrix.
        - eigenvectors (np.ndarray, optional): The eigenvectors corresponding to the eigenvalues, 
          only if return_eigenvectors is True.
    """
    # cleaning
    
    if clean :
        dense_matrix[abs(dense_matrix) < tol] = 0
    
    # Convert dense matrix to sparse (CSR format)
    
    sparse_matrix = sparse_type(dense_matrix)
    
    #print(sparse_matrix.count_nonzero() / (sparse_matrix.shape[0] * sparse_matrix.shape[1]))

    # Compute the N smallest eigenvalues and eigenvectors
    if return_eigenvectors:
        eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(sparse_matrix, k=N, which='LM',sigma=0)  # 'SM' for smallest magnitude
        return eigenvalues, eigenvectors
    else:
        eigenvalues = sp.sparse.linalg.eigsh(sparse_matrix, k=N, which='LM',sigma=0, return_eigenvectors=False)  # Only eigenvalues
        return eigenvalues
    
    
def sorted_eigen_by_magnitude(matrix, hermitian=True, get_vecs=True):
    """
    Compute the eigenvalues and eigenvectors of a matrix, 
    sorted by the absolute magnitude of the eigenvalues in ascending order.

    Parameters:
        matrix (numpy.ndarray): A Hermitian (or real symmetric) matrix.

    Returns:
        sorted_eigenvalues (numpy.ndarray): Eigenvalues sorted by absolute magnitude.
        sorted_eigenvectors (numpy.ndarray): Corresponding eigenvectors sorted accordingly.
    """
    if get_vecs :
        if hermitian :
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        else :
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # Sort by absolute magnitude of eigenvalues (ascending order)
        sorted_indices = np.argsort(np.abs(eigenvalues),axis=-1)
        sorted_eigenvalues = eigenvalues[..., sorted_indices]
        sorted_eigenvectors = eigenvectors[..., sorted_indices]
        
        return sorted_eigenvalues, sorted_eigenvectors
    else : 
        if hermitian :
            # Compute eigenvalues and eigenvectors
            eigenvalues = np.linalg.eigvalsh(matrix)
        else :
            # Compute eigenvalues and eigenvectors
            eigenvalues = np.linalg.eigvals(matrix)
        
        # Sort by absolute magnitude of eigenvalues (ascending order)
        sorted_indices = np.argsort(np.abs(eigenvalues))
        sorted_eigenvalues = eigenvalues[sorted_indices]
        return sorted_eigenvalues
    
def sorted_eigen_by_magnitude_old(matrix, hermitian=True):
    """
    Compute the eigenvalues and eigenvectors of a matrix, 
    sorted by the absolute magnitude of the eigenvalues in ascending order.

    Parameters:
        matrix (numpy.ndarray): A Hermitian (or real symmetric) matrix.

    Returns:
        sorted_eigenvalues (numpy.ndarray): Eigenvalues sorted by absolute magnitude.
        sorted_eigenvectors (numpy.ndarray): Corresponding eigenvectors sorted accordingly.
    """
    if hermitian :
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    else :
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Sort by absolute magnitude of eigenvalues (ascending order)
    sorted_indices = np.argsort(np.abs(eigenvalues))
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    return sorted_eigenvalues, sorted_eigenvectors



def clean_arr(arr,tol=1e-8) :
    
    shape = arr.shape
    temp_arr = arr.flatten()
    
    for ii in range(len(temp_arr)) :
        if np.abs(temp_arr[ii]) < tol :
            temp_arr[ii] = 0 
            
    return temp_arr.reshape(shape)


def get_overlap_matrix(vecs, list_or_operator="list") :
    
    
    match list_or_operator :
        case "list" : 
            num_vecs = len(vecs)
            
            overlaps = np.zeros((num_vecs,num_vecs))
            
            for ii,vec1 in enumerate(vecs) :
                for jj,vec2 in enumerate(vecs) :
                    overlaps[ii,jj] = abs(np.dot(np.conj(vec1),vec2))
        
        case "operator" :
            overlaps = np.conj(vecs.T) @ vecs
    
    return overlaps



def unt_to_real_imag(O, use_diag=False) : 
    """Takes a unitary operator and returns the real and imaginary parts

    Args:
        O (_ndarray_): Some unitary (or generally complex) matrix represented as a 2D ndarray
        
    Returns : 
        O_real(_ndarray_), O_imag(_ndarray_)
    """
    
    if use_diag :
        w,v = np.linalg.eig(O) 
        
        O_real = v @ np.diag(w.real) @ np.conj(v.T)
        O_imag = v @ np.diag(w.imag) @ np.conj(v.T)
    
    else : 
        O_dagger = np.conj(np.moveaxis(O,-1,-2)) # transposing last two dimensions
        O_real = 0.5*(O + O_dagger)
        O_imag = -0.5j*(O - O_dagger)
    
    return O_real, O_imag

def enforce_unitarity(O, print_metrics=False) :
    
    V_in, S, V_out = sp.linalg.svd(O,lapack_driver="gesvd")
    
    if print_metrics :
        diff = np.round((np.sum(np.eye(len(S)) - (np.diag(S))))/len(S),2)
        mean = np.round(np.mean(S),2)
        min = np.round(np.min(S),2)
        max = np.round(np.max(S),2)
        
        print(f"singular value metrics | max : {max} | min : {min} | mean : {mean} | average deviation from 1 : {diff}")
        
    return V_in @ V_out

def lowdin_ortho(V,test_S = False,tol=1e-8, num_S = 3,return_S=False) :
    """Implements Lowdin symmetrix orthogonalization for a given basis.

    Args:
        V (_type_): Numpy array with basis as column vectors.

    Returns:
        _type_: _description_
    """
    
    ov = np.conj(V[:,:].T) @ V[:,:]

    w,v = np.linalg.eigh(ov)
    
    if test_S : 
    
        min_w = np.min(np.abs(w))
        max_w = np.max(np.abs(w))
        if min_w/max_w < tol : 
            
            sorted_ws = np.sort(np.abs(w))/max_w

            print(f"bad eval in Lowdin : {sorted_ws[:num_S]}")



    w2 = w**(-1/2)

    S_half = v @ np.diag(w2) @ np.conj(v.T)
    
    if return_S : 
        return V @ S_half, w/np.max(w.__abs__())
        
    else : 
        return V @ S_half # act on basis and return
        
        
        
# def compute_localization_functional_2D(WFs, r_x, r_y):
#     """
#     Compute the gauge-invariant and gauge-dependent parts of the localization functional for a 2D system.

#     Parameters:
#     - WFs: numpy array of shape (num_cells_x, num_cells_y, num_bands, num_states)
#         Contains Wannier functions labeled by (unit cell_x, unit cell_y, band index) in the Hamiltonian basis.
#     - r_x, r_y: numpy arrays of shape (num_states, num_states)
#         Position operator matrices in the Hamiltonian basis for x and y directions.

#     Returns:
#     - Omega_I: Gauge-invariant part.
#     - Omega_D: Gauge-dependent diagonal part.
#     - Omega_OD: Gauge-dependent off-diagonal part.
#     - Omega: Total spread functional.
#     """
#     num_cells_x, num_cells_y, num_bands, num_states = WFs.shape

#     # Apply position operators to the Wannier functions: <Rm | r_i | 0n >
#     WF_r_x = np.tensordot(WFs.conj(), r_x, axes=([3], [0]))  # (num_cells_x, num_cells_y, num_bands, num_bands)
#     WF_r_y = np.tensordot(WFs.conj(), r_y, axes=([3], [0]))  # (num_cells_x, num_cells_y, num_bands, num_bands)

#     # Compute < 0n | r_i | 0n > for (R_x, R_y) = (0, 0) (Wannier centers)
#     r_bar_x = np.sum(WF_r_x[0, 0, :, :], axis=1)  # (num_bands,)
#     r_bar_y = np.sum(WF_r_y[0, 0, :, :], axis=1)  # (num_bands,)

#     # Compute < 0n | r_i^2 | 0n > for (R_x, R_y) = (0, 0)
#     WF_r2_x = np.tensordot(WFs.conj(), r_x @ r_x, axes=([3], [0]))
#     WF_r2_y = np.tensordot(WFs.conj(), r_y @ r_y, axes=([3], [0]))

#     r2_mean_x = np.sum(WF_r2_x[0, 0, :, :], axis=1)  # (num_bands,)
#     r2_mean_y = np.sum(WF_r2_y[0, 0, :, :], axis=1)  # (num_bands,)

#     # Total mean squared position
#     r2_mean_total = r2_mean_x + r2_mean_y

#     # Compute gauge-invariant part Omega_I
#     Omega_I = np.sum(r2_mean_total - (
#         np.sum(np.abs(WF_r_x)**2, axis=(0, 1, 2, 3)) +
#         np.sum(np.abs(WF_r_y)**2, axis=(0, 1, 2, 3))
#     ))

#     # Compute gauge-dependent diagonal part Omega_D (sum over (R_x, R_y) ≠ (0,0))
#     Omega_D = (
#         np.sum(np.abs(WF_r_x[1:, :, :, :])**2) +
#         np.sum(np.abs(WF_r_x[:, 1:, :, :])**2) +
#         np.sum(np.abs(WF_r_y[1:, :, :, :])**2) +
#         np.sum(np.abs(WF_r_y[:, 1:, :, :])**2)
#     )  # Ignore (R_x, R_y) = (0,0)

#     # Compute gauge-dependent off-diagonal part Omega_OD (inter-band terms)
#     Omega_OD = (
#         np.sum(np.abs(WF_r_x)**2) +
#         np.sum(np.abs(WF_r_y)**2)
#     ) - Omega_D

#     # Total spread functional
#     Omega = Omega_I + Omega_D + Omega_OD

#     return Omega_I, Omega_D, Omega_OD, Omega
    



def compute_localization_func_2_bands(vecs, X, Y) : 
    """Function to compute the gauge independent and gauge dependent portions of the Marzari-Vanderbilt localization functional. 

    Args:
        vecs (_ndarray_): Wannier functions, shaped as (h_dim,site_index,band_index) where h_dim holds the wannier function in the full Hilbert space.
        X (_ndarray_): non projected X position operator
        Y (_ndarray_): non projected Y position operator
    """
    def variance(O,v) : 
        
        return exp_val(O @ O,v) - exp_val(O,v)**2
       
    h_dim, num_sites, num_bands = vecs.shape
    
    if num_bands != 2 :
        raise ValueError("Number of bands is different than 2, please make sure you want to use this function.")
    
    vecs_op = vecs.reshape((h_dim,num_sites*num_bands)) # reshaping into a matrix of column vectors
    
    
    P = vecs_op @ np.conj(vecs_op.T) # projector onto occupied bands
    
    Q = np.eye(len(P)) - P # projector onto unocupied bands 
    
    
    total_var = np.zeros(num_sites,dtype=complex)
    gauge_invariant = np.zeros(num_sites) # gauge invariant contributions
    PxQx  = P @ X @ Q @ X 
    PyQy = P @ Y @ Q @ Y
    
    
    # calculating gauge invariant piece
    for site_index in range(num_sites) : # iterating through all sites
        
        
        
        current_vecs = vecs[:,site_index,:]
        
        
        total_var[site_index] += variance(X,current_vecs[:,0]) + variance(Y,current_vecs[:,0])
        total_var[site_index] += variance(X,current_vecs[:,1]) + variance(Y,current_vecs[:,1])
        
        
        vx = PxQx @ current_vecs
        vy = PyQy @ current_vecs
        
        valx = np.trace(np.abs((np.conj(current_vecs.T) @ vx)))
        valy = np.trace(np.abs((np.conj(current_vecs.T) @ vy)))
        
        gauge_invariant[site_index] = (valx) + (valy)
        
    # calculating gauge-dependent pieces
    
    
    gauge_dependent = np.zeros(num_sites)
    
    for site_index in range(num_sites) : # iterating through all sites
        
        current_vecs = vecs[:,site_index,:]
        vxs = X @ current_vecs
        vys = Y @ current_vecs
        
        for jj in range(num_sites) : # iterating through all sites again
            
            if jj == site_index : # if on the same site
                
                gauge_dependent[site_index] += 2*np.abs(np.vdot(current_vecs[:,1],vxs[:,0]))**2 + 2*np.abs(np.vdot(current_vecs[:,1],vys[:,0]))**2
                
            else : 
                current_vecs_2 = vecs[:,jj,:] 
                
                Mx = np.conj(current_vecs_2.T) @ vxs
                My = np.conj(current_vecs_2.T) @ vys
                
                gauge_dependent[site_index] += np.sum(np.abs(Mx)**2) + np.sum(np.abs(My)**2)
                
                
                
    return gauge_invariant, gauge_dependent, total_var


def compute_localization_func_1_band(vecs, X, Y) : 
    """Function to compute the gauge independent and gauge dependent portions of the Marzari-Vanderbilt localization functional. 

    Args:
        vecs (_ndarray_): Wannier functions, shaped as (h_dim,site_index,band_index) where h_dim holds the wannier function in the full Hilbert space.
        X (_ndarray_): non projected X position operator
        Y (_ndarray_): non projected Y position operator
    """
    def variance(O,v) : 
        
        return exp_val(O @ O,v) - exp_val(O,v)**2
       
    h_dim, num_sites = vecs.shape
    

    
    vecs_op = vecs.reshape((h_dim,num_sites)) # reshaping into a matrix of column vectors
    
    
    P = vecs_op @ np.conj(vecs_op.T) # projector onto occupied bands
    
    Q = np.eye(len(P)) - P # projector onto unocupied bands 
    
    
    total_var = np.zeros(num_sites,dtype=complex)
    gauge_invariant = np.zeros(num_sites) # gauge invariant contributions
    PxQx  = P @ X @ Q @ X 
    PyQy = P @ Y @ Q @ Y
    
    
    # calculating gauge invariant piece
    for site_index in range(num_sites) : # iterating through all sites
        
        
        
        current_vecs = vecs[:,site_index]
        
        
        total_var[site_index] += variance(X,current_vecs[:]) + variance(Y,current_vecs[:])

        
        
        vx = PxQx @ current_vecs
        vy = PyQy @ current_vecs
        
        valx = (np.abs((np.conj(current_vecs.T) @ vx)))
        valy = (np.abs((np.conj(current_vecs.T) @ vy)))
        
        gauge_invariant[site_index] = (valx) + (valy)
        
    # calculating gauge-dependent pieces
    
    
    gauge_dependent = np.zeros(num_sites)
    
    for site_index in range(num_sites) : # iterating through all sites
        
        current_vecs = vecs[:,site_index]
        vxs = X @ current_vecs
        vys = Y @ current_vecs
        
        for jj in range(num_sites) : # iterating through all sites again
            
            if jj == site_index : # if on the same site
                
                pass
                
            else : 
                current_vecs_2 = vecs[:,jj] 
                
                Mx = np.conj(current_vecs_2.T) @ vxs
                My = np.conj(current_vecs_2.T) @ vys
                
                gauge_dependent[site_index] += np.sum(np.abs(Mx)**2) + np.sum(np.abs(My)**2)
                
                
                
    return gauge_invariant, gauge_dependent, total_var

def roll_vec_QTI(vec,L,roll_x=0, roll_y=0) :
    
    arr = vec.reshape(L,2,L,2) # basis x, orb, y, orb
    
    arr = np.roll(arr, roll_x, 0) # roll in x 
    
    arr = np.roll(arr, roll_y, 2) # roll in y
    
    return arr.reshape(4*L**2) # return reshaped array


def roll_vec_BHZ(vec,L,roll_x=0, roll_y=0) :
    
    arr = vec.reshape(L,L,4) # basis x, orb, y, orb
    
    arr = np.roll(arr, roll_x, 0) # roll in x 
    
    arr = np.roll(arr, roll_y, 1) # roll in y
    
    return arr.reshape(4*L**2) # return reshaped array
    
def roll_vec_QWZ(vec,L,roll_x=0, roll_y=0) :
    
    arr = vec.reshape(L,L,2) # basis x, orb, y, orb
    
    arr = np.roll(arr, roll_x, 0) # roll in x 
    
    arr = np.roll(arr, roll_y, 1) # roll in y
    
    return arr.reshape(2*L**2) # return reshaped array

def overlap_checks_degenerate_2_bands(points, vecs, L, roll_func=roll_vec_QTI) :
    
    ov = 0
    
    for init_ii in range(len(points)) : ## overlapping checks
        for final_ii in range(len(points[init_ii:])) :
        
            
            dpt = np.round(points[final_ii] -  points[init_ii],0)

            arr =  roll_func(vecs[:,2*init_ii],L,roll_x=int(dpt[0]),roll_y=int(dpt[1]))
            arr2 = roll_func(vecs[:,2*init_ii+1],L,roll_x=int(dpt[0]),roll_y=int(dpt[1]))

            arr3 = vecs[:,2*final_ii]
            arr4 = vecs[:,2*final_ii+1]


            ov += 2 # maximum overlap between two vector spaces of dimension 2
            for v1 in (arr, arr2) :
                for v2 in (arr3, arr4) :
                    ov -=  np.abs(np.vdot(v1,v2))**2
                    
    return ov/len(points)

def overlap_checks_degenerate_1_band(points, vecs, L, roll_func=roll_vec_QTI) :
    
    ov = 0
    
    for init_ii in range(len(points)) : ## overlapping checks
        for final_ii in range(len(points[init_ii:])) :
        
            
            dpt = np.round(points[final_ii] -  points[init_ii],0)

            arr =  roll_func(vecs[:,init_ii],L,roll_x=int(dpt[0]),roll_y=int(dpt[1]))


            arr2 = vecs[:,final_ii]



            ov += 1 # maximum overlap between two vector spaces of dimension 2
            ov -=  np.abs(np.vdot(arr,arr2))**2
                    
    return ov/len(points)









def ft_1D(L) : 
    
    op = np.zeros((L,L),dtype=complex)
    
    for ix in range(L) : 
        for ik in range(L) :
            
            op[ik,ix] = (1/np.sqrt(L))*np.exp(2j*-np.pi*(ik)*(ix)/L)
            
    return op  
            
def ft_2D(Lx,Ly) :
    """Fourier transform, not inverse (i.e., from position to momentum)"""
    
    
    ft_x = ft_1D(Lx)
    ft_y = ft_1D(Ly)
    
    return np.kron(ft_x,ft_y)








