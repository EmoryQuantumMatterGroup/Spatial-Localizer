"""Module for various hamiltonians and their operators"""
import numpy as np
from .matrices import get_pauli



def getHam_QWZ_bloch(kx,ky,m, theta=0, n=1, vectorized=False) :
    
    sx,sy,sz = get_pauli('xyz')
    
    if vectorized :
        return np.tensordot(np.sin(kx),sx,0) + np.tensordot(np.sin(n*ky+theta),sy,0) + np.tensordot((m*np.ones(kx.shape,dtype=complex) + np.cos(kx) + np.cos(n*ky+theta)),sz,0)
    else :
        return np.sin(kx)*sx + np.sin(n*ky+theta)*sy + (m + np.cos(kx) + np.cos(n*ky+theta))*sz
 
def getHam_2D_Chern(L,m, bc='open') : 
    
    hopping = np.zeros((L,L),dtype=complex) # NN hopping matrix
    
    for ii in range(L-1) :
        hopping[ii,ii+1] = 1
        
    if bc=='pbc' :
        hopping[-1,0] = 1
    
    hop_x = np.kron(hopping, np.eye(L,dtype=complex)) # tensor with y basis
    
    hop_x = np.kron(hop_x, 0.5*(get_pauli('z') - 1j*get_pauli('x'))) # tensor with orbitals
    
    hop_y = np.kron(np.eye(L,dtype=complex),hopping) # tensor with x basis
    
    hop_y = np.kron(hop_y,0.5*(get_pauli('z') - 1j*get_pauli('y'))) # tensor with orbitals
    
    onsite = m*np.kron(np.eye(L**2,dtype=complex),get_pauli('z')) # onsite terms
    
    H = hop_x + hop_y
    
    H += np.conj(np.transpose(H)) # add h.c. hopping
    
    H += onsite # add onsite terms
    
    return H

def getX_2D_Chern(L, offset=0) :
    
    X1 = np.zeros((L,L),dtype=complex)
    X2 = np.zeros((L,L),dtype=complex)
    
    o1 = np.array([[1,0],[0,0]],dtype=complex)
    o2 = np.array([[0,0],[0,1]],dtype=complex)
    
    for ii in range(L) :
        X1[ii,ii] = ii - offset/2
        X2[ii,ii] = ii + offset/2
        
    
    X1 = np.kron(X1,np.eye(L))
    X2 = np.kron(X2,np.eye(L))

    X = np.kron(X1,o1) + np.kron(X2,o2)    
    
    
    
    return X

def getY_2D_Chern(L,offset=0) :
    
    X1 = np.zeros((L,L),dtype=complex)
    X2 = np.zeros((L,L),dtype=complex)
    
    o1 = np.array([[1,0],[0,0]],dtype=complex)
    o2 = np.array([[0,0],[0,1]],dtype=complex)
    
    for ii in range(L) :
        X1[ii,ii] = ii - offset/2
        X2[ii,ii] = ii + offset/2
        
    
    X1 = np.kron(np.eye(L),X1)
    X2 = np.kron(np.eye(L),X2)

    Y = np.kron(X1,o1) + np.kron(X2,o2)    
    
    return Y

def getXYH_Chern(L,m,bc) :
    
    X, Y = getX_2D_Chern(L), getY_2D_Chern(L)
    H = getHam_2D_Chern(L, m, bc=bc)
    
    return X,Y,H
