"""Module for various operations in condensed matter"""
import numpy as np

def get_K_vectors(a1,a2,a3) :
    """Returns the reciprocal lattice vectors, given some set of direct lattice vectors
        For 2D lattice, just let a3 = [0,0,1]
    """  
  
    b1 = 2*np.pi*(np.cross(a2,a3))/(np.dot(a1,np.cross(a2,a3)))
    b2 = 2*np.pi*(np.cross(a3,a1))/(np.dot(a2,np.cross(a3,a1)))
    b3 = 2*np.pi*(np.cross(a1,a2))/(np.dot(a3,np.cross(a1,a2)))    
    
  
    return b1,b2,b3  

def get_berry_connection_of_state_central(vec, dkx=1.0, dky=1.0, renorm=True, eps=1e-14):
    u = vec.astype(np.complex128, copy=True)

    if renorm:
        nrm = np.linalg.norm(u, axis=-1, keepdims=True)
        u /= np.where(nrm > eps, nrm, 1.0)

    # roll(-1) is + direction in index space
    u_xp = np.roll(u, shift=-1, axis=0)
    u_xm = np.roll(u, shift=+1, axis=0)
    u_yp = np.roll(u, shift=-1, axis=1)
    u_ym = np.roll(u, shift=+1, axis=1)

    s_xp = np.sum(np.conj(u) * u_xp, axis=-1)
    s_xm = np.sum(np.conj(u) * u_xm, axis=-1)
    s_yp = np.sum(np.conj(u) * u_yp, axis=-1)
    s_ym = np.sum(np.conj(u) * u_ym, axis=-1)

    Ax = -(1j * (s_xp - s_xm) / (2 * dkx)).real
    Ay = -(1j * (s_yp - s_ym) / (2 * dky)).real
    return Ax, Ay
