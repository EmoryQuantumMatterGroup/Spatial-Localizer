import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import sys
import os


sys.path.insert(0,"./../../")

from toolkit_local import matrices as mm

def get_face_shifts_and_spanners() :
    
    (a1,a2,a3), _ = f222_lat_vecs()

    faces_shifts =   [np.zeros(3),a1+a3,a2+a3,(a1+a2+a3)*0.5,(a1+a2+a3)*0.5,(a1+a2+a3)*0.5]
    faces_spanners = [[a1,a2],[a2,-a3],[a1,-a3],[0.5*a1,-0.5*a3],[0.5*a1,0.5*a2],[0.5*a2,-0.5*a3]]
    
    return faces_shifts,faces_spanners
    

def f222_rice_mele_ham(k,t,epsilon=1,delta=0.0) : 
    
    sx,sy,sz = mm.get_pauli("xyz")
    s0 = mm.get_pauli("I")
    
    kx,ky = k[0],k[1]
    kz = k[2]
    
    return (-epsilon/2)*(s0+sz) - (t/2)*(s0 + np.cos(kz)*sz + np.sin(kz)*sy) + delta*(np.cos(kx)-np.cos(ky))*np.sin(kz)*sx
    
def f222_lat_vecs(a=1,b=1,c=1) :
    
    a_pre_factor = 0.5
    
    b_pre_factor = 2*np.pi
    
    a1,a2,a3 = a_pre_factor*np.array([0,b,c]), a_pre_factor*np.array([a,0,c]), a_pre_factor*np.array([a,b,0]) 
    
    b1,b2,b3 = b_pre_factor*np.array([-1/a,1/b,1/c]),b_pre_factor*np.array([1/a,-1/b,1/c]),b_pre_factor*np.array([1/a,1/b,-1/c])
    
    
    return (a1,a2,a3), (b1,b2,b3)
    
    
    

