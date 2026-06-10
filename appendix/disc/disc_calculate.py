
import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp

import os 
import sys 

sys.path.insert(0,"./../../")
import toolkit_local.matrices as mm 
import toolkit_local.hdf5 as hdf

def get_unit_cell_loops_centered(X, Y, res=10):
    """
    Create square loops centered on each point in X, Y (from meshgrid),
    including the last row/column by padding the unit cells out.
    """
    Ny, Nx = X.shape
    dx = X[0,1] - X[0,0] if Nx > 1 else 1
    dy = Y[1,0] - Y[0,0] if Ny > 1 else 1

    loops = []
    for i in range(Ny):
        for j in range(Nx):
            xc = X[i, j]
            yc = Y[i, j]

            x0, x1 = xc - dx/2, xc + dx/2
            y0, y1 = yc - dy/2, yc + dy/2

            bottom = [(xi, y0) for xi in np.linspace(x0, x1, res, endpoint=False)]
            right  = [(x1, yi) for yi in np.linspace(y0, y1, res, endpoint=False)]
            top    = [(xi, y1) for xi in np.linspace(x1, x0, res, endpoint=False)]
            left   = [(x0, yi) for yi in np.linspace(y1, y0, res, endpoint=False)]

            loop = bottom + right + top + left
            loops.append(loop)

    return loops


def get_XY_disclination(N) : 
    x,y = np.meshgrid(np.array(range(0,N+1),dtype=float),np.array(range(0,N//2),dtype=float))
    
    print(x.shape)

    center = (N/2,-0.5)

    x -= center[0]
    y -= center[1]

    # plt.scatter(x,y)
    # plt.scatter(*(0,0),marker='*',s=500,zorder=-1)
    # plt.show()
    
    uc_arr = np.array(get_unit_cell_loops_centered(x,y,20))
    
    uc_x = uc_arr[...,0]
    uc_y = uc_arr[...,1]
    
    dis_uc = uc_x + 1j*uc_y
    
    # plt.plot(dis_uc.T.real,dis_uc.T.imag)
    # plt.show()
    
    dis_uc = dis_uc**(2/3)
    




    dis = x + 1j*(y)

    dis = dis**(2/3)

    dis2 = np.exp(2j*np.pi/3)*dis 

    dis3 = np.exp(2j*np.pi/3)*dis2 


    X_diag = np.zeros((3,*dis.shape),dtype=complex)

    X_diag[0,:,:] = dis.real
    X_diag[1,:,:] = dis2.real 
    X_diag[2,:,:] = dis3.real 

    Y_diag = np.zeros((3,*dis.shape),dtype=complex)

    Y_diag[0,:,:] = dis.imag
    Y_diag[1,:,:] = dis2.imag 
    Y_diag[2,:,:] = dis3.imag 
    
    X_diag = X_diag.flatten()
    
    Y_diag = Y_diag.flatten()
    
    return np.diag(X_diag), np.diag(Y_diag), (3,*x.shape), dis_uc


def hopping_internal(N,mat_shape=True) : 
    "hoppings inside of rectangles"
    hop = np.zeros((3,N//2,N+1,3,N//2,N+1),dtype=complex) 
    
    h_dim = 3*(N//2)*(N+1)
    
    hop_long = np.copy(hop)
    hop_short = np.copy(hop)
    
    for kk in range(3) : 
        for ii in range((N + 1 - 1)) : 
            for jj in range(N//2) : 
                hop_long[kk, jj,ii,kk, jj,ii+1] = 1 
                
        
            
        for ii in range((N//2) - 1) : 
            for jj in range((N+1)) : 
                hop_short[kk, ii,jj,kk, ii+1,jj] = 1 
            

    if mat_shape :
         
        return np.reshape(hop_short,(h_dim,h_dim)), np.reshape(hop_long,(h_dim,h_dim))
    
    else : 
        return hop_short,hop_long


def hopping_external(N,mat_shape=True) : 
    "hoppings between rectangles"
    
    h_dim = 3*(N//2)*(N+1)
    
    hop = np.zeros((3,N//2,N+1,3,N//2,N+1))
    
    hop_01 = np.copy(hop)
    
    hop_12 = np.copy(hop)
    
    hop_20 = np.copy(hop)
    
    for ii in range(N//2) : 
        hop_01[0,0,ii,1,0,-1-ii] = 1
        hop_12[1,0,ii,2,0,-1-ii] = 1
        hop_20[2,0,ii,0,0,-1-ii] = 1
    
    
    if mat_shape : 
        return np.reshape(hop_01,(h_dim,h_dim)), np.reshape(hop_12,(h_dim,h_dim)), np.reshape(hop_20,(h_dim,h_dim)) 
    else :
        return hop_01, hop_12, hop_20 
    
    
def hopping_inner_triangle(N,mat_shape=True) : 
    "hoppings between rectangles"
    
    h_dim = 3*(N//2)*(N+1)
    
    hop = np.zeros((3,N//2,N+1,3,N//2,N+1))
    hop_alt = np.zeros((3,N//2,N+1,3,N//2,N+1))
    
    hop[0,0,N//2,1,0,N//2] = 1
    hop[1,0,N//2,2,0,N//2] = 1
    hop_alt[2,0,N//2,0,0,N//2] = 1
    
    
    if mat_shape : 
        return np.reshape(hop,(h_dim,h_dim)), np.reshape(hop_alt,(h_dim,h_dim))
    else :
        return hop, hop_alt


def getHam_eq_54_ins(N,delta,intra_delta=0, inner_hops="square",delta_inner_triangle=0, delta_intra_tri_region=0) : 
    
    sx,sy,sz = mm.get_pauli("xyz")
    s0 = mm.get_pauli("I")
    
    
    o_1_short = np.array([[1,0],[0,0]],dtype=complex)
    
    o_2_long = np.array([[0,0],[0,1]],dtype=complex)
    
    
    # a_1 is hop_short
    
    if N % 2 == 1 : 
        N -= 1 
    
    
    hop_01, hop_12, hop_20 = hopping_external(N) 
    hop_short, hop_long = hopping_internal(N)
    
    hop_triangle, hop_tri_alt = hopping_inner_triangle(N)
    
    
    hop_short_orb = np.kron(o_1_short,0.5*(sx - 1j*sy))
    
    hop_long_orb = np.kron(o_2_long, 0.5*(sx - 1j*sy) )
    
    hop_orb_33 = np.zeros((4,4),dtype=complex)
    hop_orb_33[0,0] = 1
    # hop_orb_33[2,2] = 1    
    
    match inner_hops :
        case 'square' : 
                    
            inner_hop_orb = np.kron(sx, sx)
            
        case 'cross' : 
                    
            inner_hop_orb = np.kron(s0,sx)
            
        case _ : 
            
            raise ValueError("why?")
    
    
    H = delta*(np.kron(hop_short   ,hop_short_orb ) + np.kron(hop_long , hop_long_orb))
    
    H += delta_intra_tri_region*np.kron(hop_20 + hop_01 + hop_12,hop_orb_33)
    
    H += delta_inner_triangle*(np.kron(hop_triangle + hop_tri_alt, hop_orb_33))
    
    
    H += np.conj(H.T)
    
    H += intra_delta*np.kron(np.eye(len(hop_short),dtype=complex),inner_hop_orb)
    
    return H 




def calculate_topo(N,scan_res) :



    X,Y, shape, dis_uc = get_XY_disclination(N)

    X = np.kron(X,np.eye(4,dtype=complex))
    Y = np.kron(Y,np.eye(4,dtype=complex))

    t=1
    t_intra=0.3

    ham = getHam_eq_54_ins(N,t,inner_hops="square",intra_delta=t_intra, delta_inner_triangle=t,delta_intra_tri_region=t)
        
    ws,vs = np.linalg.eigh(ham)



    plt.plot(ws,'o',ms=3)

    # plt.ylim(-0.1,0.1)
    proj_min, proj_max =  -3,-0.4

    vs_1 = vs[:,ws<proj_max]

    ws_1 = ws[ws< proj_max]

    vs_2 = vs_1[:,ws_1 > proj_min]

    U = vs_2 #vs[:,ws<-0.2]

    X_U = mm.project(X,U)
    Y_U = mm.project(Y,U)

    eye = np.eye(len(Y_U),dtype=complex)

    scan_range = (-2.8,2.8)
    sx,sy,sz = mm.get_pauli("xyz")

    xs = np.linspace(*scan_range,scan_res)
    ys = xs[-1::-1] 



    wif_index = len(X_U)

    As = np.zeros((scan_res,scan_res))
    Bs = np.zeros_like(As)
    middles = np.zeros_like(As)
    mus = np.zeros_like(As)
    S1s = np.zeros_like(As)
    S2s = np.zeros_like(As)
    true_vars = np.zeros_like(As)
    deviations = np.zeros_like(As)

    for ii, x in enumerate(xs) : 
        for jj, y in enumerate(ys) :
            
            xtemp = X_U - x*eye
            ytemp = Y_U - y*eye
            
            comm = mm.commutator(xtemp,ytemp)
            
            loc = np.kron(ytemp, sy) + np.kron(xtemp,sx)
            
            ws,vs = np.linalg.eigh(loc)
            
            v_wif = vs[:,wif_index].reshape(wif_index,2)
            
            Us, Ss, Vs = np.linalg.svd(v_wif,full_matrices=False)
            
            
            p1 = Us[:,0]
            p2 = Us[:,1]
            
            s1, s2 = Ss
            
            S1s[jj,ii] = s1
            S2s[jj,ii] = s2
            
            mu = ws[wif_index]
            
            mus[jj,ii] = mu
            
            b1 = np.abs(np.vdot(p1, comm @ p1))
            b2 = (np.vdot(p1, xtemp @ xtemp @ p1) + np.vdot(p1, ytemp @ ytemp @ p1)).real
            b3 = mu**2 + (s2/s1)*np.abs(np.vdot(p1,comm @ p2))
            
            b4 = (np.vdot(p1,X_U @ p1) - x).real**2 + (np.vdot(p1,Y_U @ p1) - y).real**2
            
            b5 = mm.variance(X_U,p1) + mm.variance(Y_U,p1)
            
            As[jj,ii] = b1
            middles[jj,ii] = b2 - b4
            Bs[jj,ii] = b1 + b3 - b4
            true_vars[jj,ii] = b5
            deviations[jj,ii] = b4
            

    comp1s = middles - As

    comp2s = Bs - middles

    comp3s = 2*(mus**2 + np.linalg.norm(comm,ord=2)) - middles


    fname = f"./disc_h5s/bound_comparisons_topo_scan_res={scan_res}.h5"

    hdf.save_numpy_to_hdf5(fname,As,"/As")
    hdf.save_numpy_to_hdf5(fname,Bs,"/Bs")
    hdf.save_numpy_to_hdf5(fname,S1s,"/S1s")
    hdf.save_numpy_to_hdf5(fname,S2s,"/S2s")
    hdf.save_numpy_to_hdf5(fname,middles,"/middles")
    hdf.save_numpy_to_hdf5(fname,comp1s,"/comp1s")
    hdf.save_numpy_to_hdf5(fname,comp2s,"./comp2s")
    hdf.save_numpy_to_hdf5(fname,comp3s,"./comp3s")
    hdf.save_numpy_to_hdf5(fname,mus,"./mus")
    hdf.save_numpy_to_hdf5(fname,2*[*scan_range],"./extent")
    hdf.save_numpy_to_hdf5(fname,dis_uc,"./dis_uc")
    hdf.save_numpy_to_hdf5(fname,deviations,"./deviations")

def calculate_triv(N,scan_res) :



    X,Y, shape, dis_uc = get_XY_disclination(N)

    X = np.kron(X,np.eye(4,dtype=complex))
    Y = np.kron(Y,np.eye(4,dtype=complex))

    t=0.3
    t_intra=1

    ham = getHam_eq_54_ins(N,t,inner_hops="square",intra_delta=t_intra, delta_inner_triangle=t,delta_intra_tri_region=t)
        
    ws,vs = np.linalg.eigh(ham)



    plt.plot(ws,'o',ms=3)

    # plt.ylim(-0.1,0.1)
    proj_min, proj_max =  -3,-0.4

    vs_1 = vs[:,ws<proj_max]

    ws_1 = ws[ws< proj_max]

    vs_2 = vs_1[:,ws_1 > proj_min]

    U = vs_2 #vs[:,ws<-0.2]

    X_U = mm.project(X,U)
    Y_U = mm.project(Y,U)

    eye = np.eye(len(Y_U),dtype=complex)

    scan_range = (-2.8,2.8)
    sx,sy,sz = mm.get_pauli("xyz")

    xs = np.linspace(*scan_range,scan_res)
    ys = xs[-1::-1] 



    wif_index = len(X_U)

    As = np.zeros((scan_res,scan_res))
    Bs = np.zeros_like(As)
    middles = np.zeros_like(As)
    mus = np.zeros_like(As)
    S1s = np.zeros_like(As)
    S2s = np.zeros_like(As)
    true_vars = np.zeros_like(As)
    deviations = np.zeros_like(As)

    for ii, x in enumerate(xs) : 
        for jj, y in enumerate(ys) :
            
            xtemp = X_U - x*eye
            ytemp = Y_U - y*eye
            
            comm = mm.commutator(xtemp,ytemp)
            
            loc = np.kron(ytemp, sy) + np.kron(xtemp,sx)
            
            ws,vs = np.linalg.eigh(loc)
            
            v_wif = vs[:,wif_index].reshape(wif_index,2)
            
            Us, Ss, Vs = np.linalg.svd(v_wif,full_matrices=False)
            
            
            p1 = Us[:,0]
            p2 = Us[:,1]
            
            s1, s2 = Ss
            
            S1s[jj,ii] = s1
            S2s[jj,ii] = s2
            
            mu = ws[wif_index]
            
            mus[jj,ii] = mu
            
            b1 = np.abs(np.vdot(p1, comm @ p1))
            b2 = (np.vdot(p1, xtemp @ xtemp @ p1) + np.vdot(p1, ytemp @ ytemp @ p1)).real
            b3 = mu**2 + (s2/s1)*np.abs(np.vdot(p1,comm @ p2))
            
            b4 = (np.vdot(p1,X_U @ p1) - x).real**2 + (np.vdot(p1,Y_U @ p1) - y).real**2
            
            b5 = mm.variance(X_U,p1) + mm.variance(Y_U,p1)
            
            As[jj,ii] = b1
            middles[jj,ii] = b2 - b4
            Bs[jj,ii] = b1 + b3 - b4
            true_vars[jj,ii] = b5
            deviations[jj,ii] = b4
            

    comp1s = middles - As

    comp2s = Bs - middles

    comp3s = 2*(mus**2 + np.linalg.norm(comm,ord=2)) - middles


    fname = f"./disc_h5s/bound_comparisons_triv_scan_res={scan_res}.h5"

    hdf.save_numpy_to_hdf5(fname,As,"/As")
    hdf.save_numpy_to_hdf5(fname,Bs,"/Bs")
    hdf.save_numpy_to_hdf5(fname,S1s,"/S1s")
    hdf.save_numpy_to_hdf5(fname,S2s,"/S2s")
    hdf.save_numpy_to_hdf5(fname,middles,"/middles")
    hdf.save_numpy_to_hdf5(fname,comp1s,"/comp1s")
    hdf.save_numpy_to_hdf5(fname,comp2s,"./comp2s")
    hdf.save_numpy_to_hdf5(fname,comp3s,"./comp3s")
    hdf.save_numpy_to_hdf5(fname,mus,"./mus")
    hdf.save_numpy_to_hdf5(fname,2*[*scan_range],"./extent")
    hdf.save_numpy_to_hdf5(fname,dis_uc,"./dis_uc")
    hdf.save_numpy_to_hdf5(fname,deviations,"./deviations")


os.makedirs("./disc_h5s/",exist_ok=True)
os.makedirs("./figs/",exist_ok=True)


N = 6
scan_res = int(sys.argv[1])

calculate_topo(N,scan_res)
calculate_triv(N,scan_res)
    

