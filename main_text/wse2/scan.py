import sys
import os 
os.makedirs("./h5_files/",exist_ok=True)
os.makedirs("./figs/",exist_ok=True)

from methods import *
from matplotlib.colors import LogNorm
L = int(sys.argv[1])


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

def extract(L,r_ex,b_1,b_2,a1,a2,dk,X_U1,X_U2,X_U3) :
    
    if True :
        
        r_a1 = np.array([(1/(2*np.pi))*np.dot(r_ex,b_1)])
        r_a2 = np.array([(1/(2*np.pi))*np.dot(r_ex,b_2)])
        
        p_space1_pt = np.exp(1j*(r_a1*dk))   # np.exp(1j*dk1*xmesh) #

        p_space2_pt = np.exp(1j*(r_a2*dk))   # np.exp(1j*dk2*ymesh) #

        p_space3_pt = np.conj(p_space1_pt)*np.conj(p_space2_pt)

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

        ws_loc,vs_loc = np.linalg.eigh(obj.locs[0])
        


    mu = ws_loc[len(ws_loc)//2]

    



    return mu

def scan(L,scan_res) : 
    
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

    
    for ix in range(scan_res) : 
        for iy in range(scan_res) : 
            
            print(ix,iy)
            r_ex = np.array([xsmesh[iy,ix],ysmesh[iy,ix]])
            gaps[iy,ix] = extract(L,r_ex,b_1,b_2,a1,a2,dk,X_U1,X_U2,X_U3)
            
    (Us_WC, Ss_WC, Vs_WC), p_spaces_WC, gaps_WC = extract_and_schmidt(L,r_WC,b_1,b_2,a1,a2,dk,X_U1,X_U2,X_U3)
    
    
    folder = "./h5_files/"
    os.makedirs(folder,exist_ok=True) 
            
    fname = folder +  f"L={L}_scan_res={scan_res}.h5"
    

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
    
    hdf.save_numpy_to_hdf5(fname,np.array([p_spaces_WC]),"/p_spaces_WC")
    
    plt.imshow(gaps,origin="lower",cmap="magma_r",norm=LogNorm())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set(xticks=[],yticks=[])
    plt.colorbar()
    plt.savefig(f"./figs/main_text_figure_1_b_LIF_N={L}_scan_res={scan_res}.png")
    


for scan_res in [21] : 

    scan(L,scan_res)

