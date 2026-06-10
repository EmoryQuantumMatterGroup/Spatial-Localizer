from hams import * 
import toolkit_local.plotting as pp
import toolkit_local.localizer_general as lg
import toolkit_local.hdf5 as hdf
import toolkit_local.cond_mat as cm

my_dict = {}
my_dict["params_dict"] = {}
params_dict = my_dict["params_dict"]

kres = int(sys.argv[1])
scan_res = int(sys.argv[2])


phase=str(sys.argv[3])
if phase == "topo" : 
    t = 1.5 
else : 
    phase = "triv" 
    t = 0.5

params_dict["kres"] = kres
# t=1.5

params_dict["t"] = t
delta=0.05
params_dict["delta"] = delta
# scan_res = 40
scan_res_small = scan_res//2

params_dict["scan_res"] = scan_res
params_dict["scan_res_small"] = scan_res_small

scan_range = (0,1)

(a1,a2,a3), (b1,b2,b3) = f222_lat_vecs()

my_dict["lat_vecs_direct"] = np.stack([a1,a2,a3])
my_dict["lat_vecs_reciprocal"] = np.stack([b1,b2,b3]) 

ks_line = np.linspace(0,1,kres+1)[:-1]


b1s_mesh,b2s_mesh,b3s_mesh = np.meshgrid(ks_line,ks_line,ks_line,indexing="ij")


ks = np.tensordot(b1s_mesh,b1,axes=0) + np.tensordot(b2s_mesh,b2,axes=0) + np.tensordot(b3s_mesh,b3,axes=0)
my_dict["ks"] = ks

orb_dim = 2
# Es_ham = np.zeros((kres,kres,kres,orb_dim))
# vs_ham = np.zeros((kres,kres,kres,orb_dim,orb_dim),dtype=complex)

V_lower = np.zeros((kres,kres,kres,orb_dim,kres**3),dtype=complex)



state_counter=0
for ix in range(kres) : 
    for iy in range(kres) : 
        for iz in range(kres) :
            
            
            _, vs_tmp = np.linalg.eigh(f222_rice_mele_ham(ks[ix,iy,iz],t=t,delta=delta))
            
            V_lower[ix,iy,iz,:,state_counter] = vs_tmp[:,0]

            
            state_counter += 1
            
            
V_lower = V_lower.reshape(orb_dim*kres**3,kres**3)

my_dict["V_lower"] = V_lower

            
hop = np.eye(kres,k=-1,dtype=complex)
hop[0,-1] = 1 

eye_resta = np.eye(kres,dtype=complex)


X1_R = np.kron(np.kron(hop,eye_resta),eye_resta)
X2_R = np.kron(np.kron(eye_resta,hop),eye_resta)
X3_R = np.kron(np.kron(eye_resta,eye_resta),hop)


X1_R,X2_R,X3_R = np.kron(X1_R,np.eye(2)),np.kron(X2_R,np.eye(2)),np.kron(X3_R,np.eye(2))

X4_R = (X1_R @ X2_R @ X3_R ).conj().T


my_dict["ops_dict"] = {}

ops_dict = my_dict["ops_dict"]


ops_dict["X1_R"] = X1_R
ops_dict["X2_R"] = X2_R
ops_dict["X3_R"] = X3_R
ops_dict["X4_R"] = X4_R
            
enforce_unt = True
if enforce_unt :

    X1_V, X2_V, X3_V = mm.enforce_unitarity(mm.project(X1_R,V_lower),print_metrics=True),mm.enforce_unitarity(mm.project(X2_R,V_lower)),mm.enforce_unitarity(mm.project(X3_R,V_lower))
    X4_V = mm.enforce_unitarity(mm.project(X4_R,V_lower),print_metrics=True)

    # X1_Vupper,X2_Vupper,X3_Vupper = mm.enforce_unitarity(mm.project(X1_R,V_upper)),mm.enforce_unitarity(mm.project(X2_R,V_upper)),mm.enforce_unitarity(mm.project(X3_R,V_upper))
    # X4_Vupper = mm.enforce_unitarity(mm.project(X4_R,V_upper),print_metrics=True)
    
else : 
    X1_V, X2_V, X3_V = (mm.project(X1_R,V_lower)),(mm.project(X2_R,V_lower)),(mm.project(X3_R,V_lower))
    X4_V = (mm.project(X4_R,V_lower))

    # X1_Vupper,X2_Vupper,X3_Vupper = (mm.project(X1_R,V_upper)),(mm.project(X2_R,V_upper)),(mm.project(X3_R,V_upper))
    # X4_Vupper = (mm.project(X4_R,V_upper))



ops_dict["X1_V"] = X1_V
ops_dict["X2_V"] = X2_V
ops_dict["X3_V"] = X3_V
ops_dict["X4_V"] = X4_V

# ops_dict["X1_Vupper"] = X1_Vupper
# ops_dict["X2_Vupper"] = X2_Vupper
# ops_dict["X3_Vupper"] = X3_Vupper
# ops_dict["X4_Vupper"] = X4_Vupper


my_dict["scan_meshes_dict"] = {f"{ii}" : {} for ii in range(6)}

scan_meshes_dict = my_dict["scan_meshes_dict"]




faces_shifts =   [np.zeros(3),a1+a3,a2+a3,(a1+a2+a3)*0.5,(a1+a2+a3)*0.5,(a1+a2+a3)*0.5]
faces_spanners = [[a1,a2],[a2,-a3],[a1,-a3],[0.5*a1,-0.5*a3],[0.5*a1,0.5*a2],[0.5*a2,-0.5*a3]]

# scan_meshes_dict["face_spanners"] = {} # add dict to save spanners and shifts


for ii in range(6) :  # continue here
    
    if ii < 3 : 
        scan_res_tmp = scan_res
    else : 
        scan_res_tmp = scan_res_small
    
    xs_line = np.linspace(*scan_range,scan_res_tmp)

    xs_mesh, ys_mesh = np.meshgrid(xs_line,xs_line,indexing="ij")
    ones_mesh = np.ones_like(xs_mesh)
    
    tmp_dict = scan_meshes_dict[f"{ii}"]
    
    span1_tmp = faces_spanners[ii][0]
    span2_tmp = faces_spanners[ii][1]
    shift_tmp = faces_shifts[ii]
    

    rs_scan = np.tensordot(xs_mesh,span1_tmp,axes=0) + np.tensordot(ys_mesh,span2_tmp,axes=0) + np.tensordot(ones_mesh,shift_tmp,axes=0)

    ps1 = np.exp((1j/kres)*np.tensordot(rs_scan,b1,axes=(-1,0)))
    ps2 = np.exp((1j/kres)*np.tensordot(rs_scan,b2,axes=(-1,0)))
    ps3 = np.exp((1j/kres)*np.tensordot(rs_scan,b3,axes=(-1,0)))


    tmp_dict["ps1"] = ps1
    tmp_dict["ps2"] = ps2
    tmp_dict["ps3"] = ps3
    tmp_dict["ps4"] = (ps1*ps2*ps3).conj()
    tmp_dict["xs_mesh"] = xs_mesh
    tmp_dict["ys_mesh"] = ys_mesh
    tmp_dict["rs_scan"] = rs_scan


folder = f"./h5_files_new/"
fname = f"prep_phase={phase}_kres={kres}_scan_res={scan_res}.h5"

os.makedirs(folder,exist_ok=True)

hdf.save_dict_to_hdf5(my_dict,folder + fname,"prep/")


