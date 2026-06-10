


def scan_parallel_fn(kres,scan_res,phase,face_index,y_index,if_sparse = False) : 
    import numpy as np
    import scipy as sp
    import sys
    import os
    os.makedirs("./h5_files_new/uncompiled/",exist_ok=True)

    sys.path.insert(0,"./../../")

    from toolkit_local import matrices as mm
    import toolkit_local.localizer_general as lg
    import toolkit_local.hdf5 as hdf



    if_print_progress = True

    load_fname = f"./h5_files_new/prep_phase={phase}_kres={kres}_scan_res={scan_res}.h5"
    save_fname = f"./h5_files_new/uncompiled/prep_t={phase}_kres={kres}_scan_res={scan_res}_face={face_index}_y_ind={y_index}.h5"

    if face_index < 0 or face_index > 5:
        raise ValueError("face_index must be between 0 and 5")


    ops = [hdf.load_hdf5_to_numpy(load_fname,f"prep/ops_dict/X{ii+1}_V") for ii in range(4)]
    cliffs = mm.cliff_generators(4)


    pspaces_all = [hdf.load_hdf5_to_numpy(load_fname,f"prep/scan_meshes_dict/{face_index}/ps{ii+1}") for ii in range(4)]

    x_res, y_res = pspaces_all[0].shape

    if y_index < 0 or y_index >= y_res:
        raise ValueError(f"y_index must be between 0 and {y_res-1} for face {face_index}")

    tmp_gaps = np.zeros((x_res,y_res))

    if if_print_progress : print(f"Scanning started")

    def sparse_gap(loc_tmp, x_index):
        # print(np.linalg.norm(mm.commutator(loc_tmp,loc_tmp.conj().T)))
        # loc_tmp = 0.5*(loc_tmp + loc_tmp.conj().T) 
        loc_sparse = sp.sparse.csr_array(loc_tmp)
        print(loc_sparse.nnz/(loc_tmp.shape[0]*loc_tmp.shape[1]))
        try:
            ws_tmp = sp.sparse.linalg.eigsh(loc_sparse,k=1,sigma=1e-8,which="LM",return_eigenvectors=False)
            return np.abs(ws_tmp[0])
        except Exception:
            if if_print_progress : print(f"shift-invert failed at face {face_index}, y {y_index}, x {x_index}; trying which='SM'")

            try:
                ws_tmp = sp.sparse.linalg.eigsh(loc_sparse,k=1,which="SM",return_eigenvectors=False)
                return np.abs(ws_tmp[0])
            except Exception:
                if if_print_progress : print(f"sparse diagonalization failed at face {face_index}, y {y_index}, x {x_index}; falling back to dense")

                ws_tmp = np.abs(np.linalg.eigvalsh(loc_tmp))
                return np.min(ws_tmp,axis=-1)

    if if_sparse : 
        
        for x_index in range(x_res) :
            if if_print_progress : print(f"face : {face_index} | y : {y_index} | x : {x_index}")
            
            pspaces = [np.array([pspaces_all[ii][x_index,y_index]]) for ii in range(4)]
            
            loc_obj_tmp = lg.Localizer(ops,pspaces,cliffs)
            
            loc_obj_tmp.build_locs()
            
            loc_tmp = loc_obj_tmp.locs[0]

            
            tmp_gaps[x_index,y_index] = sparse_gap(loc_tmp,x_index)
    
    else :
        for x_index in range(x_res) :
            if if_print_progress : print(f"face : {face_index} | y : {y_index} | x : {x_index}")
            
            pspaces = [np.array([pspaces_all[ii][x_index,y_index]]) for ii in range(4)]
            
            loc_obj_tmp = lg.Localizer(ops,pspaces,cliffs)
            
            loc_obj_tmp.build_locs()
            
            ws_tmp = np.abs(np.linalg.eigvalsh(loc_obj_tmp.locs))

            
            tmp_gaps[x_index,y_index] = np.min(ws_tmp,axis=-1)[0]

                
      
        

    hdf.save_numpy_to_hdf5(save_fname,tmp_gaps,"gaps_lower_line")
