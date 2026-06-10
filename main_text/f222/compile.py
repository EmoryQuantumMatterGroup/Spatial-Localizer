import sys
sys.path.insert(0,"./../../")

import toolkit_local.hdf5 as hdf


kres = int(sys.argv[1])
scan_res = int(sys.argv[2])
phase = str(sys.argv[3])

fname_save = f"./h5_files_new/prep_phase={phase}_kres={kres}_scan_res={scan_res}.h5"

face_gaps = {}

for y_index in range(scan_res) : 
    
    if y_index == 0 : 
        
        for face_index in range(6) : 
            fname_tmp = f"./h5_files_new/uncompiled/prep_t={phase}_kres={kres}_scan_res={scan_res}_face={face_index}_y_ind={y_index}.h5"
            
            face_gaps[f"{face_index}"] = hdf.load_hdf5_to_numpy(fname_tmp,"gaps_lower_line")
            
    elif y_index < scan_res//2 : 
        
        for face_index in range(6) : 
            fname_tmp = f"./h5_files_new/uncompiled/prep_t={phase}_kres={kres}_scan_res={scan_res}_face={face_index}_y_ind={y_index}.h5"
            
            face_gaps[f"{face_index}"] += hdf.load_hdf5_to_numpy(fname_tmp,"gaps_lower_line")
            
    else : 
        
        for face_index in range(3) : 
            fname_tmp = f"./h5_files_new/uncompiled/prep_t={phase}_kres={kres}_scan_res={scan_res}_face={face_index}_y_ind={y_index}.h5"
            
            face_gaps[f"{face_index}"] += hdf.load_hdf5_to_numpy(fname_tmp,"gaps_lower_line")
            
hdf.save_dict_to_hdf5(face_gaps,fname_save)