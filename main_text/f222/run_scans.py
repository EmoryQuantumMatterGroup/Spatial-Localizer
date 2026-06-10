import sys


from scan_parallel import scan_parallel_fn

print("Running all scans")

kres = int(sys.argv[1])
scan_res = int(sys.argv[2])
phase = str(sys.argv[3])

ys_indices_all = [range(scan_res) if ii < 3 else range(scan_res//2) for ii in range(6)]


for face_index in range(6) : 
    
    print(f"current face : {face_index}")

    for y_index in ys_indices_all[face_index] : 
        print(f"y_index={y_index}")
        
        scan_parallel_fn(kres,scan_res,phase,face_index,y_index)
        

            