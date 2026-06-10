from methods import *

os.makedirs("./h5_files/",exist_ok=True)
os.makedirs("./figs/",exist_ok=True)

L = int(sys.argv[1])
scan_res=int(sys.argv[2])

for m in [1,3] : 
    
    build_and_save(L,scan_res,m)
    
    calc_bound_elements_p1_parallel_trivial(L,scan_res,m)


