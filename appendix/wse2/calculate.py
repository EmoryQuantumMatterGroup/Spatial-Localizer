from methods import *
os.makedirs("./figs/",exist_ok=True)
os.makedirs("./h5_files/",exist_ok=True)


L = int(sys.argv[1])
scan_res=int(sys.argv[2])
    
build_and_save(L,scan_res)

calc_bound_elements(L,scan_res)