from methods import *


def calc_scaling_elements_p1_parallel_local(L,m,speed='fast') : 
    
    
    fname = f"./h5_files_scaling/qwz_scaling_info_L={L}_m={m}.h5"
    
    Us_all = hdf.load_hdf5_to_numpy(fname,"/Us_all")
    Us_WC = hdf.load_hdf5_to_numpy(fname,"/Us_WC")
    # X_U3 = hdf.load_hdf5_to_numpy(fname,"/X_U3")
    gaps = hdf.load_hdf5_to_numpy(fname,"/gaps")
    xsmesh = hdf.load_hdf5_to_numpy(fname,"/xsmesh")
    ysmesh = hdf.load_hdf5_to_numpy(fname,"/ysmesh")
    four_T = hdf.load_hdf5_to_numpy(fname,"/four_T")
    U = hdf.load_hdf5_to_numpy(fname,"/U")
    X_rspace = hdf.load_hdf5_to_numpy(fname,"/X_rspace")
    Y_rspace = hdf.load_hdf5_to_numpy(fname,"/Y_rspace")
    

    print(gaps.shape)
    

    # variances_rspace_p1 = np.zeros((scan_res,scan_res))
    # variances_occupied_p1 = np.zeros((scan_res,scan_res))
    # variances_wf = np.zeros((scan_res,scan_res))
    # variances_wf_occupied = np.zeros((scan_res,scan_res))
    # coherence_cartesian_wf = np.zeros((scan_res,scan_res))
    # coherence_cartesian_p1 = np.zeros((scan_res,scan_res))
    deviations_rspace_p1 = np.zeros((2))
    # deviations_rspace_wf = np.zeros((scan_res,scan_res))
    # deviations_rspace_p1_v2 = np.zeros((scan_res,scan_res))
    
    # deviations_from_WC_WF = np.zeros((scan_res,scan_res))
    # deviations_from_WC_p1 = np.zeros((scan_res,scan_res))
    
    P_rspace = (four_T @ U ) @ (four_T @ U ).conj().T
    
    U_rspace = four_T @ U
    X_P = P_rspace @ X_rspace @ P_rspace
    Y_P = P_rspace @ Y_rspace @ P_rspace
    
    Z = X_P + Y_P*1j
    Z_dag = X_P + Y_P*1j
    
    comm_cart = mm.commutator(X_P,Y_P)
    
    print("norm")
    print(np.linalg.norm(comm_cart,ord=2))
    
    p1_WC = Us_WC[:,0]
    wf_WC = mm.normalize(p1_WC.copy()/np.abs(p1_WC))
    
    p1_WC_rspace = four_T @ U @ p1_WC
    wf_WC_rspace = four_T @ U @ wf_WC
    
    p1_WC_meanx, p1_WC_meany = mm.exp_val(X_rspace,p1_WC_rspace).real, mm.exp_val(Y_rspace,p1_WC_rspace).real
    wf_WC_meanx, wf_WC_meany = mm.exp_val(X_rspace,wf_WC_rspace).real, mm.exp_val(Y_rspace,wf_WC_rspace).real
    p1_WC_var   = mm.variance(X_rspace,p1_WC_rspace) + mm.variance(Y_rspace,p1_WC_rspace)
    wf_WC_var   = mm.variance(X_rspace,wf_WC_rspace) + mm.variance(Y_rspace,wf_WC_rspace)
    
    print(f"p1 mean : {p1_WC_meanx,p1_WC_meany}")
    
    # variances_occupied_p1 = hdf.load_hdf5_to_numpy(fname,"variances_occupied")
    # variances_wf_occupied = hdf.load_hdf5_to_numpy(fname,"variances_wf_occupied")
    
    if False : 
        for ix in range(scan_res) : 
            for iy in range(scan_res) :  
                
                
                # Us = Us_all[iy,ix]
                
                p1 = Us_all[iy,ix,:,0]
    

                p1_rspace = (four_T @ U @ p1) #np.roll((four_T @ U @ p1).reshape(L,L,3),axis=(0,1),shift=(L//2,L//2)).reshape(3*L**2)
                
                meanx_p1 = mm.exp_val(X_rspace,p1_rspace).real
                meany_p1 = mm.exp_val(Y_rspace,p1_rspace).real
                
                r_x = xsmesh[iy,ix]
                r_y = ysmesh[iy,ix]
                
                deviations_rspace_p1[iy,ix] = np.sqrt((meanx_p1 - r_x)**2 + (meany_p1 - r_y)**2)
                
                
                
                # print(p1_rspace.shape,X_rspace.shape)
                # variances_rspace_p1[iy,ix] = mm.variance(X_rspace,p1_rspace) + mm.variance(Y_rspace,p1_rspace)
                # variances_occupied_p1[iy,ix] = mm.variance(X_P,p1_rspace) + mm.variance(Y_P,p1_rspace)
                
                # variances_wf[iy,ix] = mm.variance(X_rspace,wf_rspace) + mm.variance(Y_rspace,wf_rspace)
                # variances_wf_occupied[iy,ix] = mm.variance(X_P,wf_rspace) + mm.variance(Y_P,wf_rspace)

                # coherence_cartesian_p1[iy,ix] = variances_occupied_p1[iy,ix] - np.abs(np.vdot(p1_rspace, comm_cart @ p1_rspace))
                
                # deviations_rspace_p1_v2[iy,ix] = np.sqrt((meanx_p1 - (24 - r_x))**2 + (meany_p1 - (24 - r_y))**2)
                # deviations_rspace_wf[iy,ix] = np.sqrt((meanx_p1 - r_x)**2 + (meany_p1 - r_y)**2)
                # deviations_from_WC_p1[iy,ix] = np.sqrt((meanx_p1 - p1_WC_meanx)**2 + (meany_p1 - p1_WC_meany)**2)

            
    
    if True :  
        
        p1s_all = Us_all[...,0]
      
        wfs_all = p1s_all/np.abs(p1s_all)
        
        wfs_all = parallel_normalization(parallel_mmult(wfs_all,U_rspace))
        
        
        
        
        
        p1s_all = parallel_mmult(p1s_all,U_rspace)
        
        norms_Z = parallel_mmult(p1s_all,Z)
        norms_Z_dag = parallel_mmult(p1s_all,Z_dag)
        
        
        
        mean_xs_all = parallel_exp_val(p1s_all, X_rspace ).real
        
        mean_ys_all = parallel_exp_val(p1s_all, Y_rspace ).real
        
        
        mean_x2s_all = parallel_exp_val(p1s_all, X_rspace @ X_rspace ).real
        
        mean_y2s_all = parallel_exp_val(p1s_all, Y_rspace @ Y_rspace ).real
        
        mean_x2Ps_all = parallel_exp_val(p1s_all, X_P @ X_P).real
        
        mean_y2Ps_all = parallel_exp_val(p1s_all, Y_P @ Y_P).real
        
        comm_exp_vals = parallel_exp_val(p1s_all, comm_cart)
        
        
        
        var_occ_x = (mean_x2Ps_all - mean_xs_all**2)
        var_occ_y = (mean_y2Ps_all - mean_ys_all**2)
        
        
        deviations_from_extract = np.sqrt((mean_xs_all - xsmesh)**2 + (mean_ys_all - ysmesh)**2)
        
        deviations_from_WC = np.sqrt((mean_xs_all-12)**2 + (mean_ys_all-12)**2)
        
        var_x = (mean_x2s_all - mean_xs_all**2) 
        var_y = (mean_y2s_all - mean_ys_all**2)
        
        coherence = np.sqrt(var_occ_x + var_occ_y) - np.sqrt(np.abs(comm_exp_vals))
        

        
        wfs_mean_xs_all = parallel_exp_val(wfs_all, X_rspace ).real
        
        wfs_mean_ys_all = parallel_exp_val(wfs_all, Y_rspace ).real
        
        
        wfs_mean_x2s_all = parallel_exp_val(wfs_all, X_rspace @ X_rspace ).real
        
        wfs_mean_y2s_all = parallel_exp_val(wfs_all, Y_rspace @ Y_rspace ).real
        
        wfs_mean_x2Ps_all = parallel_exp_val(wfs_all, X_P @ X_P).real
        
        wfs_mean_y2Ps_all = parallel_exp_val(wfs_all, Y_P @ Y_P).real
        
        wfs_comm_exp_vals = parallel_exp_val(wfs_all, comm_cart)
        
        
        
        wfs_var_occ_x = (wfs_mean_x2Ps_all - wfs_mean_xs_all**2) 
        wfs_var_occ_y = (wfs_mean_y2Ps_all - wfs_mean_ys_all**2)
        
        
        wfs_deviations_from_extract = np.sqrt((wfs_mean_xs_all - xsmesh)**2 + (wfs_mean_ys_all - ysmesh)**2)
        
        wfs_deviations_from_WC = np.sqrt((wfs_mean_xs_all-12)**2 + (wfs_mean_ys_all-12)**2)
        
        wfs_var_x = (wfs_mean_x2s_all - wfs_mean_xs_all**2)
        wfs_var_y = (wfs_mean_y2s_all - wfs_mean_ys_all**2)
        
        wfs_coherence = np.sqrt(wfs_var_occ_x + wfs_var_occ_y) - np.sqrt(np.abs(wfs_comm_exp_vals))
        var_both = np.sqrt(var_x + var_y)
        
        diff_gaps        = gaps[1] - gaps[0] 
        diff_gaps_scaled = (gaps[1] - gaps[0])*L 
        diff_coherence   = coherence[1] - coherence[0]
        diff_variance    = var_both[1] - var_both[0]
        
        
        
        
        

    
    fname_new = f"./h5_files_scaling/qwz_scaling_info_L={L}_m={m}.h5"
    hdf.save_numpy_to_hdf5(fname_new,np.array([diff_gaps       ]),"/diff_gaps")
    hdf.save_numpy_to_hdf5(fname_new,np.array([diff_gaps_scaled]),"/diff_gaps_scaled")
    hdf.save_numpy_to_hdf5(fname_new,np.array([diff_coherence  ]),"/diff_coherence")
    hdf.save_numpy_to_hdf5(fname_new,np.array([diff_variance   ]),"/diff_variance")
    
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_occ_x,"/wfs_var_occ_x")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_occ_y,"/wfs_var_occ_y")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_occ_x + wfs_var_occ_y,"/wfs_var_occ_both")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_occ_x - wfs_var_occ_y,"/wfs_var_occ_diff")
    hdf.save_numpy_to_hdf5(fname_new,(wfs_var_occ_x - wfs_var_occ_y).__abs__(),"/wfs_var_occ_diff_abs")
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_deviations_from_extract,"/wfs_deviations_from_extract")
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_deviations_from_WC,"/wfs_deviations_from_WC")
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_x,"/wfs_var_x")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_y,"/wfs_var_y")
    hdf.save_numpy_to_hdf5(fname_new,np.sqrt(wfs_var_x + wfs_var_y),"/wfs_var_both")
    hdf.save_numpy_to_hdf5(fname_new,wfs_var_x - wfs_var_y,"/wfs_var_diff")
    hdf.save_numpy_to_hdf5(fname_new,(wfs_var_x - wfs_var_y).__abs__(),"/wfs_var_diff_abs")
    
    hdf.save_numpy_to_hdf5(fname_new,wfs_coherence,"/wfs_coherence")
    hdf.save_numpy_to_hdf5(fname_new,wfs_comm_exp_vals,"/wfs_comm_exp_vals")
        
    hdf.save_numpy_to_hdf5(fname_new,np.array([p1_WC_meanx]),"p1_WC_meanx")
    hdf.save_numpy_to_hdf5(fname_new,np.array([wf_WC_meanx]),"wf_WC_meanx") 
    hdf.save_numpy_to_hdf5(fname_new,np.array([p1_WC_meany]),"p1_WC_meany")
    hdf.save_numpy_to_hdf5(fname_new,np.array([wf_WC_meany]),"wf_WC_meany")    
    hdf.save_numpy_to_hdf5(fname_new,np.array([p1_WC_var  ]),"WC_var")    
    hdf.save_numpy_to_hdf5(fname_new,np.array([wf_WC_var  ]),"wfs_WC_var")    
        
        
    
    # fname_new = fname[:-3] + "_NEW2_p1_data.h5"
    
    hdf.save_numpy_to_hdf5(fname_new,var_occ_x,"/var_occ_x")
    hdf.save_numpy_to_hdf5(fname_new,var_occ_y,"/var_occ_y")
    hdf.save_numpy_to_hdf5(fname_new,var_occ_x + var_occ_y,"/var_occ_both")
    hdf.save_numpy_to_hdf5(fname_new,var_occ_x - var_occ_y,"/var_occ_diff")
    hdf.save_numpy_to_hdf5(fname_new,(var_occ_x - var_occ_y).__abs__(),"/var_occ_diff_abs")
    
    hdf.save_numpy_to_hdf5(fname_new,deviations_from_extract,"/deviations_from_extract")
    
    hdf.save_numpy_to_hdf5(fname_new,deviations_from_WC,"/deviations_from_WC")
    
    hdf.save_numpy_to_hdf5(fname_new,var_x,"/var_x")
    hdf.save_numpy_to_hdf5(fname_new,var_y,"/var_y")
    hdf.save_numpy_to_hdf5(fname_new,np.sqrt(var_x + var_y),"/var_both")
    hdf.save_numpy_to_hdf5(fname_new,var_x - var_y,"/var_diff")
    hdf.save_numpy_to_hdf5(fname_new,(var_x - var_y).__abs__(),"/var_diff_abs")
    
    hdf.save_numpy_to_hdf5(fname_new,coherence,"/coherence")
    hdf.save_numpy_to_hdf5(fname_new,comm_exp_vals,"/comm_exp_vals")
    
    

    # # hdf.save_numpy_to_hdf5(fname_new,variances_rspace_p1,"/variances_rspace_p1")
    # hdf.save_numpy_to_hdf5(fname_new,variances_occupied_p1,"/variances_occupied_p1")
    # # hdf.save_numpy_to_hdf5(fname_new,variances_wf,"/variances_wf")
    # hdf.save_numpy_to_hdf5(fname_new,variances_wf_occupied,"/variances_wf_occupied")
    # hdf.save_numpy_to_hdf5(fname_new,coherence_cartesian_wf,"/coherence_cartesian_wf")
    # hdf.save_numpy_to_hdf5(fname_new,coherence_cartesian_p1,"/coherence_cartesian_p1")

    # hdf.save_numpy_to_hdf5(fname_new,deviations_rspace_p1,"/deviations_rspace_p1")
    # # hdf.save_numpy_to_hdf5(fname_new,deviations_rspace_p1_v2,"/deviations_rspace_p1_v2")
    # hdf.save_numpy_to_hdf5(fname_new,deviations_from_WC_p1,"/deviations_from_WC_p1")
    # hdf.save_numpy_to_hdf5(fname_new,deviations_from_WC_WF,"/deviations_from_WC_WF")
    # hdf.save_numpy_to_hdf5(fname_new,variances_wf_occupied - mm.variance(X_P,wf_WC_rspace) - mm.variance(Y_P,wf_WC_rspace),"/deviations_variance_from_WC_WF")
    
    # print(mm.variance(X_P,wf_WC_rspace) + mm.variance(Y_P,wf_WC_rspace))


os.makedirs("./h5_files_scaling/",exist_ok=True)
os.makedirs("./figs/",exist_ok=True)

Ls = [16,17,18,19,20,21,22,23,24,25,26,27,28]#,32,36,40]


for L in Ls : 
    
    print(L)
    
    for m in [1,3] :
        build_and_save_scaling(L,m)
        
        calc_scaling_elements_p1_parallel_local(L,m)

