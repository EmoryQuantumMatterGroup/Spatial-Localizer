This repository corresponds to minimal code to reproduce the main text figures and key appendix figures of "A spatial localizer for electrons in insulators" by Gerhard, H.; Wang, Y.; Cerjan, A.; and Benalcazar, W.

This code has been tested on Ubuntu 22.04 and MacOS 26.5, and is written to be compatible with UNIX based operating systems (e.g., slashes in file paths are /, not \ as one has in Windows). For Windows users, Ubuntu on WSL (https://ubuntu.com/wsl) may allow one to run the code directly using the provided .sh scripts without any OS-related refactoring, such as the backslash convention in file paths.

We note that we have adapted the code here to run on a single machine in series. Originally, much of this code was split apart to run as parallel jobs on a compute cluster.


## Running the code
The code here is set up to run on bash-like shells/terminals by the command '. ./run_everything.sh' from the root directory. run_everything.sh moves into each folder and produces the figures from scratch, taking ~10 minutes to run on the local machine used for testing. At the end, run_everything.sh will copy all pngs present in the repo to the folder ./all_figs in the root directory. We provide an example output in "all_figs_example_output.zip." Specifically, the provided code generates the numeric figures for the main text figures 1-4, as well as appendix figures 1,2,4, and 11-14 as numerical evidence for the central claims of this work.

For the more memory/computationally consuming computations, we have set the default parameters to be smaller/coarser than those used for publication. These parameters can be changed in the corresponding "run_all*.sh" files in each sub-directory, where we also denote the parameters used in the manuscript for ease of use as comments. For many of these .sh scripts, we have scan resolution (scan_res) as a parameter, or clearly visible in the corresponding .py. For the main text figures, we use scanning resolutions of either 101, 201, or 301. Setting scan_res to 101 should suffice if one desires a higher resolution for the sake of clarification/confirmation.

A typical usage of run_everything.sh produces the terminal output :
```
Running main text calculations
    Running Figure 1 simulations
    Running Figure 2 simulations
    Running Figure 3 simulations
    Running Figure 4 simulations
Running selected appendix calculations
    Running disclination-related appendix calculations
    Running QWZ-related appendix calculations
    Running WSe2-related appendix calculations
Copying figures to root folder
```

Furthermore, in each subdirectory, text files of the form *.out contain the outputs of the corresponding python scripts used.


## Python (Anaconda) Environment
A build file for the conda environment used in testing is provided at environment.yml for Linux OS and environment_general_OS.yml for generic OS. We note that the environment_general_OS.yml makes no assumption of particular numerical backends (e.g., Intel MKL vs OpenBLAS for implementing LAPACK/BLAS libraries), so runtimes may vary when using environment_general_OS.yml. To build a environment and run the provided code, use the below commands in the root of this repo.
```
conda env create -f environment.yml
conda activate spat_loc
. ./run_everything.sh
```


## HDF5 
Besides Python and the provided conda environment, we note that we use the HDF5 file system software to store/access data, such as dictionaries or Numpy arrays. 
This is accessed via the h5py Python package, but HDF5 often must be separately installed on the machine (see https://www.hdfgroup.org/download-hdf5/). The test machine is using HDF5 version 1.14.5.



## Runtime information
Installing the Conda environment and HDF5 should take ~5 minutes. Running the code itself takes ~30 minutes.


