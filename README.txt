This repository corresponds to minimal code to reproduce the main text figures and key appendix figures of "A spatial localizer for electrons in insulators" by Gerhard, H.; Wang, Y.; Cerjan, A.; and Benalcazar, W.

This code has been tested on Ubuntu 22.04, and is written to be compatible with UNIX based systems (e.g., slashes in file paths are /, not \ as one has in Windows). For Windows users, we note the existence of Ubuntu on WSL (https://ubuntu.com/wsl), which might allow one to run the code directly using the provided .sh scripts without any OS-related refactoring, such as the backslash convention.

We note that we have adapted the code here to run on a single machine in series. Originally, much of this code was split apart to run as parallel jobs on a compute cluster since many things can be parallelized.


## Running the code
The code here is set up to run on bash-like terminals by the command '. ./run_everything.sh' from the root repo directory. run_everything.sh cds into each folder and produces the figures from scratch. At the end, run_everything.sh will copy all pngs present in the repo to the folder ./all_figs in the root directory.

For the more memory/computationally consuming parts, we have set the default parameters to be smaller/coarser than used for publication. These parameters can be changed in the corresponding "run_all*.sh" files in each sub-directory, where we also denote the parameters used in the manuscript for ease of use as comments. For many of these .sh scripts, we have scan resolution (scan_res) as a parameter. For the main text figures, we use scanning resolutions of either 101, 201, or 301. Setting scan_res to 101 should suffice if one desires a higher resolution for the sake of understanding.

A test run of run_everything.sh produces the output :
"""Running main text calculations
    Running Figure 1 simulations
    Running Figure 2 simulations
    Running Figure 3 simulations
    Running Figure 4 simulations
Running selected appendix calculations
    Running disclination-related appendix calculations
    Running QWZ-related appendix calculations
    Running WSe2-related appendix calculations
Copying figures to root folder
"""

Furthermore, in each directory, text files of the form *.out contain the outputs of the corresponding python scripts used.


## Python (Anaconda) Environment
A build file for the conda environment used in testing is provided at environment.yml. To build the environment provided in environment.yml and run the provided code, use the below commands in the root of this repo.
'''
conda env create -f environment.yml
conda activate spat_loc
. ./run_everything.sh
'''

## HDF5 
Besides Python and the provided conda environment, we note that we use the HDF5 file system software to store/access data, such as dictionaries or Numpy arrays. 
This is accessed via the h5py Python package, but HDF5 generally must be separately installed on the machine (see https://www.hdfgroup.org/download-hdf5/). 