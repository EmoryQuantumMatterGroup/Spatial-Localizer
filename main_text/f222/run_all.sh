kres=4 # 6 used in manuscript
scan_res=11 # 101 used in the manuscript
phase1="triv"
phase2="topo"

python3 prep.py $kres $scan_res $phase1 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 run_scans.py $kres $scan_res $phase1 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 compile.py $kres $scan_res $phase1 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }

python3 prep.py $kres $scan_res $phase2 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; } 
python3 run_scans.py $kres $scan_res $phase2 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 compile.py $kres $scan_res $phase2 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }



mkdir figs/

python3 make_lattice_figure_pyvista.py || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 make_lattice_figure_pyvista_matching.py || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }

python3 plot_LIF_pyvista.py $kres $scan_res $phase1 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; } 
python3 plot_LIF_pyvista.py $kres $scan_res $phase2 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
