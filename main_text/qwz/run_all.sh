kres_scan=4 # 24 is used in the manuscript

python3 scan.py $kres_scan 1 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 scan.py $kres_scan 3 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }



kres_states=30 # 50 is used in the manuscript

python3 make_states.py $kres_states 1 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 make_states.py $kres_states 3 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }

python3 plot_bcs.py $kres_states || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 plot_decays.py $kres_states || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
