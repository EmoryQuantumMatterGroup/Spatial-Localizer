kres_scan=12 # 24 used in manuscript
kres_state=24 # 24 used in manuscript

python3 scan.py $kres_scan || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 plot_state.py $kres_state || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }

