scan_res=11 # 301 used in manuscript

python3 disc_calculate.py 11 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 make_figure_disc.py 11 || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }