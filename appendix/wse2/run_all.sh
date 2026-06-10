L=12 # 24 used in main text
scan_res=21 

python3 calculate.py $L $scan_res || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 make_figure.py $L $scan_res || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }