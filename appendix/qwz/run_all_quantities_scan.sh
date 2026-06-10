L=12 # 24 used in the manuscript
scan_res=21 # multiples of 4 plus one will hit all relevant locations exactly (origins and boundaries of unit cells)


python3 calculate.py $L $scan_res || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 fig_making.py $L $scan_res || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }


