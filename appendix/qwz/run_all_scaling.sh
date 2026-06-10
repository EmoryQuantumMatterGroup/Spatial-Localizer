


python3 calculate_scaling.py || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
python3 fig_making_scaling.py || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }

