scanres=101 # increase this to see a refined picture of the LIF. ~201 produces a resolution comprable to the manuscript figure

python3 scan.py $scanres triv || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }

python3 scan.py $scanres topo || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }

python3 fig_gen.py $scanres || { echo "Critical Error: Python script failed! (Location: $PWD)"; return 1; }
