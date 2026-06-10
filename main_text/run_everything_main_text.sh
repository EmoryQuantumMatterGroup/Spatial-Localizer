qwz_folder="./qwz"
wse2_folder="./wse2"
disc_folder="./disclination"
f222_folder="./f222"

## running
echo "    Running Figure 1 simulations"
cd $wse2_folder 
. ./run_all.sh > wse2.out || { echo "Critical Error: Script failed! (Location: $PWD)"; return 1; }
cd ..

echo "    Running Figure 2 simulations"
cd $f222_folder 
. ./run_all.sh > f222.out || { echo "Critical Error: Script failed! (Location: $PWD)"; return 1; }
cd ..

echo "    Running Figure 3 simulations"
cd $disc_folder 
. ./run_all.sh > disc.out || { echo "Critical Error: Script failed! (Location: $PWD)"; return 1; }
cd ..

echo "    Running Figure 4 simulations"
cd $qwz_folder 
. ./run_all.sh > qwz.out || { echo "Critical Error: Script failed! (Location: $PWD)"; return 1; }
cd ..