

cd ./disc/
echo "    Running disclination-related appendix calculations"
. ./run_all.sh > disc.out || { echo "Critical Error: Script failed! (Location: $PWD)"; return 1; }

cd .. 
echo "    Running QWZ-related appendix calculations"
cd ./qwz/

. ./run_all.sh > qwz.out || { echo "Critical Error: Script failed! (Location: $PWD)"; return 1; }

cd .. 
echo "    Running WSe2-related appendix calculations"
cd ./wse2/

. ./run_all.sh > wse2.out || { echo "Critical Error: Script failed! (Location: $PWD)"; return 1; } 

cd ..