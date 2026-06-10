cd ./main_text/ 
echo "Running main text calculations"
. ./run_everything_main_text.sh || { echo "Critical Error: Script failed! (Location: $PWD)"; return 1; } 

cd ..

cd ./appendix/
echo "Running selected appendix calculations"
. ./run_everything_appendix.sh || { echo "Critical Error: Script failed! (Location: $PWD)"; return 1; }

cd ..

echo "Copying figures to root folder" 

. ./copy_figures.sh