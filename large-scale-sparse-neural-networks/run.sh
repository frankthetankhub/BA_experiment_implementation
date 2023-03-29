#!/bin/bash

ARGS="--monitor --epochs 10 --processes 6 --n-neurons 784 --n-neurons 1000 --n-neurons 1000 --n-neurons 1000 --n-neurons 10"
location="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/Ba/large-scale-sparse-neural-networks/parallel_training.py"
main_location="/home/jan/BA/code/large-scale-sparse-neural-networks/parallel_training.py"
python_loc="/home/student/j/jankettler/python/Python-3.7.16/python"

#echo mpiexec -n 6 python parallel_training.py $ARGS

#for SEED in 1 2 3 4 5; do echo mpiexec -n 6 python parallel_training.py $ARGS --seed $SEED; done

#mpiexec -4 6 $python_loc $location $ARGS
# for seed in 1 2 3 4 5; do echo qhost -h qhost -h xdev.ikw.uni-osnabrueck.de; done

#-l um spezifizierungen anzugeben
#-b um binaries zu zulassen
#-v PATH --> die neue umgebung übernimmt die PATH variable aus dem scope wo wir gerade sind(venv) 
#-cwd: current working directory; or wd=path/to/directory um working directory zu setzen

cmd="qsub -v -b -cwd -l mem=8G,cuda=1 mpiexec -n 6 $python_loc $location"

#for SEED in 1 2 3 4 5; do echo $cmd $ARGS --seed $SEED; done

echo "First argument passed should be seed. Its value is: $1"
echo $cmd $ARGS --seed $1