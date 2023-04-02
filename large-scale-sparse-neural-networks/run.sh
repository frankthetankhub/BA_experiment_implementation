#!/bin/bash

ARGS="--monitor --epochs 10 --processes 2 --n-neurons 784 --n-neurons 1000 --n-neurons 1000 --n-neurons 1000 --n-neurons 10"
location="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/Ba/large-scale-sparse-neural-networks/parallel_training.py"

python_loc="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/python/bin/python3.7"

#echo mpiexec -n 6 python parallel_training.py $ARGS 

python_cmd="$HOME/scratch/openmpi/bin/mpiexec -n 2 --report-bindings $python_loc $location"
no_mpi_command="python parallel_training.py"
#for SEED in 1 2 3 4 5; do echo $cmd $ARGS --seed $SEED; done

echo "First argument passed should be seed. Its value is: $1"
echo "Check CPU with -lscpu-"
lscpu
echo $LD_LIBRARY_PATH
echo $PATH
$python_cmd $ARGS --seed $1
# $no_mpi_command $ARGS --seed $1
