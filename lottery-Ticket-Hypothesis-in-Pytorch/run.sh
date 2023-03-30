#!/bin/bash

MAIN="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/Ba/lottery-Ticket-Hypothesis-in-Pytorch/main.py"

ARGS="--dataset fashionmnist --arch_type fc1 --end_iter 25 --prune_iterations 24 --prune_percent 20 --trial_iterations 1 --patience 4"

ARGS_TEST="--dataset fashionmnist --arch_type fc1 --end_iter 1 --prune_iterations 1 --prune_percent 20 --trial_iterations 1 --patience 4"

python_loc="/home/student/j/jankettler/python/Python-3.7.16/python"

cluster_cmd="qsub -v -b -cwd -l mem=8G,cuda=1" 

#for seed in 1 2 3 4 5; do echo $cluster_cmd $python_loc $MAIN $ARGS --seed $seed; done

$cluster_cmd $python_loc $MAIN $ARGS_TEST --seed $seed