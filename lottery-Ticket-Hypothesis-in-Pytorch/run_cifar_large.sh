#!/bin/bash

MAIN="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/Ba/lottery-Ticket-Hypothesis-in-Pytorch/main.py"

ARGS="--dataset cifar10 --arch_type fc1 --end_iter 25 --prune_iterations 24 --prune_percent 20 --trial_iterations 1 --patience 4 --arch_size cifar_large"

ARGS_TEST="--dataset fashionmnist --arch_type fc1 --end_iter 1 --prune_iterations 1 --prune_percent 20 --trial_iterations 1 --patience 4"

python_loc="/home/student/j/jankettler/.local/share/pdm/venvs/lottery-Ticket-Hypothesis-in-Pytorch-ER_pAaTh-3.7/bin/python"

cluster_cmd="qsub -b y -cwd -l mem=8G,cuda=1 -V"
#-b
#cuda=1 

#for seed in 1 2 3 4 5; do echo $cluster_cmd $python_loc $MAIN $ARGS --seed $seed; done

echo $cluster_cmd $python_loc $MAIN $ARGS_TEST --seed 1
echo "PATH:" $PATH
echo "which python3.7"
which python3.7
echo "working directory and content"
pwd
ls
echo "setting LD_LIBRARY_PATH to /lib"
export LD_LIBRARY_PATH="/lib/":$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
name="cifar10_large"
out_file="out_lt_cifar10_large/$(date)/"
error_file="error_lth_cifar10_large/$(date)/"
echo $out_file$name$seed
for seed in 1 2 3 4 5; do echo $cluster_cmd -o $out_file$name$seed -e $out_file$name$seed $python_loc $MAIN $ARGS_TEST --seed $seed >> "runs from $(date) cifar10_large.txt"; done
for seed in 1 2 3 4 5; do $cluster_cmd -o $out_file$name$seed -e $error_file$name$seed $python_loc $MAIN $ARGS_TEST --seed $seed; done