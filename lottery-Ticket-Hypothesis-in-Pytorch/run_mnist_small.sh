#!/bin/bash

MAIN="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/Ba/lottery-Ticket-Hypothesis-in-Pytorch/main.py"

ARGS="--dataset mnist --arch_type fc1 --end_iter 25 --prune_iterations 24 --prune_percent 20 --trial_iterations 1 --patience 9 --arch_size mnist_small"

# python_loc="/home/student/j/jankettler/.local/share/pdm/venvs/lottery-Ticket-Hypothesis-in-Pytorch-ER_pAaTh-3.7/bin/python"

cluster_cmd="qsub -b y -cwd -l mem=8G,cuda=1 -V"

echo "PATH:" $PATH
echo "working directory and content"
pwd
ls
name="mnist_small"
out_file="out_lt_mnist_small/$(date)/"
error_file="error_lth_mnist_small/$(date)/"
eval $(pdm venv activate 3.7)
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/lib/":$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
which python
for seed in 1 2 3 4 5; do echo $cluster_cmd -o $out_file$name$seed -e $error_file$name$seed python $MAIN $ARGS_TEST --seed $seed >> "runs from $(date) mnist_small.txt"; done
for seed in 1 2 3 4 5; do $cluster_cmd python $MAIN $ARGS --seed $seed; done
