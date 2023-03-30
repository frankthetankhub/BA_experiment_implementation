#!/bin/bash
cluster_cmd="qsub -v -b -cwd -pe default 2 -l cpu=2,num_proc=2,mem=2G ./run.sh" 
echo "the following command was executed in super_run.sh followed by a seed(1-5)"
echo $cluster_cmd

for seed in 1; do $cluster_cmd $seed; done
