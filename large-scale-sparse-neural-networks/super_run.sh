#!/bin/bash
cluster_cmd="qsub -V -b y -now y -cwd -pe mpi 2 -l mem=2G ./run.sh"
#cpu=2,num_proc=2,
echo "the following command was executed in super_run.sh followed by a seed(1-5)"
echo $cluster_cmd

for seed in 1; do $cluster_cmd $seed; done
