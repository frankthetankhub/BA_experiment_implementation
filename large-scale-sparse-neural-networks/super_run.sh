#!/bin/bash
output_location="$HOME/scratch/Ba/large-scale-sparse-neural-networks/cluster_outputfiles/"
cluster_cmd="qsub -V -now y -cwd -o $output_location -e $output_location -pe default 2 -l mem=2G ./run_cluster.sh"
#cpu=2,num_proc=2,
echo "the following command was executed in super_run.sh followed by a seed(1-5)"
echo $cluster_cmd

for seed in 1; do $cluster_cmd $seed; done
