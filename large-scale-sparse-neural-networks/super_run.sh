#!/bin/bash
cluster_cmd="qsub -v -b -cwd -l mem=8G,cuda=1 ./run.sh" 

for seed in 1 2; do $cluster_cmd $seed; done