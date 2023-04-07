# This is the runfile for not submitting jobs

# TODO include copying of config file into the relevant folder
for FILE in configs/*; 
    do echo $FILE; 
    ARGS=$(cat $FILE)
    echo $ARGS
    sas="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python"
    p="/scratch/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python3.7"
    python="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/python/Python-3.7.16/python"
    cwd="/home/student/j/jankettler/scratch/Ba/large-scale-sparse-neural-networks"
    cmd_alt="mpiexec -n 6 $sas $cwd/parallel_training.py $ARGS --config_file $FILE"
    cmd="mpiexec -n 6 pdm run python parallel_training.py $ARGS --config_file $FILE"
    cluster_cmd="qsub -b y -V -l mem=8G -cwd -pe default 6"
    echo $cmd
    echo "testing for seed loop"
    for SEED in {1..2}; do echo $cluster_cmd $cmd --seed $SEED; done
    for SEED in {1..2}; do $cmd --seed $SEED; done
done 
