# This is the runfile for not submitting jobs
for FILE in configs/*; 
    do echo $FILE; 
    ARGS=$(cat $FILE)
    echo $ARGS
    sas="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python"
    p="/scratch/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python3.7"
    python="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/python/Python-3.7.16/python"
    cwd="/home/student/j/jankettler/scratch/Ba/large-scale-sparse-neural-networks"
    cmd="mpiexec -n 6 $sas $cwd/parallel_training.py $ARGS"
    echo $cmd
    echo "testing for seed loop"
    for SEED in {1..5}; do echo $cmd --seed $SEED; done
    #$cmd
done 
