# This is the runfile for not submitting jobs

# TODO include copying of config file into the relevant folder
echo please first specify a dataset to use <searches configs> and then an Host to run on
echo $1, $2
for FILE in configs/$1/*; 
    do echo $FILE; 
    full_name=$FILE
    base_name=$(basename ${full_name})
    echo ${base_name}
    cv_hosts="albireo alioth beam bias cujam dimension gremium light nashira perception rigel shadow twilight vector voxel"
    host_commands="h=albireo,h=alioth,h=beam,h=biash,h=cujam,h=dimension,h=gremium,h=light,h=nashira,h=perception,h=rigel,h=shadow,h=twilight,h=vector,h=voxel"
    queue="cv.q"
    ARGS=$(cat $FILE)
    EXP_SETUP_ARGS="--epsilon 10 --start_epoch_importancepruning 100"
    #sas="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python"
    #p="/scratch/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python3.7"
    #python="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/python/Python-3.7.16/python"
    cwd="/home/student/j/jankettler/scratch/Ba/large-scale-sparse-neural-networks"
    #cmd_alt="mpiexec -n 6 $sas $cwd/parallel_training.py $ARGS --config_file $base_name"
    cmd="mpiexec -n 6 pdm run python parallel_training.py $ARGS $EXP_SETUP_ARGS --config_file $base_name"
    cluster_cmd="qsub -b y -V -l mem=8G,h=$2 -cwd -pe default 6"
    cluster_cmd_cifar="qsub -b y -V -l mem=20G,h=$2 -cwd -pe default 6"
    echo $cmd
    if [[ $base_name == cifar10* ]];
    then
        echo cifar10
        for SEED in {1..5}; do echo $cluster_cmd_cifar -N ${base_name}_seed$SEED $cmd --seed $SEED; done
        for SEED in {1..5}; do $cluster_cmd_cifar -N ${base_name}_seed$SEED $cmd --seed $SEED; done
    else
        for SEED in {1..5}; do echo $cluster_cmd -N ${base_name}_seed$SEED $cmd --seed $SEED; done
        for SEED in {1..5}; do $cluster_cmd -N ${base_name}_seed$SEED $cmd --seed $SEED; done
    fi
done
