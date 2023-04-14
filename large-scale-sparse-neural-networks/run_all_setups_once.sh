# This is the runfile for not submitting jobs

# TODO include copying of config file into the relevant folder
echo please first specify a dataset to use <searches configs> and then an experimental hyperparameter configuration
if [[ $# -eq 0 ]] ; then
    echo 'Please specify a seed to use for all experimental setups'
    exit 0
fi
SEED=$1
echo $SEED
for FILE in configs/*/*;  
    do echo $FILE; 
    full_name=$FILE
    base_name=$(basename ${full_name})
    echo ${base_name}
    #|rigel.cv.uni-osnabrueck.de |cujam.cv.uni-osnabrueck.de
    CV_HOSTS='(albireo.cv.uni-osnabrueck.de|alioth.cv.uni-osnabrueck.de|beam.cv.uni-osnabrueck.de|bias.cv.uni-osnabrueck.de|dimension.cv.uni-osnabrueck.de|gremium.cv.uni-osnabrueck.de|light.cv.uni-osnabrueck.de|nashira.cv.uni-osnabrueck.de|perception.cv.uni-osnabrueck.de|shadow.cv.uni-osnabrueck.de|twilight.cv.uni-osnabrueck.de|vector.cv.uni-osnabrueck.de|voxel.cv.uni-osnabrueck.de)'
    ARGS=$(cat $FILE)
    echo $EXP_SETUP_ARGS
    cwd="/home/student/j/jankettler/scratch/Ba/large-scale-sparse-neural-networks"
    cluster_cmd="qsub -b y -V -l mem=8G,h=$CV_HOSTS -cwd -pe default 4"
    cluster_cmd_cifar="qsub -b y -V -l mem=20G,h=$CV_HOSTS -cwd -pe default 4"
    echo $cmd
    for CONF in configs/*; 
        do
        #test if it is an experimental setup file or a folder containing configs
        if test -f $CONF; then
            echo $CONF
            EXP_SETUP_ARGS=$(cat $CONF)
            cmd="mpiexec -n 4 pdm run python parallel_training.py $ARGS $EXP_SETUP_ARGS --config_file $base_name"
            if [[ $base_name == cifar10* ]];
            then
                echo cifar10
                echo $cluster_cmd_cifar -N ${base_name}_seed$SEED $cmd --seed $SEED
                #$cluster_cmd_cifar -N ${base_name}_seed$SEED $cmd --seed $SEED
            else
                echo $cluster_cmd -N ${base_name}_seed$SEED $cmd --seed $SEED
                #$cluster_cmd -N ${base_name}_seed$SEED $cmd --seed $SEED
            fi
        fi
    done
done