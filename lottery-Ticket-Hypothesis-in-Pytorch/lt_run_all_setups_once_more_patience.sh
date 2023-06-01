if [[ $# -eq 0 ]] ; then
    echo 'Please specify a seed to use for all experimental setups'
    exit 0
fi
SEED=$1
echo $SEED
for FILE in configs/more_patience/*;  
    do echo $FILE; 
    full_name=$FILE
    base_name=$(basename ${full_name})
    echo ${base_name}
    #|rigel.cv.uni-osnabrueck.de |cujam.cv.uni-osnabrueck.de|beam.cv.uni-osnabrueck.de
    CV_HOSTS='(albireo.cv.uni-osnabrueck.de|alioth.cv.uni-osnabrueck.de|bias.cv.uni-osnabrueck.de|dimension.cv.uni-osnabrueck.de|gremium.cv.uni-osnabrueck.de|light.cv.uni-osnabrueck.de|nashira.cv.uni-osnabrueck.de|perception.cv.uni-osnabrueck.de|shadow.cv.uni-osnabrueck.de|twilight.cv.uni-osnabrueck.de|vector.cv.uni-osnabrueck.de|voxel.cv.uni-osnabrueck.de)'
    ARGS=$(cat $FILE)

    cluster_cmd="qsub -b y -V -l mem=3G,cuda=1,h=$CV_HOSTS -cwd"
    cluster_cmd_cifar="qsub -b y -V -l mem=5G,cuda=1,h=$CV_HOSTS -cwd"
    cmd="pdm run python main.py $ARGS --config_file ${base_name}_50_patience"
    if [[ $base_name == lt_cifar10* ]]; then
        echo cifar10
        echo $cluster_cmd_cifar -N ${base_name}_s$SEED $cmd --seed $SEED
        $cluster_cmd_cifar -N ${base_name}_s$SEED $cmd --seed $SEED
    else
        echo $cluster_cmd -N ${base_name}_s$SEED $cmd --seed $SEED
        $cluster_cmd -N ${base_name}_s$SEED $cmd --seed $SEED
    fi   
done