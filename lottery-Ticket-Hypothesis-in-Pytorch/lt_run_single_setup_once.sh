if [[ $# -eq 1 ]] ; then
    echo 'Please specify a seed to use for all experimental setups as first argument, and the experimental setup as second argument.'
    exit 0
fi
SEED=$1
echo $SEED
full_name=configs/dataset_size/${2}
base_name=$(basename ${full_name})
echo ${base_name}
#|rigel.cv.uni-osnabrueck.de |cujam.cv.uni-osnabrueck.de|beam.cv.uni-osnabrueck.de
CV_HOSTS='(albireo.cv.uni-osnabrueck.de|alioth.cv.uni-osnabrueck.de|bias.cv.uni-osnabrueck.de|dimension.cv.uni-osnabrueck.de|gremium.cv.uni-osnabrueck.de|light.cv.uni-osnabrueck.de|nashira.cv.uni-osnabrueck.de|perception.cv.uni-osnabrueck.de|shadow.cv.uni-osnabrueck.de|twilight.cv.uni-osnabrueck.de|vector.cv.uni-osnabrueck.de|voxel.cv.uni-osnabrueck.de)'
ARGS=$(cat $full_name)

cluster_cmd="qsub -b y -V -l mem=3G,cuda=1,h=$CV_HOSTS -cwd"
cluster_cmd_cifar="qsub -b y -V -l mem=6G,cuda=1,h=$CV_HOSTS -cwd"
cmd="pdm run python main.py $ARGS --config_file $base_name"
if [[ $base_name == lt_cifar10* ]]; then
    echo cifar10
    echo $cluster_cmd_cifar -N ${base_name}_s$SEED $cmd --seed $SEED
    $cluster_cmd_cifar -N ${base_name}_s$SEED $cmd --seed $SEED
else
    echo $cluster_cmd -N ${base_name}_s$SEED $cmd --seed $SEED
    $cluster_cmd -N ${base_name}_s$SEED $cmd --seed $SEED
fi