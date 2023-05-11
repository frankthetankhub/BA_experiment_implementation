if [[ $# -eq 1 ]] ; then
    echo 'Please specify a seed to use for all experimental setups as first argument, and the experimental setup as second argument.'
    exit 0
fi
SEED=$1
#SETUP=$2
echo $SEED
# for FILE in configs/dataset_size/*;  
#     do echo $FILE; 
full_name=configs/more_patience/${2}
base_name=$(basename ${full_name})

#ARCHS="cifar_large cifar_medium cifar_small fashionmnist_large fashionmnist_medium fashionmnist_small mnist_large mnist_medium mnist_small"
echo ${base_name}
#|rigel.cv.uni-osnabrueck.de |cujam.cv.uni-osnabrueck.de|beam.cv.uni-osnabrueck.de
CV_HOSTS='(albireo.cv.uni-osnabrueck.de|alioth.cv.uni-osnabrueck.de|bias.cv.uni-osnabrueck.de|dimension.cv.uni-osnabrueck.de|gremium.cv.uni-osnabrueck.de|light.cv.uni-osnabrueck.de|nashira.cv.uni-osnabrueck.de|perception.cv.uni-osnabrueck.de|shadow.cv.uni-osnabrueck.de|twilight.cv.uni-osnabrueck.de|vector.cv.uni-osnabrueck.de|voxel.cv.uni-osnabrueck.de)'
ARGS=$(cat $full_name)

cluster_cmd="qsub -b y -V -l mem=3G,cuda=1,h=$CV_HOSTS -cwd"
cluster_cmd_cifar="qsub -b y -V -l mem=6G,cuda=1,h=$CV_HOSTS -cwd"
#for CONF in configs/*; 
    #do
    #test if it is an experimental setup file or a folder containing configs
#if test -f $CONF; then
#full_conf_name=$CONF
#base_conf_name=$(echo $full_conf_name | sed 's/\///' )
#echo $base_conf_name
#CONF_FILE_SAVE_PARAMETER="$base_conf_name/$base_name"
#EXP_SETUP_ARGS=$(cat $CONF)
#cmd="mpiexec -n 4 pdm run python parallel_training.py $ARGS $EXP_SETUP_ARGS --config_file $CONF_FILE_SAVE_PARAMETER"
cmd="pdm run python main.py $ARGS --config_file ${base_name}_50_patience"
if [[ $base_name == lt_cifar10* ]]; then
    echo cifar10
    echo $cluster_cmd_cifar -N ${base_name}_s$SEED $cmd --seed $SEED
    $cluster_cmd_cifar -N ${base_name}_s$SEED $cmd --seed $SEED
else
    echo $cluster_cmd -N ${base_name}_s$SEED $cmd --seed $SEED
    $cluster_cmd -N ${base_name}_s$SEED $cmd --seed $SEED
fi
    
#done