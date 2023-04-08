base_name=cifar10_large
echo ${base_name}
cv_hosts="albireo alioth beam bias cujam dimension gremium light nashira perception rigel shadow twilight vector voxel"
host_commands="h=albireo,h=alioth,h=beam,h=biash,h=cujam,h=dimension,h=gremium,h=light,h=nashira,h=perception,h=rigel,h=shadow,h=twilight,h=vector,h=voxel"
queue="cv.q"
ARGS="--dataset cifar10 --arch_type fc1 --end_iter 250 --prune_iterations 24 --prune_percent 20 --trial_iterations 1 --patience 14 --arch_size cifar_large"
EXP_SETUP_ARGS="--epsilon 20 --start_epoch_importancepruning 200"
#sas="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python"
#p="/scratch/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python3.7"
#python="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/python/Python-3.7.16/python"
cwd="/home/student/j/jankettler/scratch/Ba/lottery-Ticket-Hypothesis-in-Pytorch"
#cmd_alt="mpiexec -n 6 $sas $cwd/parallel_training.py $ARGS --config_file $base_name"
cmd="pdm run python main.py $ARGS"
cluster_cmd="qsub -q $queue -b y -cwd -l mem=20G,cuda=1 -V"
cluster_cmd_cifar="qsub -q $queue -b y -cwd -l mem=20G,cuda=1 -V"
echo $cmd
#if [[ $base_name == cifar10* ]];
#then
#    echo cifar10
#    for SEED in {1..5}; do echo $cluster_cmd_cifar -N ${base_name}_seed$SEED $cmd --seed $SEED; done
#    for SEED in {1..5}; do $cluster_cmd_cifar -N ${base_name}_seed$SEED $cmd --seed $SEED; done
#else
for SEED in {1..5}; do echo $cluster_cmd -N ${base_name}_seed$SEED $cmd --seed $SEED; done
for SEED in {1..5}; do $cluster_cmd -N ${base_name}_seed$SEED $cmd --seed $SEED; done
#fi
#done