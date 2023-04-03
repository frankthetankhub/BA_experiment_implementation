# This is the runfile for not submitting jobs
ARGS="--monitor --epochs 10 --processes 6 --n-neurons 784 --n-neurons 1000 --n-neurons 1000 --n-neurons 1000 --n-neurons 10"
echo "Seed: $1"
sas="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python"
p="/scratch/pdm_venvs/large-scale-sparse-neural-networks-yLgfzVYs-3.7/bin/python3.7"
python="/net/projects/scratch/summer/valid_until_31_January_2024/jankettler/python/Python-3.7.16/python"
cwd="/home/student/j/jankettler/scratch/Ba/large-scale-sparse-neural-networks"
cmd="mpiexec -n 6 $sas $cwd/parallel_training.py $ARGS --seed $1"
echo $cmd
$cmd
