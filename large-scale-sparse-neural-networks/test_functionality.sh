qsub -q cv.q -b y -V -l mem=18G -cwd -pe default 6 -N test_run mpiexec -n 6 pdm run python parallel_training.py --config_file test --dataset cifar10 --monitor --epochs 5 --n-neurons 2000 --n-neurons 1000 --n-neurons 2000 --activations -0.75 --activations 0.75 --activations -0.75 --weight-decay 0.0 --n-training-samples 50000 --seed 1