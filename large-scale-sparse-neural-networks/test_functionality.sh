qsub -b y -V -l mem=14G,h=twilight -cwd -pe default 4 -N test_run mpiexec -n 4 pdm run python parallel_training.py --config_file test --dataset cifar10 --monitor --epochs 5 --n-neurons 2000 --n-neurons 1000 --n-neurons 2000 --activations -0.75 --activations 0.75 --activations -0.75 --weight-decay 0.0 --n-training-samples 50000 --seed 1 --start_epoch_importancepruning 0 --importance_pruning_frequency 1
