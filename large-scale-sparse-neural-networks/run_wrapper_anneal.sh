if [[ $# -eq 0 ]] ; then
    echo 'Please specify a seed to use for all experimental setups'
    exit 0
fi
for conf in {1..8};
    do /bin/bash run_single_setup_once_single_worker.sh $1 $conf --anneal-zeta;
done