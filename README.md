# BA_experiment_implementation

## Structure

This repo contains all the code I used to run my experiments for my bachelor's thesis.
This "master" branch is not used, but I uploaded the plots and tables I collected here as well. These are also available on the large_scale_cluster branch.
large_scale_cluster contains all the code I used to collect data for my experiments using the dynamic sparse training approach. It also contains all the code I used to generate the plots and tables. The lth_pytorch_cluster branch contains all the code used to generate results for the lottery hypothesis.
As the name suggests, I ran all my experiments on the uni osnabrück cluster, so the code should be used on that cluster.  
  

## How to use
Clone this repo and checkout the appropriate branch (one of the two above). In each of the subfolders (lottery-ticket-hypothesis-in-Pytorch; large-scale-sparse-neural-networkslarge_scale) you will find a pdm.lock file as well as a pyproject.toml and similar requirements. I suggest installing the dependencies using python pdm, which would allow you to sync to the exact versions of the requirements I used, as these are pinned in the lockfile.
I used python 3.7 for all my experiments.  
To run the experiments on the cluster, run one of the run_*****.sh files, depending on which configurations and datasets you want to use.  
If you want to run these implementations locally, you would need to run *main.py* for the LTH implementation, and either *parallel_training.py* or *set_mlp_sequential.py* for the DST approaches. You can also have a look at the .vscode/launch.json files for debugging purposes etc.
In addition, you will find the README from the original authors of the code in each subfolder. These may give you a better idea of how to use the code, and also show that most of the code has been taken from their respective repositories, which you can also refer to for more information.  

**A word of warning:**   
The results I have gathered regarding the DST implementation require mpi4py to work properly, at least when using the multiproccessing approach. mpi4py in turn relies on an mpich or openmpi implementation. This needs to be manually installed, and the mpi4py package needs to be custom configured to know how to find this installation and work properly. This requires specific steps that I will not go into here. But as a reference, have a look at these websites:  
 https://mpi4py.readthedocs.io/en/stable/install.html ; https://edu.itp.phys.ethz.ch/hs12/programming_techniques/openmpi.pdf .  
 In general, I think it would be easiest if you try to repeat the experiments on your local machine, although I cannot give you any hints on specific steps that would be required or if any problems would occur.