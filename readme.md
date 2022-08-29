# A quick tutorial of using chester to launch large-scale experiments on SEUSS cluster
This repo contains a minimal working example of using chester, a pacakge developed by lan PhD alumni Xingyu Lin, to easily launch multiple jobs from your local desktop/labtop to the lab cluster SEUSS.

## How does chester work?
The SEUSS cluster uses a job allocation system named `slurm` to run jobs. To use the slurm system, one needs to write a slurm script to wrap the acutal program (e.g., a python script) you want to run, and submit the slurm script to the slurm system. So without chester, what you will usually do for running jobs on SEUSS would be:
- copy your local code, e.g., via github or scp, from your local desktop/laptop to seuss.  
- write a slurm script that runs the desired code.  
- submit the script to the slurm system.  
Chester takes care for you all these three steps so you no longer need to do these manually. Basically, by writing a custom launch file using chester (e.g., the `examples/launch.py`), this pacakge will automatically use rsync to synchronize your local code to seuss, generate the slurm script for running your code, and submit it to the slurm system. It will also dump the output and error to files so you can check the status of your job and do debugging. 

## Instructions
- Clone this repo.
- Make sure you have an account on SEUSS. Please follow instructions here to obtain an account on the seuss cluster.
- Modify the following variables in `chester/config.py`:
    - `SEUSS_HOME_FOLDER`: change to your home folder on seuss. Usually, this will be your andrew id.
    - `SEUSS_PROJECT_NAME`: change this to be the name of your project. A folder at `/home/your_andrew_id/projects/SEUSS_PROJECT_NAME` will be created to store your source code for the project (all code in the current folder will be copied to this directory on seuss). A folder at `/data/your_andrew_id/SEUSS_PROJECT_NAME/data/local/exp_prefix` will be created to store all outputs, logging of your experiments. For what `exp_prefix` means, keep reading.
- Write a launch file for launching the jobs to SEUSS. An example launch file is provided in this repo at `examples/launch.py`. The launch file should have the following things:
    - It should import a `run_task` function that runs the actual code you want to run, in this example, it is defined in `examples/train.py`, which trains a MLP to fit randomly generated data.
    - Chester uses `VariantGenerator` to do parameter sweep. See the launch file to see how you can define different parameters.
    - There is another parameter named `task_per_gpu`, which controls how many tasks you can run on a single GPU in parallel. The slurm system will grab a whole GPU at one time no matter how much memory/computation you acutally need, so sometimes you can run multiple jobs on 1 GPU if each of your job does not have a large memory/computation need.
    - You can use `exp_prefix` to define the name of your experiments. 
    - You don't need to worry about the other stuff in the launch file for now.
- You should wrap all your running code in a `run_task` function. See `run_task` function in `examples/train.py` for example. It also contains example code of how to use the chester logger.
- You should also create a `prepare.sh` to, e.g., activate your conda environment on SEUSS, and add current path to `PYTHONPAHT`. See `prepare.sh` for example.
- To launch jobs, you can run `python examples/lauch.py seuss --no-debug`. This will crate four jobs on SEUSS, each on 1 GPU, for the combination of 2 seeds and 2 learning rates.
- Your experimental results will be stored on SEUSS at the folder `/data/your_andrew_id/SEUSS_PROJECT_NAME/data/local/exp_prefix`. It will contain 4 folders, each corresponding to 1 combination of the hyper-parameters. In each of the 4 foler, you will see `slurm.out` which stores the output from your program, `slurm.err` that stores stderr (useful for debugging!), `variant.json` that stores all the hyper-parameters of your experiments, and `progress.csv` that store all the things you logges using chester.logger. 
- You can run `python chester/pull_result.py seuss exp_prefix` to pull all these to your local desktop/laptop, e.g., for visualizations. The pulled results will be stored at `data/seuss/exp_prefix`.
- For the example in this repo, you should see sth like this for `slurm.out`:
```
compute-0-19.local
0/10, Loss: 0.33745935559272766
1/10, Loss: 0.27606400847435
2/10, Loss: 0.27246713638305664
3/10, Loss: 0.27210861444473267
4/10, Loss: 0.2602773904800415
5/10, Loss: 0.2573573589324951
6/10, Loss: 0.2634148895740509
7/10, Loss: 0.2601708769798279
8/10, Loss: 0.2550356686115265
9/10, Loss: 0.25528818368911743
```
- And sth like this for `slurm.err`:
```
+ set -u
+ set -e
+ srun hostname
+ module load singularity
++ /usr/bin/modulecmd bash load singularity
+ eval LC_ALL=C ';export' 'LC_ALL;LC_CTYPE=C' ';export' 'LC_CTYPE;LOADEDMODULES=rocks-openmpi:singularity' ';export' 'LOADEDMODULES;SINGULARITY_SHELL=/bin/bash' ';export' 'SINGULARITY_SHELL;_LMFILES_=/usr/share/Modules/modulefiles/rocks-openmpi:/usr/share/Modules/modulefiles/singularity' ';export' '_LMFILES_;'
++ LC_ALL=C
++ export LC_ALL
++ LC_CTYPE=C
++ export LC_CTYPE
++ LOADEDMODULES=rocks-openmpi:singularity
++ export LOADEDMODULES
++ SINGULARITY_SHELL=/bin/bash
++ export SINGULARITY_SHELL
++ _LMFILES_=/usr/share/Modules/modulefiles/rocks-openmpi:/usr/share/Modules/modulefiles/singularity
++ export _LMFILES_
+ module load cuda-91
++ /usr/bin/modulecmd bash load cuda-91
+ eval CUDA_PATH=/opt/cuda/9.1 ';export' 'CUDA_PATH;LD_LIBRARY_PATH=/opt/cuda/9.1/lib64:/opt/cuda/9.1/lib:/opt/openmpi/lib:/home/yufeiw2/.mujoco/mujoco200/bin' ';export' 'LD_LIBRARY_PATH;LOADEDMODULES=rocks-openmpi:singularity:cuda-91' ';export' 'LOADEDMODULES;PATH=/opt/cuda/9.1/bin:/home/yufeiw2/miniconda3/bin:/home/yufeiw2/miniconda3/condabin:/opt/openmpi/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/bin:/opt/ganglia/bin:/opt/ganglia/sbin:/opt/pdsh/bin:/opt/rocks/bin:/opt/rocks/sbin' ';export' 'PATH;_LMFILES_=/usr/share/Modules/modulefiles/rocks-openmpi:/usr/share/Modules/modulefiles/singularity:/usr/share/Modules/modulefiles/cuda-91' ';export' '_LMFILES_;'
++ CUDA_PATH=/opt/cuda/9.1
++ export CUDA_PATH
++ LD_LIBRARY_PATH=/opt/cuda/9.1/lib64:/opt/cuda/9.1/lib:/opt/openmpi/lib:/home/yufeiw2/.mujoco/mujoco200/bin
++ export LD_LIBRARY_PATH
++ LOADEDMODULES=rocks-openmpi:singularity:cuda-91
++ export LOADEDMODULES
++ PATH=/opt/cuda/9.1/bin:/home/yufeiw2/miniconda3/bin:/home/yufeiw2/miniconda3/condabin:/opt/openmpi/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/bin:/opt/ganglia/bin:/opt/ganglia/sbin:/opt/pdsh/bin:/opt/rocks/bin:/opt/rocks/sbin
++ export PATH
++ _LMFILES_=/usr/share/Modules/modulefiles/rocks-openmpi:/usr/share/Modules/modulefiles/singularity:/usr/share/Modules/modulefiles/cuda-91
++ export _LMFILES_
+ cd /home/yufeiw2/Projects/test
+ singularity exec -B /usr/share/glvnd --nv /home/yufeiw2/softgymcontainer_v3.simg /bin/bash -c '. ./prepare.sh && sleep 0 && python /home/yufeiw2/Projects/test/chester/run_exp_worker.py  --exp_name test_experiment-08_28_23_53_29-001  --log_dir /data/yufeiw2/test/data/local/test_experiment/test_experiment-08_28_23_53_29-001  --use_cloudpickle True  --args_data gASVHwAAAAAAAACMDmV4YW1wbGVzLnRyYWlulIwIcnVuX3Rhc2uUk5Qu  --variant_data gASVuAAAAAAAAACMD2NoZXN0ZXIucnVuX2V4cJSMC1ZhcmlhbnREaWN0lJOUKYGUKIwEc2VlZJRLZIwFZXBvY2iUSwqMB2N1ZGFfaWSUSwCMAmxylEc/hHrhR64Ue4wMX2hpZGRlbl9rZXlzlF2UjAhleHBfbmFtZZSMInRlc3RfZXhwZXJpbWVudC0wOF8yOF8yM181M18yOS0wMDGUjApncm91cF9uYW1llIwPdGVzdF9leHBlcmltZW50lHVoA2Iu'
```