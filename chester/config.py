import os.path as osp
import os

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
LOG_DIR = os.path.join(PROJECT_PATH, "data")

SEUSS_HOME_FOLDER = 'your_andrew_id'
SEUSS_PROJECT_NAME = 'test'

# Make sure to use absolute path
REMOTE_DIR = {
    'seuss': '/home/{}/Projects/{}'.format(SEUSS_HOME_FOLDER, SEUSS_PROJECT_NAME),
}

REMOTE_MOUNT_OPTION = {
    'seuss': '/usr/share/glvnd',
    'autobot': '/usr/share/glvnd',
}

REMOTE_LOG_DIR = {
    'seuss': '/data/{}/{}/data'.format(SEUSS_HOME_FOLDER, SEUSS_PROJECT_NAME),
}

# slurm header to write for the job
REMOTE_HEADER = dict(seuss="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=10
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:1 
#SBATCH --mem=10G
""".strip())

# location of the singularity file related to the project
SIMG_DIR = {
    'seuss': '/home/yufeiw2/softgymcontainer_v3.simg',
}
CUDA_MODULE = {
    'seuss': 'cuda-91',
    'autobot': 'cuda-10.2',
    'psc': 'cuda/9.0',
}
MODULES = {
    'seuss': ['singularity'],
    'autobot': ['singularity'],
    'psc': ['singularity'],
}
