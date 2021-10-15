from dataclasses import dataclass

@dataclass
class Options:
    # Default Docker image to use for the Pods
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-hippo/torch18-cu111'
    # Path to conda profile.d/conda.sh file
    CONDA_ACTIVATION_PATH = '/home/miniconda3/etc/profile.d/conda.sh'
    # Path to bash rc file
    BASH_RC_PATH = '/home/workspace/.bashrc'
    # Default conda environment on the cluster
    DEFAULT_CONDA_ENV = 'hippo'
    # Default startup directory on the cluster
    DEFAULT_STARTUP_DIR = '/home/workspace/projects/'
    # The base manifest for launching Pods
    BASE_POD_YAML_PATH = 'pod.yaml'
    # List of node pools that can be used on the cluster
    NODE_POOLS = ['t4-1', 't4-4', 'p100-1', 'p100-4', 'v100-1', 'v100-8']
    # Directory where logs for launched Pods will be stored
    JOBLOG_DIR = './joblogs'
    # GCP project name
    GCP_PROJECT = 'hai-gcp-hippo'
    # GCP zone
    GCP_ZONE = 'us-west1-a'
    # Cluster name
    GCP_CLUSTER = 'platypus-1'
    # Path to wandb authentication file
    WANDB_PATH = None
    

    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        all_args = ' '.join(args)
        if dryrun:
            cmd = f"python -m train runner=pl {all_args} wandb.group={run_name} runner.wandb=False\n"
        else:
            cmd = f"python -m train runner=pl runner.wandb=True wandb.group={run_name} {all_args}\n"
        return cmd


@dataclass
class UnagiGCPFineGrained(Options):
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-fine-grained/default'
    CONDA_ACTIVATION_PATH = '/home/common/envs/conda/etc/profile.d/conda.sh'
    BASH_RC_PATH = '/home/workspace/.bashrc'
    DEFAULT_CONDA_ENV = 'unagi'
    DEFAULT_STARTUP_DIR = '/home/workspace/projects/unagi/'
    BASE_POD_YAML_PATH = 'utils/pod-unagi-gcp-fine-grained.yaml'
    NODE_POOLS = ['t4-1', 't4-2', 't4-4', 'train-t4-1', 'v100-1-small']
    JOBLOG_DIR = './joblogs'
    GCP_PROJECT = 'hai-gcp-fine-grained'
    GCP_ZONE = 'us-west1-a'
    GCP_CLUSTER = 'cluster-1'

    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        cmd = f"unagi {' '.join(args)}"
        return cmd


@dataclass
class HippoGCPHippo(Options):
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-hippo/torch18-cu111'
    CONDA_ACTIVATION_PATH = '/home/miniconda3/etc/profile.d/conda.sh'
    BASH_RC_PATH = '/home/.bashrc'
    DEFAULT_CONDA_ENV = 'hippo'
    DEFAULT_STARTUP_DIR = '/home/workspace/hippo/'
    BASE_POD_YAML_PATH = 'utils/pod-unagi-gcp-fine-grained.yaml'
    NODE_POOLS = ['t4-1', 't4-4', 'p100-1', 'p100-4', 'v100-1', 'v100-8']
    JOBLOG_DIR = './joblogs'
    GCP_PROJECT = 'hai-gcp-hippo'
    GCP_ZONE = 'us-west1-a'
    GCP_CLUSTER = 'platypus-1'

    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        if dryrun:
            cmd = f"python -m train wandb=null {' '.join(args)}"
        else:
            cmd = f"python -m train wandb.group={run_name} {' '.join(args)}"
        return cmd

@dataclass
class HippoGCPFineGrained(Options):
    DEFAULT_IMAGE = 'gcr.io/hai-gcp-fine-grained/default'
    CONDA_ACTIVATION_PATH = '/home/common/envs/conda/etc/profile.d/conda.sh'
    BASH_RC_PATH = '/home/workspace/.bashrc'
    DEFAULT_CONDA_ENV = 'hippo'
    DEFAULT_STARTUP_DIR = '/home/workspace/projects/hippo/'
    BASE_POD_YAML_PATH = 'utils/pod-unagi-gcp-fine-grained.yaml'
    NODE_POOLS = ['t4-1', 't4-2', 't4-4', 'train-t4-1', 'v100-1-small']
    JOBLOG_DIR = './joblogs'
    GCP_PROJECT = 'hai-gcp-fine-grained'
    GCP_ZONE = 'us-west1-a'
    GCP_CLUSTER = 'cluster-1'
    WANDB_PATH = '/home/workspace/.wandb/auth'

    @staticmethod
    def main_command(run_name, args, dryrun=False):
        """
        Return the main command to be run on the cluster.

        Args:
            run_name (str): The name of the run.
            args (dict): The arguments to be passed to the main command.
            dryrun (bool): Whether to run the command in dryrun mode.

        Returns:
            str: The main command to be run on the cluster.
        """
        if dryrun:
            cmd = f"python -m train wandb=null {' '.join(args)}"
        else:
            cmd = f"python -m train wandb.group={run_name} {' '.join(args)}"
        return cmd


DEFAULTS = {
    'unagi-gcp-fg': UnagiGCPFineGrained(),
    'hippo-gcp-hippo': HippoGCPHippo(),
    'hippo-gcp-fg': HippoGCPFineGrained(),    
}
