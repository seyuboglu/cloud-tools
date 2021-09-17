import importlib
import inspect
import os
import subprocess
import yaml
import datetime
from argparse import ArgumentParser


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


def get_timestamp():
    """Get the current timestamp."""
    ts = datetime.datetime.now()
    ts_str = f"{ts:%Y-%m-%d-%H-%M-%S}"[2:]
    return ts_str

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


def cmdruns(timestamp, run_name, sweep_fn, dryrun=False):
    # Commit ID
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        commit_id = 'n/a'

    # Generate runs
    runs = sweep_fn()

    # Print to file
    cmds = {}
    filename = f'{JOBLOG_DIR}/{timestamp}.sh'
    try:
        with open(filename, 'w') as cmdfile:
            cmdfile.write('#!/usr/bin/env bash\n\n')
            cmdfile.write(f'# {timestamp}\n')
            cmdfile.write(f'# {commit_id}\n\n')
            cmdfile.write(f'# {len(runs)} configurations\n\n')
            print(f'{len(runs)} configurations')
            for i, args in enumerate(runs):
                cmd = main_command(run_name, args, dryrun)
                cmds[f'{i+1}-{run_name}'] = cmd
                cmdfile.write(cmd + '\n')

            source = inspect.getsource(sweep_fn)
            commented_source = source.replace('\n', '\n# ')
            commented_source = '\n# ' + commented_source
            cmdfile.write(commented_source)
    except:
        subprocess.run(['rm', filename])

    return filename, cmds

def pool_dependent_conda_env(pool, conda_env=DEFAULT_CONDA_ENV):
    """
    Return the conda environment to use for the given node pool.

    Args:
        pool (str): The node pool to use.
        conda_env (str): The default conda environment to use.

    Returns:
        str: The conda environment to use for the given node pool.
    """
    return conda_env

def commands(pool, cmd, startup_dir, conda_env):
    """
    Return the commands to be run on the cluster.

    Args:
        pool (str): The node pool to use.
        cmd (str): The main command to be run on the cluster.
        startup_dir (str): The startup directory on the cluster.
        conda_env (str): The conda environment to use.
    Returns:
        list: The commands to be run on the cluster.
    """
    # Optionally, use pool information to set the conda env 
    # (e.g. if need to use different setups for different GPUs)
    conda_env = pool_dependent_conda_env(pool) 
    return [
        f'source {BASH_RC_PATH}',
        f'source {CONDA_ACTIVATION_PATH}',
        f'conda activate {conda_env}',
        f'cd {startup_dir}',
        # 'bash /home/.wandb/auth',
        # 'eval `ssh-agent -s`',
        # 'ssh-add /home/.ssh/id_rsa',
        # 'git pull --rebase',
        cmd,
    ]


def launch_pod(run_name, pool, image, cmd, startup_dir):
    # Load the base manifest for launching Pods
    config = yaml.load(open(BASE_POD_YAML_PATH), Loader=yaml.FullLoader)

    # Wipe out the GPU node selector
    config['spec']['nodeSelector'] = {}
    # Specify the pool
    config['spec']['nodeSelector']['cloud.google.com/gke-nodepool'] = f'{pool}'
    # Request GPUs
    # config['spec']['containers'][0]['resources'] = {
    #     'limits': {'nvidia.com/gpu': pool.split("-")[-1]},
    #     'requests': {'nvidia.com/gpu': pool.split("-")[-1]}
    # }

    # Set the name of the Pod
    config['metadata']['name'] = config['spec']['containers'][0]['name'] = run_name
    # Set the name of the image we want the Pod to run
    config['spec']['containers'][0]['image'] = image

    # Put in a bunch of startup commands
    config['spec']['containers'][0]['command'] = ['bash', '-c']
    config['spec']['containers'][0]['args'] = [' && '.join(commands(pool, cmd, startup_dir))]

    # Store it
    yaml.dump(config, open('temp.yaml', 'w'))

    # Log
    print(f"Run name: {run_name}")

    # Launch the Pod
    subprocess.call('kubectl apply -f temp.yaml', shell=True)

    # Clean up
    os.remove('temp.yaml')


def run(args):
    """Construct the sweep and run it on the cluster."""

    # Timestamp
    timestamp = get_timestamp()
    
    if args.config and args.sweep:
        # For config and sweep, load the config Python file and the sweep function
        config = importlib.import_module(args.config)
        sweep_fn = getattr(config, args.sweep)
        run_name = f'{timestamp}--{args.config}--{args.sweep}'
        # Generate the commands
        f, cmds = cmdruns(timestamp, run_name, sweep_fn, args.dryrun)
    else:
        # Directly run a command passed in
        run_name = f'{timestamp}'
        f, cmds = None, {run_name: args.cmd}

    print("Timestamp:", timestamp)
    print("Run name:", run_name)

    if not args.autolaunch:
        if f is not None:
            subprocess.run(['cat', f])
            subprocess.run(['rm', f])
        else:
            print(f"Command: {args.cmd}")
    else:
        # Run using Kubernetes
        print(f"Launching pods...\nPool: {args.pool}\nImage: {args.image}")
        for run_name, cmd in cmds.items():
            launch_pod(run_name.replace(".", "-").replace("_", "-")[:60], args.pool, args.image, cmd, args.startup_dir)

        if f is not None:
            subprocess.run(['chmod', '777', f])


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--config', '-c',
        help='Name of the config file with dotted notation.',
        default=None,
    )
    parser.add_argument(
        '--sweep', '-s',
        help='Name of the sweep function inside the config file.',
        default=None,
    )
    parser.add_argument(
        '--pool', '-p',
        choices=NODE_POOLS,
        required=True,
        help='Node pool where the sweep should be run. '
             'Ensure that the run uses up all the GPUs on the node.',
    )
    parser.add_argument(
        '--cmd',
    )
    parser.add_argument(
        '--startup_dir', '-sd',
        default=DEFAULT_STARTUP_DIR,
        help='Startup directory for the pod.',
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Signal to the main program that this is a test run (e.g. to disable W & B).'
    )
    parser.add_argument(
        '--autolaunch', '-al',
        action='store_true',
        help='Autolaunch the sweep on the cluster.'
    )
    parser.add_argument(
        '--image',
        default=DEFAULT_IMAGE,
        help='Image to use for the pod.'
    )
    parser.add_argument(
        '--conda',
        default=DEFAULT_CONDA_ENV,
        help='The conda environment to use.'
    )
    args = parser.parse_args()

    # Make JOBLOG_DIR if it doesn't exist
    if not os.path.exists(JOBLOG_DIR):
        os.makedirs(JOBLOG_DIR, exist_ok=True)

    run(args)