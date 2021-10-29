import importlib
import inspect
import os
import subprocess
import yaml
import datetime
from argparse import ArgumentParser
from utils.constants import DEFAULTS


def get_timestamp():
    """Get the current timestamp."""
    ts = datetime.datetime.now()
    ts_str = f"{ts:%Y-%m-%d-%H-%M-%S}"[2:]
    return ts_str



def cmdruns(timestamp, run_name, sweep_fn, envvars, dryrun=False):
    # Commit ID
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        commit_id = 'n/a'

    # Generate runs
    runs = sweep_fn()

    # Print to file
    cmds = {}
    filename = f'{DEFAULT.JOBLOG_DIR}/{timestamp}.sh'
    # try:
    with open(filename, 'w') as cmdfile:
        cmdfile.write('#!/usr/bin/env bash\n\n')
        cmdfile.write(f'# {timestamp}\n')
        cmdfile.write(f'# {commit_id}\n\n')
        cmdfile.write(f'# {len(runs)} configurations\n\n')
        print(f'{len(runs)} configurations')
        for i, args in enumerate(runs):
            print(args)
            cmd = DEFAULT.main_command(run_name, args, dryrun)
            # Prefix the environment variables passed in
            cmd = " ".join([envvars, cmd])
            cmds[f'{i+1}-{run_name}'] = cmd
            cmdfile.write(cmd + '\n')

        source = inspect.getsource(sweep_fn)
        commented_source = source.replace('\n', '\n# ')
        commented_source = '\n# ' + commented_source
        cmdfile.write(commented_source)
    # except:
    #     subprocess.run(['rm', filename])

    print(f"Wrote {filename}")

    return filename, cmds

def pool_dependent_conda_env(pool, conda_env):
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
    conda_env = pool_dependent_conda_env(pool, conda_env)
    return [
        f'source {DEFAULT.BASH_RC_PATH}',
        f'source {DEFAULT.CONDA_ACTIVATION_PATH}',
        f'conda activate {conda_env}',
        f'cd {startup_dir}',
        f'bash {DEFAULT.WANDB_PATH}' if DEFAULT.WANDB_PATH else 'sleep 1',
        # 'bash /home/.wandb/auth',
        # 'eval `ssh-agent -s`',
        # 'ssh-add /home/.ssh/id_rsa',
        # 'git pull --rebase',
        cmd,
    ]


def launch_pod(run_name, pool, image, cmd, startup_dir, conda_env):
    # Load the base manifest for launching Pods
    config = yaml.load(open(DEFAULT.BASE_POD_YAML_PATH), Loader=yaml.FullLoader)

    # Wipe out the GPU node selector
    config['spec']['nodeSelector'] = {}
    # Specify the pool
    config['spec']['nodeSelector']['cloud.google.com/gke-nodepool'] = f'{pool}'
    # Request GPUs
    # TODO(karan): figure out how to parse GPU requests from pool name or add option in cmdline
    config['spec']['containers'][0]['resources'] = {
        'limits': {'nvidia.com/gpu': pool.split("-")[1]},
        'requests': {'nvidia.com/gpu': pool.split("-")[1]}
    }

    # Set the name of the Pod
    config['metadata']['name'] = config['spec']['containers'][0]['name'] = run_name
    # Set the name of the image we want the Pod to run
    config['spec']['containers'][0]['image'] = image

    # Put in a bunch of startup commands
    config['spec']['containers'][0]['command'] = ['bash', '-c']
    config['spec']['containers'][0]['args'] = [' && '.join(commands(pool, cmd, startup_dir, conda_env))]

    # Store it
    yaml.dump(config, open('temp.yaml', 'w'))

    # Log
    print(f"Run name: {run_name}")

    # Launch the Pod
    subprocess.call('kubectl apply -f temp.yaml', shell=True)

    # Clean up
    os.remove('temp.yaml')


def switch_gcp_context(project, zone, cluster):
    """
    Switch the GCP context.

    Args:
        project (str): The GCP project to use.
        zone (str): The GCP zone to use.
        cluster (str): The GCP cluster to use.
    """
    subprocess.call(f'gcloud config set project {project}', shell=True)
    subprocess.call(f'gcloud container clusters get-credentials {cluster} --zone {zone}', shell=True)


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
        f, cmds = cmdruns(timestamp, run_name, sweep_fn, args.envvars, args.dryrun)
    else:
        # Directly run a command passed in
        run_name = f'{timestamp}' if args.interactive is None else args.interactive
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
        print(f"Launching pods...\nPool: {args.pool}\nImage: {DEFAULT.DEFAULT_IMAGE}")
        for run_name, cmd in cmds.items():
            launch_pod(
                run_name.replace(".", "-").replace("_", "-")[:60].lower(), 
                args.pool, 
                DEFAULT.DEFAULT_IMAGE, 
                cmd, 
                DEFAULT.DEFAULT_STARTUP_DIR, 
                DEFAULT.DEFAULT_CONDA_ENV,
            )

        if f is not None:
            subprocess.run(['chmod', '777', f])


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--project',
        help='Project for which to use this script.',
        required=True,
        choices=list(DEFAULTS.keys()),
    )
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
        required=True,
        help='Node pool where the sweep should be run. '
             'Ensure that the run uses up all the GPUs on the node.',
    )
    parser.add_argument(
        '--cmd',
        help="Manually specify command to run on the pod.",
    )
    parser.add_argument(
        '--startup_dir', '-sd',
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
        help='Image to use for the pod. Defaults to the image specified by the cluster defaults.'
    )
    parser.add_argument(
        '--conda',
        help='The conda environment to use. Defaults to the conda env specified by the cluster defaults.'
    )
    parser.add_argument(
        '--interactive', '-i',
        help="Creates an interactive pod with the specified name. `config`, `sweep` and `cmd` are ignored.",
    )
    # add an argument to pass in a string for environment variables
    parser.add_argument(
        '--envvars',
        help='Environment variables to set for the command being run e.g. "DATA_PATH=/home/common/datasets/" will be used to prefix the command.',
        type=str,
        default='',
    )
    args = parser.parse_args()

    # Inject defaults into program globals()
    global DEFAULT 
    DEFAULT = DEFAULTS[args.project]

    # Make JOBLOG_DIR if it doesn't exist
    if not os.path.exists(DEFAULT.JOBLOG_DIR):
        os.makedirs(DEFAULT.JOBLOG_DIR, exist_ok=True)

    # Change startup directory if specified
    if args.startup_dir:
        DEFAULT.DEFAULT_STARTUP_DIR = args.startup_dir

    # Change conda env if specified
    if args.conda:
        DEFAULT.DEFAULT_CONDA_ENV = args.conda

    # Change image if specified
    if args.image:
        DEFAULT.DEFAULT_IMAGE = args.image

    if args.interactive:
        args.cmd = 'sleep infinity'
        args.config = None
        args.sweep = None

    # Change the GCP context
    switch_gcp_context(DEFAULT.GCP_PROJECT, DEFAULT.GCP_ZONE, DEFAULT.GCP_CLUSTER)

    run(args)