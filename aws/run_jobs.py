import argparse
import os
import subprocess
import time

import tqdm
import yaml
from quinine import *

from .startup import startup_commands

GPUS = {'v100': 'nvidia-tesla-v100', 'p100': 'nvidia-tesla-p100', 't4': 'nvidia-tesla-t4'}


def load_base_job_config():
    return yaml.load(open('aws/job.yaml'), Loader=yaml.FullLoader)


def create_job_config(job_key, job_id, root_path, module_to_run, config_path, store_path, use_gdb, gpu, pool):
    # Load up a base config
    base_config = load_base_job_config()

    # Modify it
    base_config['metadata']['name'] += f'-{job_key}-{job_id}'
    base_config['spec']['template']['spec']['containers'][0]['name'] += f'-{job_key}-{job_id}'

    # Add in the startup commands
    base_config['spec']['template']['spec']['containers'][0]['command'] = ['/bin/sh', '-c']
    startup_cmds = startup_commands(root_path, module_to_run, config_path, use_gdb)
    base_config['spec']['template']['spec']['containers'][0]['args'] = [' && '.join(startup_cmds)]

    # Modify the GPU
    # base_config['spec']['template']['spec']['nodeSelector']['k8s.amazonaws.com/accelerator'] = GPUS[gpu]
    if pool is not None:
        base_config['spec']['template']['spec']['nodeSelector']['kops.k8s.io/instancegroup'] = pool

    # Store this configuration
    yaml.dump(base_config, open(store_path, 'w'))

    return base_config


def launch_kubernetes_job(path):
    # Execute a job
    subprocess.run(['kubectl', 'create', '-f', f'{path}'])


def git_push(path, git_dir):
    # Change directory to the local git repository
    cwd = os.getcwd()
    os.chdir(git_dir)

    cmds = [['git', 'add', f'{path}/*'],
            ['git', 'commit', '-m', 'cfgupdates'],
            ['git', 'push']]

    for cmd in cmds:
        subprocess.run(cmd)

    # Change directory back
    os.chdir(cwd)


def launch(args):
    # Load up the sweep configuration file and create a storage path
    config = QuinSweep(sweep_config_path=args.sweep_config)
    store_path = args.sweep_config.split(".yaml")[0] + '_param_combos'
    os.makedirs(store_path, exist_ok=True)

    # Figure out the module this sweep is targeting: what's the .py file being run
    try:
        module_to_run = config[0]['general']['module']
    except:
        print("Please ensure that the config contains a general.module parameter indicating the module to be run.")
        raise

    # Create a unique key for all these jobs
    job_key = int(time.time())

    # Keep track of whatever job manifests (.yaml) we're generating
    # A single job will run a single configuration to completion
    job_yaml_paths = []

    # Go over each parameter configuration
    for i, choice in enumerate(config):
        # Dump this new configuration
        config_path = f'{store_path}/config_{i + 1}.yaml'
        yaml.dump(choice, open(config_path, 'w'))

        # Create a job configuration to run this
        job_yaml_path = f'{store_path}/job_{i + 1}.yaml'

        # Construct the remote config path
        remote_config_path = os.path.join(args.root_path, os.path.relpath(config_path, args.local_root_path))
        create_job_config(job_key=job_key,
                          job_id=i + 1,
                          root_path=args.root_path,
                          module_to_run=module_to_run,
                          config_path=remote_config_path,
                          store_path=job_yaml_path,
                          use_gdb=args.gdb,
                          gpu=args.gpu,
                          pool=args.pool)

        # Append to the queue of jobs we're running
        job_yaml_paths.append(job_yaml_path)

    # Git push
    print("####################################################################################")
    print("Pushing to Git!")
    print("####################################################################################")
    git_push(store_path, args.local_root_path)

    # Launch all the Kubernetes jobs
    if args.run or args.tentative:
        print("####################################################################################")
        print(f"Launching {'all the' if args.run else 'one of the'} Kubernetes jobs!")
        print("####################################################################################")
        for path in tqdm.tqdm(job_yaml_paths):
            launch_kubernetes_job(path)
            if args.tentative:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_config', '-c', required=True, help='Path to the sweep configuration file.')
    parser.add_argument('--root_path', '-rp', required=True, help='Path to the root directory of the repository '
                                                                  'whose method is being run.')
    parser.add_argument('--local_root_path', '-lrp', required=True, help='(Local) path to the root directory of the '
                                                                         'repository whose method is being run.')
    parser.add_argument('--tentative', '-t', action='store_true', help="Launch just a single job.")
    parser.add_argument('--run', '-r', action='store_true', help="Launch the jobs as well.")
    parser.add_argument('--gpu', '-g', type=str, choices=['v100', 'p100', 't4'], required=True)
    parser.add_argument('--pool', '-p', type=str, default=None)
    parser.add_argument('--gdb', action='store_true', help='Execute with gdb.')
    args = parser.parse_args()

    # Launch the parameter sweep
    launch(args)
