import argparse
import os
import subprocess

import yaml

from .startup import pod_startup_commands

GPUS = {'v100': 'nvidia-tesla-v100',
        'p100': 'nvidia-tesla-p100',
        't4': 'nvidia-tesla-t4'}


def launch_pod(name, gpu, cpu, pool, image):
    # Load the base manifest for launching Pods
    config = yaml.load(open('aws/pod.yaml'))

    # Modify it
    if gpu != 'none':
        config['spec']['nodeSelector']['k8s.amazonaws.com/accelerator'] = GPUS[gpu]
        if pool:
            config['spec']['nodeSelector']['kops.k8s.io/instancegroup'] = pool
    else:
        assert pool is not None, "Specify the pool if you're not using a GPU."
        # Wipe out the GPU node selector
        config['spec']['nodeSelector'] = {}
        # Specify the pool
        config['spec']['nodeSelector']['kops.k8s.io/instancegroup'] = pool
        # Wipe out the GPU request
        config['spec']['containers'][0]['resources'] = {'limits': {}, 'requests': {}}

    if cpu:
        # Put in a CPU request
        config['spec']['containers'][0]['resources']['limits']['cpu'] = cpu
        config['spec']['containers'][0]['resources']['requests']['cpu'] = cpu

    # Set the name of the Pod
    config['metadata']['name'] = config['spec']['containers'][0]['name'] = name
    # Set the name of the image we want the Pod to run
    config['spec']['containers'][0]['image'] = image

    # Put in a bunch of startup commands
    config['spec']['containers'][0]['command'] = ['/bin/sh', '-c']
    config['spec']['containers'][0]['args'] = [' && '.join(pod_startup_commands())]

    # Store it
    yaml.dump(config, open('temp.yaml', 'w'))

    # Log
    print("Launching a pod...")
    print(f"Name: {name}\tGPU: {gpu}\tImage: {image}")
    print("###################################################################################")

    # Launch the Pod
    subprocess.call('kubectl apply -f temp.yaml', shell=True)

    # Clean up
    os.remove('temp.yaml')


def main(args):
    if args.resource == 'pod':
        launch_pod(args.name, args.gpu, args.cpu, args.pool, args.image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resource', '-r', type=str, choices=['pod', 'job'], required=True)
    parser.add_argument('--gpu', '-g', type=str, choices=['v100', 'p100', 't4', 'none'], required=True)
    parser.add_argument('--cpu', '-c', type=int, default=None)
    parser.add_argument('--pool', '-p', type=str, default=None)
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--image', '-im', type=str,
                        default='291343978134.dkr.ecr.us-west-2.amazonaws.com/aws-tf2-torch:latest')

    args = parser.parse_args()

    main(args)
