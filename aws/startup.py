def install_commands():
    # insert pip install or apt-get commands here if any
    return [
        # pip
        'pip install git+https://github.com/krandiash/quinine.git',
        'pip install --upgrade git+https://github.com/tensorpack/dataflow.git',
        'pip install tensorflow-datasets toposort',
        'pip install git+https://github.com/PetrochukM/PyTorch-NLP.git',
        'pip install transformers',
        'pip install --upgrade wandb gin-config cytoolz funcy munch cerberus pytorch-ignite',
        'pip install --upgrade git+https://github.com/aleju/imgaug.git',
        # 'pip install -U git+https://github.com/albu/albumentations',

        # apt-get
        'apt-get -y install libxext6 libx11-6 libxrender1 libxtst6 libxi6 libglib2.0-0',
    ]


def git_commands():
    return [
        'mkdir ~/.ssh',
        'cp /home/.ssh/id_rsa ~/.ssh/',
        'ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts',
        'git config --global user.email "kgoel93@gmail.com"',
        'git config --global user.name "krandiash"',
        'git pull',
    ]


def wandb_commands():
    return [
        'bash /home/.wandb/auth',
    ]


def startup_commands(root_path, module_to_run, config_path, use_gdb):
    cmds = [f'cd {root_path}'] + git_commands() + install_commands() + wandb_commands()

    if not use_gdb:
        # run_cmd = f'python {method_to_run}.py --config {config_path}'
        run_cmd = f'python -m {module_to_run} --config {config_path}'
    else:
        run_cmd = f'gdb -ex -r -ex backtrace full --args python -m {module_to_run} --config {config_path}'

    cmds.append(run_cmd)

    return cmds
