def install_commands():
    """
    Commands to install any extra dependencies that are missing in the Docker image.
    Typical when you have a large project and don't want to keep changing your Docker image for a few
    extra dependencies.
    """
    # insert pip install or apt-get commands here if any
    return [
        # pip
        'pip install git+https://github.com/krandiash/quinine.git',
        'pip install --upgrade git+https://github.com/tensorpack/dataflow.git',
        'pip install transformers tensorflow-datasets toposort jupyterlab',
        'pip install python-language-server[all]',
        'pip install jupyter-lsp',
        'pip install umap-learn',
        'pip install pandas matplotlib datashader bokeh holoviews colorcet nltk',
        'pip install fastBPE regex requests sacremoses subword_nmt',
        'pip install git+https://github.com/PetrochukM/PyTorch-NLP.git',
        'pip install git+https://github.com/makcedward/nlpaug.git numpy matplotlib python-dotenv',
        'pip install --upgrade wandb gin-config cytoolz funcy munch cerberus pytorch-ignite Cython',
        'pip install --upgrade git+https://github.com/aleju/imgaug.git',
        # 'pip install -U git+https://github.com/albu/albumentations',

        # apt-get
        'apt-get -y install libxext6 libx11-6 libxrender1 libxtst6 libxi6 libglib2.0-0',

        # nodejs
        'wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash',
        'export NVM_DIR="$HOME/.nvm" ',
        '\. "$NVM_DIR/nvm.sh" ',
        # 'exec bash',
        'nvm install 12.16.3',  # upgrade this version when required
        'nvm use 12.16.3',

        # conda
        'conda install -c conda-forge nodejs',

        # jupyter
        'jupyter labextension install @jupyter-widgets/jupyterlab-manager',
        'jupyter labextension install @jupyterlab/latex',
        'jupyter labextension install @jupyterlab/toc',
        'jupyter labextension install @krassowski/jupyterlab-lsp',

    ]


def git_commands(pull):
    """
    Commands to setup authentication for git so we can git pull on any codebase on the server.
    """
    cmds = [
        'mkdir ~/.ssh',
        'cp /home/.ssh/id_rsa ~/.ssh/',
        'ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts',
        'git config --global user.email "kgoel93@gmail.com"',
        'git config --global user.name "krandiash"',
    ]
    if pull:
        return cmds + ['git pull']
    return cmds


def wandb_commands():
    """
    Commands to authenticate Weights and Biases on the server.

    For reference, the file at /home/.wandb/auth just contains "wandb login <API_KEY>".
    """
    return [
        'bash /home/.wandb/auth',
    ]


def job_startup_commands(root_path, module_to_run, config_path, use_gdb):
    """
    Move to the directory for the project we're running, set the git configuration and pull,
    install all the additional libraries we need and setup Weights and Biases.

    This works on a typical server setup where
    e.g.
    your git repo and codebase is at /home/workspace/my_ml_project/,
    you have a config file for running your code at /home/workspace/my_ml_project/configs/config_train.yaml
    and you want to run a module located at /home/workspace/my_ml_project/methods/train.py.

    The corresponding parameters to be passed in would be
    root_path = "/home/workspace/my_ml_project/"
    module_to_run = "methods.train"
    config_path = "/home/workspace/my_ml_project/configs/config_train.yaml"
    and use_gdb = False unless you're doing heavy debugging.

    This function would ensure that you pull the latest changes to your local git repo before running.

    If editing this function, your last command should be the script that you want to run.
    """
    cmds = [f'cd {root_path}'] + git_commands(pull=True) + install_commands() + wandb_commands()

    if not use_gdb:
        # run_cmd = f'python {method_to_run}.py --config {config_path}'
        run_cmd = f'python -m {module_to_run} --config {config_path}'
    else:
        run_cmd = f'gdb -ex -r -ex backtrace full --args python -m {module_to_run} --config {config_path}'

    cmds.append(run_cmd)

    return cmds


def pod_startup_commands():
    """
    Set the git configuration, install all the additional libraries we need and setup Weights and Biases.

    If editing this function, the only command that's necessary is "sleep infinity".
    """
    return git_commands(pull=False) + install_commands() + wandb_commands() + \
           ['groupadd --gid 2000 foobargroup'] + ['sleep infinity']
