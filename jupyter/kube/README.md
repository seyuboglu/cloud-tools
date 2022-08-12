# Installation of JupyterHub on Kubernetes 
Follow instructions at https://zero-to-jupyterhub.readthedocs.io/en/latest/jupyterhub/installation.html

Installing JupyterHub on Kubernetes with Helm,
```
    helm upgrade --cleanup-on-fail \
    --install jupyterhub-1.2.0 jupyterhub/jupyterhub \
    --namespace default \
    --version=1.2.0 \
    --values config.yaml
```
This command can be rerun any time config.yaml gets changed to update the JupyterHub server.


## Make a node pool
Make a new node pool like normal e.g. `jupyterhub-t4-1`. Add the following label when creating it:
```
hub.jupyter.org/node-purpose:user
```
and the following taint
```
hub.jupyter.org/dedicated=user:NoSchedule
```

## The joyvan folder
If you're mounting a NFS volume at /home/, you'll want to make sure that you create
```
mkdir /home/joyvan/work
chmod 777 /home/joyvan
chmod 777 /home/joyvan/work
```
This folder is used by Docker to do stuff inside, so we need to make sure it's present.

## Docker container
