from dataclasses import dataclass
import os
import subprocess
from typing import List
import pathlib
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader
from .pod import PodBuilder
from kubernetes import client, config


class Volume:
    def __init__(self, name):
        self.name = name
        self.pv_name = f"pv-{name}"
        self.pvc_name = f"pvc-{name}"
        self.mount_path = f"/{name}"


class JobBuilder(PodBuilder):

    def create_job(self):
        """
        Create a new NFS server
        """
        path = self._render(
            template_path="jobs/command-job.yaml",
            job_name=f"job-{self.name}",
            volumes=self.volumes,
            node_pool=self.node_pool,
            gpus=self.gpus,
            command=self.command,
            user=self.user,
            gpu_type=self.gpu_type
        )
        subprocess.run(["kubectl", "apply", "-f", path])

    def __call__(self):
        self.create_job()


from .cli import cli
import click


@cli.command()
@click.argument("name", default="dev", type=click.STRING)
@click.option(
    "--volumes", "-v", default=["home", "data"], type=click.STRING, multiple=True
)
@click.option("--user", "-u", default="sabri", type=click.STRING)
@click.option("--node-pool", "-n", default="dev", type=click.STRING)
@click.option("--gpus", "-g", default=0, type=click.INT)
@click.option("--gpu-type", "-t", default="nvidia-tesla-t4", type=click.STRING)
@click.option("--command", "-c", default="sleep infinity", type=click.STRING)
@click.pass_context
def job(ctx, name: str, volumes: List[str], user: str, node_pool: str, gpus: int, gpu_type: str, command: str):
    """
    Create a new job
    """
    builder = JobBuilder(
        name=name,
        volumes=volumes,
        user=user, 
        node_pool=node_pool,
        gpus=gpus,
        gpu_type=gpu_type,
        command=command,
        build_dir=ctx.obj["build_dir"],
        templates_dir=ctx.obj["templates_dir"],
    )
    if ctx.obj["mode"] == "create":
        builder()
    elif ctx.obj["mode"] == "delete":
        builder.delete()
