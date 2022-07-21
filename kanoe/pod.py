from dataclasses import dataclass
import os
import subprocess
from typing import List
import pathlib
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader
from kubernetes import client, config


class Volume:
    def __init__(self, name):
        self.name = name
        self.pv_name = f"pv-{name}"
        self.pvc_name = f"pvc-{name}"
        self.mount_path = f"/{name}"


class PodBuilder:
    def __init__(
        self,
        name: str,
        volumes: List[str],
        node_pool: str = "dev",
        user: str = "sabri",
        gpus: int = 0,
        command: str = "sleep infinity",
        templates_dir: str = None,
        build_dir: str = None,
    ):
        self.name = name
        self.user = user 
        self.node_pool = node_pool
        self.volumes = [Volume(name=volume) for volume in volumes]
        self.gpus = gpus
        self.command = command

        if templates_dir is None:
            dir = pathlib.Path(__file__).parent.resolve().parent.resolve()
            self.templates_dir = os.path.join(dir, "templates")
        else:
            self.templates_dir = templates_dir

        if build_dir is None:
            dir = pathlib.Path(__file__).parent.resolve().parent.resolve()
            self.build_dir = os.path.join(dir, "build")
        else:
            self.build_dir = build_dir

    def _render(self, template_path: str, **kwargs):
        env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template(template_path)
        yaml = template.render(**kwargs)
        path = os.path.join(self.build_dir, template_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(yaml)
        return path

    def create_pod(self):
        """
        Create a new NFS server
        """
        path = self._render(
            template_path="pods/dev-pod.yaml",
            pod_name=f"pod-{self.name}",
            volumes=self.volumes,
            node_pool=self.node_pool,
            gpus=self.gpus,
            command=self.command,
            user=self.user
        )
        subprocess.run(["kubectl", "apply", "-f", path])

    def __call__(self):
        self.create_pod()


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
@click.option("--command", "-c", default="sleep infinity", type=click.STRING)
@click.pass_context
def pod(ctx, name: str, volumes: List[str], user: str, node_pool: str, gpus: int, command: str):
    """
    Create a new pod
    """
    builder = PodBuilder(
        name=name,
        volumes=volumes,
        user=user, 
        node_pool=node_pool,
        gpus=gpus,
        command=command,
        build_dir=ctx.obj["build_dir"],
        templates_dir=ctx.obj["templates_dir"],
    )
    if ctx.obj["mode"] == "create":
        builder()
    elif ctx.obj["mode"] == "delete":
        builder.delete()
