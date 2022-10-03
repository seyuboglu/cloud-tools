import os
import subprocess
import pathlib

from jinja2 import Environment, FileSystemLoader
from kubernetes import client, config


class VolumeBuilder:
    def __init__(
        self,
        name: str,
        zone: str = "us-west1-a",
        node_pool: str = "io-nfs-small",
        pd_name: str = None,
        create_pd: bool = False, 
        size: int = 10,  # in GB
        type: str = "pd-ssd",
        build_dir: str = None,
        templates_dir: str = None,
    ):
        self.name = name
        self.zone = zone
        self.node_pool = node_pool
        self.size = size
        self.type = type
        self.create_pd = create_pd

        if pd_name is None:
            self.pd_name = f"pd-{self.name}"
        else:
            self.pd_name = pd_name

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

    def create_nfs_server(self):
        """
        Create a new NFS server
        """
        path = self._render(
            template_path="deployments/nfs-server.yaml",
            pd_name=self.pd_name,
            server_name=f"nfs-server-{self.name}",
            node_pool=self.node_pool,
        )
        subprocess.run(["kubectl", "apply", "-f", path])

    def create_nfs_service(self):
        service_name = f"nfs-service-{self.name}"
        path = self._render(
            template_path="services/nfs-service.yaml",
            service_name=service_name,
            server_name=f"nfs-server-{self.name}",
        )
        subprocess.run(["kubectl", "apply", "-f", path])

        config.load_kube_config()
        api = client.CoreV1Api()
        self.service_cluster_ip = api.read_namespaced_service(
            name=service_name, namespace="default"
        ).spec.cluster_ip

    def create_nfs_pv(self):
        """
        Create a new NFS PV
        """
        path = self._render(
            template_path="volumes/nfs-pv.yaml",
            pv_name=f"pv-{self.name}",
            pvc_name=f"pvc-{self.name}",
            size=f"{self.size}Gi",
            server_ip=self.service_cluster_ip,
        )
        subprocess.run(["kubectl", "apply", "-f", path])

    def create_persistent_disk(
        self,
    ):
        # gcloud compute disks create --size=10GB --zone=us-east1-b gce-nfs-disk
        subprocess.call(
            [
                "gcloud",
                "compute",
                "disks",
                "create",
                f"--type={self.type}",
                f"--size={self.size}GB",
                f"--zone={self.zone}",
                f"{self.pd_name}",
            ]
        )

    def __call__(self):
        if self.create_pd:
            self.create_persistent_disk()
        self.create_nfs_server()
        self.create_nfs_service()
        self.create_nfs_pv()

    def delete(self):
        subprocess.run(["kubectl", "delete", "pvc", f"pvc-{self.name}"])
        subprocess.run(["kubectl", "delete", "pv", f"pv-{self.name}"])
        subprocess.run(["kubectl", "delete", "service", f"nfs-service-{self.name}"])
        subprocess.run(["kubectl", "delete", "deployment", f"nfs-server-{self.name}"])


from .cli import cli
import click


@cli.command()
@click.argument("name")
@click.option("--zone", default="us-west1-a")
@click.option("--node-pool", default="io-nfs-small")
@click.option("--size", default=10, type=int)
@click.option("--type", default="pd-ssd")
@click.option("--pd-name", default=None, type=str)
@click.option("--create-pd", default=False, is_flag=True)
@click.pass_context
def volume(
    ctx,
    name: str,
    pd_name: str,
    zone: str,
    node_pool: str,
    create_pd: bool,
    size: int,
    type: str,
):
    builder = VolumeBuilder(
        name=name,
        zone=zone,
        node_pool=node_pool,
        pd_name=pd_name,
        create_pd=create_pd,
        size=size,
        type=type,
        build_dir=ctx.obj["build_dir"],
        templates_dir=ctx.obj["templates_dir"],
    )

    if ctx.obj["mode"] == "create":
        builder()
    elif ctx.obj["mode"] == "delete":
        builder.delete()
