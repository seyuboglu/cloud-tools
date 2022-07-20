import click


@click.group()
@click.argument("mode", default="create", type=click.Choice(["create", "delete"]))
@click.option("--build-dir", default=None)
@click.option("--templates-dir", default=None)
@click.pass_context
def cli(
    ctx,
    mode: str,
    templates_dir: str = None,
    build_dir: str = None,
):
    ctx.ensure_object(dict)
    ctx.obj["mode"] = mode 
    ctx.obj["templates_dir"] = templates_dir
    ctx.obj["build_dir"] = build_dir


