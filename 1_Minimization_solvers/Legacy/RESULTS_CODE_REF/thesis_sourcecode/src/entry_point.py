from clis import cli


def main():
    cli.app(prog_name='by-mtsc-unilu')
    cli.app.add_typer()