"""CLI commands for report generation."""

from pathlib import Path

import click

from hercule.reports import generate_report


@click.command()
@click.argument("experiment_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for the generated report (default: experiment_path/report.py)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def generate(experiment_path: Path, output: Path | None, verbose: bool):
    """
    Generate a report for an experiment.

    EXPERIMENT_PATH: Path to the experiment directory containing JSON files
    """
    if verbose:
        click.echo(f"Generating report for experiment: {experiment_path}")

    try:
        report_path = generate_report(experiment_path, output)
        click.echo(f"✅ Report generated successfully: {report_path}")
        click.echo("\nTo run the report:")
        click.echo("1. Open a Jupyter notebook")
        click.echo(f"2. Run the cells in: {report_path}")
        click.echo("3. The report will display visualizations and analysis")

    except Exception as e:
        click.echo(f"❌ Error generating report: {e}", err=True)
        raise click.Abort()


@click.group()
def reports():
    """Report generation commands for Hercule experiments."""
    pass


reports.add_command(generate)


if __name__ == "__main__":
    reports()
