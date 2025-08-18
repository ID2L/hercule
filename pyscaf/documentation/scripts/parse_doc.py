import subprocess
import sys
from pathlib import Path

import tomli


def load_pdoc_config(pyproject_path: Path) -> dict:
    if not pyproject_path.exists():
        print(f"pyproject.toml not found at {pyproject_path}")
        sys.exit(1)
    with pyproject_path.open("rb") as f:
        pyproject = tomli.load(f)
    return pyproject.get("tool", {}).get("pyscaf", {}).get("documentation", {}).get("pdoc", {})


def load_documentation_config(pyproject_path: Path) -> dict:
    """Load the complete documentation configuration."""
    if not pyproject_path.exists():
        print(f"pyproject.toml not found at {pyproject_path}")
        sys.exit(1)
    with pyproject_path.open("rb") as f:
        pyproject = tomli.load(f)
    return pyproject.get("tool", {}).get("pyscaf", {}).get("documentation", {})


def get_poetry_package_paths() -> list:
    """Extract package paths from Poetry configuration."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return []

    with pyproject_path.open("rb") as f:
        pyproject = tomli.load(f)

    packages = pyproject.get("tool", {}).get("poetry", {}).get("packages", [])
    paths = []

    for package in packages:
        if isinstance(package, dict):
            include = package.get("include")
            from_dir = package.get("from", "")

            if include:
                if from_dir:
                    path = f"{from_dir}/{include}"
                else:
                    path = include
                paths.append(path)

    return paths


def config_to_pdoc_args(config: dict) -> list:
    args = []
    for key, value in config.items():
        if key == "output":
            print("Warning: 'output' argument ignored. Use gen_doc function for output directory management.")
            continue
        cli_key = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                args.append(cli_key)
            else:
                args.append(f"--no-{cli_key}")
        elif isinstance(value, list):
            for v in value:
                args.extend([cli_key, str(v)])
        else:
            args.extend([cli_key, str(value)])
    return args


def serve_doc():
    """Serve the documentation using pdoc."""
    pyproject_path = Path("pyproject.toml")
    config = load_pdoc_config(pyproject_path)
    args = config_to_pdoc_args(config)
    # Add modules/paths to document (required by pdoc)
    modules = config.get("modules")
    if modules:
        if isinstance(modules, str):
            args.append(modules)
        elif isinstance(modules, list):
            args.extend(modules)
    # Remove 'modules' from config to avoid duplicate
    if "modules" in config:
        del config["modules"]

    # Add Poetry package paths as positional arguments
    package_paths = get_poetry_package_paths()
    args.extend(package_paths)

    cmd = [sys.executable, "-m", "pdoc"] + args
    print(f"Running: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


def gen_doc():
    """Generate documentation to the specified output directory."""
    pyproject_path = Path("pyproject.toml")
    doc_config = load_documentation_config(pyproject_path)
    pdoc_config = doc_config.get("pdoc", {})

    # Get output path from configuration
    output_path = doc_config.get("output_path")
    if not output_path:
        print("Error: 'output_path' not specified in [tool.pyscaf.documentation]")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare pdoc arguments (excluding output)
    args = config_to_pdoc_args(pdoc_config)

    # Add output directory
    args.extend(["--output", str(output_dir)])

    # Add modules/paths to document
    modules = pdoc_config.get("modules")
    if modules:
        if isinstance(modules, str):
            args.append(modules)
        elif isinstance(modules, list):
            args.extend(modules)

    # Add Poetry package paths as positional arguments
    package_paths = get_poetry_package_paths()
    args.extend(package_paths)

    cmd = [sys.executable, "-m", "pdoc"] + args
    print(f"Running: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


def main():
    """Main entry point - choose between serve and generate modes."""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "serve":
            serve_doc()
        elif mode == "generate":
            gen_doc()
        else:
            print("Usage: python parse_doc.py [serve|generate]")
            print("  serve: Start pdoc server")
            print("  generate: Generate documentation to output directory")
            sys.exit(1)
    else:
        # Default to serve mode
        serve_doc()


if __name__ == "__main__":
    main()
