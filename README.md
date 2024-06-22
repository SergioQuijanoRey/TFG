# TFG Trabajo Fin de Grado

- Repository where I develop my _TFG_ ML project, thesis and dissertation.

## Contact

- Sergio Quijano Rey.
- [mail](sergioquijanorey@protonmail.com).
- [Personal Webpage](https://sergioquijanorey.github.io/).

## Pre-requisites

This project can be configured using either `poetry` or `nix`. We recommend the later as it is more robust.

### Nix

You can install it and learn more [here](https://nixos.org). Once installed, all the dev environments are described in `flake.nix` file. Nothing else is needed to run the project. Now, you can:

1. Enter the Python environment running `nix develop`
2. Enter the Latex environment running `nix develop .#writing`

### Poetry

You can install it and learn more [here](https://python-poetry.org/). With `poetry` installed, all the packages and their versions are described in `pyproject.toml` file.

`python 3.10` is needed in order to run the project.

1. Install all the dependencies: `poetry install`
2. Run all the unit tests: `poetry run python -m unittest src/test/*.py`
3. Run the notebook: `poetry run jupyter notebook`
