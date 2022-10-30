# End of Grade Thesis (TFG Trabajo Fin de Grado)

- Repository where I develop my end of grade thesis

## Contact

- Sergio Quijano Rey
- [mail](sergioquijanorey@protonmail.com)
- [Personal Webpage](https://sergioquijanorey.github.io/)

## Pre-requisites

This project is configured via either `poetry` or `nix`.

### Poetry

You can install it and learn more [here](https://python-poetry.org/). With `poetry` installed, all the packages and their versions are described in `pyproject.toml` file.

`python 3.10` is needed in order to run the project.

1. Install all the dependencies: `poetry install`
2. Run all the unit tests: `poetry run python -m unittest src/test/*.py`
3. Run the notebook: `poetry run jupyter notebook`

### Nix

You can install it and learn more [here](https://nixos.org). Once installed, all the packages are described in `shell.nix` file. Nothing else is needed to run the project. Now, you can:

1. Run all the unit tests: `nix-shell --pure --run "python -m unittest src/test/*.py"`
2. Run the notebook: `nix-shell --pure --run "jupyter notebook"`
3. Enter a development shell:
    1. Impure shell: `nix-shell`
    2. Pure shell: `nix-shell --pure `
    - Pure shell has only access to packages defined in `shell.nix`. Impure shell has access to packages installed in your system
