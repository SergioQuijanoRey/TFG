# The current jupyter notebook
# This is the notebook that we are going to use when working (upload, download)
# with Google Colab
CURRENT_NOTEBOOK := "CACD Notebook.ipynb"

# The current script
# We are using this script to run the training / evaluation
CURRENT_SCRIPT := "src/FG-Net.py"

# Uni server parameters
SSH_ALIAS := "ugr"
SSH_DATA_PATH := "/mnt/homeGPU/squijano/TFG/"

# Anaconda parameters
ANACONDA_ENV_NAME := "squijano"

# Default command that list all the available commands
default:
	@just --list --unsorted


# == GOOGLE COLAB ==
# ==============================================================================

# NOTE: `REMOTE` should be `UniDrive` or `MyDrive` (as they are my remotes
# configured in rclone
#
# Uploads the notebook, the lib code and also the benchmarks
upload_all REMOTE:
    just upload_lib "{{REMOTE}}"
    just upload_benchmarks "{{REMOTE}}"
    rclone copy "src/{{CURRENT_NOTEBOOK}}" "{{REMOTE}}:Colab Notebooks" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

# Uploads just lib code
upload_lib REMOTE:
	rclone sync --progress src/lib/ "{{REMOTE}}:Colab Notebooks/lib/" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

# Uploads just benchmarks code and the benchmarking notebook
upload_benchmarks REMOTE:
    # Upload the benchmarking code in .py files
    rclone sync --progress src/benchmarks/ "{{REMOTE}}:Colab Notebooks/benchmarks/" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

    # Now upload the notebook that wraps that .py code to run it in Google Colab
    rclone copy "src/benchmarks/Benchmarking notebook.ipynb" "{{REMOTE}}:Colab Notebooks/benchmarks/" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

# Uploads both lib and benchmarks code
upload_lib_benchmarks REMOTE:
    just upload_lib "{{REMOTE}}"
    just upload_benchmarks "{{REMOTE}}"


# Downloads the notebook from Google Colab
download REMOTE:
	rclone copy --progress "{{REMOTE}}:Colab Notebooks/{{CURRENT_NOTEBOOK}}" src/ && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"


# == UGR UNI's SERVERS ==
# ==============================================================================


# Upload code to UNI server
upload_uni:
    rsync -zaP \
        --exclude 'custom_env_squijano' \
        --exclude '.git' \
        --exclude '.github' \
        --exclude '.mypy_cache' \
        --exclude 'justfile' \
        --exclude 'slurm*.out' \
        --exclude 'wandb' \
        --exclude '.worktree' \
        --exclude 'tmp' \
        --exclude 'cached_models' \
        --exclude 'cached_augmented_dataset.pt' \
        --exclude 'data' \
        --exclude 'src/data' \
        --exclude 'src/lib/__pycache__' \
        --exclude 'training.log' \
        --exclude '.venv' \
        --exclude 'thesis' \
        ./ {{SSH_ALIAS}}:{{SSH_DATA_PATH}}

# Download code from the UNI server
# Avoid downloading useless logs and other files
download_uni:
    rsync -zaP \
        --exclude 'custom_env_squijano' \
        --exclude '.git' \
        --exclude '.github' \
        --exclude '.mypy_cache' \
        --exclude 'justfile' \
        --exclude 'slurm*.out' \
        --exclude 'wandb' \
        --exclude '.worktree' \
        --exclude 'tmp' \
        --exclude 'cached_models' \
        --exclude 'cached_augmented_dataset.pt' \
        --exclude 'data' \
        --exclude 'src/data' \
        --exclude 'src/lib/__pycache__' \
        --exclude 'training.log' \
        --exclude '.venv' \
        {{SSH_ALIAS}}:{{SSH_DATA_PATH}} ./

# Create a remote file system, so we can work easily on the server
remote_fs:
    sshfs {{SSH_ALIAS}}:{{SSH_DATA_PATH}} ./remote_dev


# == LOCAL TASKS ==
# ==============================================================================

# Run the current script
run:
    python {{CURRENT_SCRIPT}}


# Runs all the benchmarks, using shell.nix
benchmarks:
    #!/usr/bin/env zsh

    # Iterate over all files and run the benchmarks
    for file in src/benchmarks/*.py
    do
        nix-shell --run "zsh -c 'python $file'"
    done

# Runs all the tests, both unit tests and integration tests
tests:
    python -m unittest src/test/*.py && notify-send "🟢 All unit tests passed" || notify-send -u critical "🔴 Some unit tests failed"
    python -m unittest src/integration_tests/*.py && notify-send "🟢 All integration tests passed" || notify-send -u critical "🔴 Some integration tests failed"

# Run all the linters configured for the project
lint:
    ruff src/lib || echo "🏮Lib is not clean"
    ruff src/benchmarks || echo "🏮Benchmarks are not clean"
    ruff src/tests || echo "🏮Unit Tests are not clean"
    ruff src/integration_tests || echo "🏮Integration Tests are not clean"

# Run type checks using mypy
type_lint:
    mypy src/lib || echo "🏮Lib is not well typed"
    mypy src/benchmarks || echo "🏮Benchmarks are not well typed"
    mypy src/tests || echo "🏮Unit Tests are not well typed"
    mypy src/integration_tests || echo "🏮Integration Tests are not well typed"


# == ANACONDA ENVIROMENT ==
# ==============================================================================

# Activate the conda enviroment and then export it into a `enviroment.yml` file
conda_export:
    conda activate {{ANACONDA_ENV_NAME}}
    conda env export > enviroment.yml

# Use a `enviroment.yml` file to create a conda enviroment
conda_import:
    conda env create -f environment.yml
    conda env list
