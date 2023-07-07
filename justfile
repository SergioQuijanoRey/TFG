CURRENT_NOTEBOOK := "LFW Notebook.ipynb"

# Uni server parameters
SSH_ALIAS := "ugr"
SSH_DATA_PATH := "/mnt/homeGPU/squijano/TFG/"

# Default command that list all the available commands
default:
	@just --list

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
        {{SSH_ALIAS}}:{{SSH_DATA_PATH}} ./

remote_fs:
    sshfs {{SSH_ALIAS}}:{{SSH_DATA_PATH}} ./remote_dev

# Downloads the notebook from Google Colab
download REMOTE:
	rclone copy --progress "{{REMOTE}}:Colab Notebooks/{{CURRENT_NOTEBOOK}}" src/ && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

# Runs all the benchmarks, using shell.nix
benchmarks:
    #!/usr/bin/env zsh

    # We need to add some paths to PYTHONPATH
    # Otherwise, imports will not work properly
    export PYTHONPATH=$PYTHONPATH:./
    export PYTHONPATH=$PYTHONPATH:./src

    # Iterate over all files and run the benchmarks
    for file in src/benchmarks/*.py
    do
        nix-shell --run "zsh -c 'python $file'"
    done

# Runs all the tests
tests:
    python -m unittest src/test/*.py && notify-send "🟢 All tests passed" || notify-send -u critical "🔴 Some tests failed"

# Run all the linters configured for the project
lint:
    ruff src/lib || echo "Lib is not clean"
    ruff src/benchmarks || echo "Benchmarks are not clean"
    ruff src/tests || echo "Tests are not clean"

# Run type checks using mypy
type_lint:
    mypy src/lib || echo "Lib is not well typed"
    mypy src/benchmarks || echo "Benchmarks are not well typed"
    mypy src/tests || echo "Tests are not well typed"
