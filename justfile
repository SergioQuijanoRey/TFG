CURRENT_NOTEBOOK := "LFW Notebook.ipynb"

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
    rclone copy "src/Benchmarking notebook.ipynb" "{{REMOTE}}:Colab Notebooks/" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

# Uploads both lib and benchmarks code
upload_lib_benchmarks REMOTE:
    just upload_lib "{{REMOTE}}"
    just upload_benchmarks "{{REMOTE}}"

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

