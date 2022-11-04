current_notebook := "LFW Notebook.ipynb"

# Default command that list all the available commands
default:
	@just --list

# Uploads the notebook, the lib code and also the benchmarks
upload_all REMOTE:
    just upload_lib "{{REMOTE}}"
    just upload_benchmarks "{{REMOTE}}"
    rclone copy "src/{{current_notebook}}" "{{REMOTE}}:Colab Notebooks" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

# Uploads just lib code
upload_lib REMOTE:
	rclone sync --progress src/lib/ "{{REMOTE}}:Colab Notebooks/lib/" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

# Uploads just benchmarks code
upload_benchmarks REMOTE:
    rclone sync --progress src/benchmarks/ "{{REMOTE}}:Colab Notebooks/benchmarks/" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"


# Downloads the notebook from Google Colab
download REMOTE:
	rclone copy --progress "{{REMOTE}}:Colab Notebooks/{{current_notebook}}" src/ && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

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
