current_notebook := "LFW Notebook.ipynb"


default:
	@just --list

# Uploads the notebook and the lib code
upload_all REMOTE:
	rclone sync --progress src/lib/ "{{REMOTE}}:Colab Notebooks/lib/" && rclone copy "src/{{current_notebook}}" {{REMOTE}}:Colab Notebooks" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

# Uploads just lib code
upload_lib REMOTE:
	rclone sync --progress src/lib/ "{{REMOTE}}:Colab Notebooks/lib/" && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"

# Downloads the notebook from Google Colab
download REMOTE:
	rclone copy --progress "{{REMOTE}}:Colab Notebooks/{{current_notebook}}" src/ && notify-send "🟢 Rclone succeed" || notify-send -u critical "🔴 Rclone failed"
