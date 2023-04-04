#!/bin/bash


#SBATCH --job-name TripletLoss    # Nombre del proceso
#SBATCH --partition dios    	  # Cola para ejecutar
#SBATCH --gres=gpu:1              # Numero de gpus a usar

# Setup conda
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/squijano/TFG/custom_env_squijano
export TFHUB_CACHE_DIR=.

# Launch the python process
MAIN_FILE="./src/LFW Notebook.py"
python "$MAIN_FILE" || mail -s "Proceso fallido" sergioquijano@correo.ugr.es <<< "El proceso no se ha ejecutado correctamente"

# Notify process has finished succesfully
mail -s "Proceso finalizado" sergioquijano@correo.ugr.es <<< "El proceso ha finalizado"
