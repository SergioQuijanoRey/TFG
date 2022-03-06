# End of Grade Thesis (TFG Trabajo Fin de Grado)

- Repository where I develop my end of grade thesis

## Contact

- Sergio Quijano Rey
- [mail](sergioquijanorey@protonmail.com)
- [Personal Webpage](https://sergioquijanorey.github.io/)

## Pre-requisites

Using poetry works fine through the command line, but `pyproject.toml` does not reflect all packages installed.

In order to install pytorch, I had to follow next steps:

1. Install python3.8.12 on my system (Arch Linux)
2. Change poetry env to python3.8.12
    - `which python3.8.12`
    - Use that ouput (`<path>`) and then `poetry env use <path>`
3. Install pytorch using pip inside local env:
    - Go to [official website](https://pytorch.org/get-started/locally/) to see which comamnd I have to execute
    - Then, run ` poetry pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html` as offial docs suggest
    - This is a problem because the execution of such command does not reflect on pyproject.toml

