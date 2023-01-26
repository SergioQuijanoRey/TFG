let

    # Pinning, because last version does not work!
    # Moreover, with pinning we have a more robust and reproducible env
    pkgs = import (builtins.fetchTarball {
        name = "NixOS_Stable_05_September_2022";
        url = "https://github.com/nixos/nixpkgs/archive/7d7622909a38a46415dd146ec046fdc0f3309f44.tar.gz";
        sha256 = "016dcmy9gkmrzd5gsmvz4rd6gh5phf2kg7jf8g2y276v5km50nwf";
    }) {};


    # Python packages need to be wrapped this way
    custom_python_packages = python-packages: with python-packages; [

        # Notebooks
        jupyter
        jupyterlab

        # Machine/Deep learning libraries
        pytorch
        torchvision
        scikit-learn

        # Visualizations
        matplotlib
        seaborn

        numpy
        snakeviz

        # Loggin
        wandb
        tqdm

        # LSP Packages
        python-lsp-server
        pylsp-mypy
        pyls-isort
        pyls-flake8
        mypy
        isort
    ];

    # With the list of packages, create a custom python env
    custom_python_env = pkgs.python310.withPackages custom_python_packages;

in
pkgs.mkShell {
    buildInputs = [
        # Pretty terminal experience
        pkgs.zsh
        pkgs.starship

        # Build tool
        pkgs.just

        # For sync with Google Colab / Google Drive
        pkgs.rsync

        # Custom Python Enviroment
        custom_python_env
    ];

    shellHook = ''
    '';
}
