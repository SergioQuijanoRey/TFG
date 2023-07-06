{
    # TODO -- use a proper description for the project!
    description = "Flake to handle some tasks and dependencies for this project";
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let

        python = "python310";
        pkgs = nixpkgs.legacyPackages.${system};

        # Define a custom python enviroment
        custom_python_packages = python-packages: with python-packages; [

            # Notebooks
            jupyter
            jupyterlab

            # Machine/Deep learning libraries
            torch-bin
            torchvision-bin
            scikit-learn

            # Visualizations
            matplotlib
            seaborn
            snakeviz

            # Dataframes
            pandas

            # Basic library for computation
            numpy

            # Hyperparameter tuning
            # TODO -- uncomment
            # optuna

            # Loggin
            wandb
            tqdm

            # LSP Packages
            python-lsp-server
            pyls-isort
            pyls-flake8
            mypy
            isort

            # Debugger used in nvim-dap
            debugpy
        ];
        custom_python_env = pkgs.${python}.withPackages custom_python_packages;

      in {

        # Packages that we use in `nix develop`
        devShell = pkgs.mkShell {
            buildInputs = [
                # Use our custom python enviroment
                custom_python_env

                # Pretty terminal experience
                pkgs.zsh
                pkgs.starship

                # Build tool
                pkgs.just

                # For sync with Google Colab / Google Drive
                pkgs.rsync

            ];

            shellHook = ''
                # Log that we're in a custom enviroment
                echo "❄️  Running custom dev enviroment with python and other packages"
            '';
        };
      });
}
