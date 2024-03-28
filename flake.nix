{
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
          python-dotenv

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

        # Declare the latex enviroment (base enviroment plus additional packages)
        custom_tex_env = (pkgs.texlive.combine {

          # Base latex env
          inherit (pkgs.texlive) scheme-medium

            # Packages that I need for my thesis template to compile
            koma-script
            xpatch
            cabin
            fontaxes
            inconsolata
            xurl
            upquote

            # Extra packages that we want
            amsmath
            hyperref
            cancel
            esvect
            pgf
            tikz-cd
            tikzmark
            todonotes
            cleveref
            ;
        });

        # Packages that we are going to use in both shells, for coding and for
        # writing the Latex thesis
        shared_packages = [
          # Pretty terminal experience
          pkgs.zsh
          pkgs.starship

          # Build tool
          pkgs.just

          # For sync with Google Colab / Google Drive
          pkgs.rsync

          # For launching github actions locally
          pkgs.act
        ];

      in
      {

        # Packages that we use in `nix develop`
        devShells.default = pkgs.mkShell {
          buildInputs = shared_packages ++ [
            # Use our custom python enviroment
            custom_python_env
          ];


          # Add some paths to PYTHONPATH
          PYTHONPATH = "${custom_python_env}/${custom_python_env.sitePackages}:.:./src:./src/lib";

          shellHook = ''
            # Log that we're in a custom enviroment
            echo "❄️  Running custom dev enviroment with python and other packages"
          '';
        };

        # Second devshell that we use when writing the Latex thesis
        # Can be run using `nix develop .#writing`
        devShells.writing = pkgs.mkShell {
          buildInputs = shared_packages ++ [
            # Use our custom latex enviroment
            custom_tex_env
          ];

          shellHook = ''
            # Log that we're in a custom enviroment
            echo "❄️  Running custom dev enviroment with python and other packages"
          '';
        };
      });
}
