{
    description = "Flake to handle some tasks for this project";
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        utils.url = "github:numtide/flake-utils";

    };

    outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        python = "python39";
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            # dev packages
            (pkgs.${python}.withPackages
              (ps: with ps; [

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

        ]))

        pkgs.# Pretty terminal experience
        pkgs.zsh
        pkgs.starship

        pkgs.# Build tool
        pkgs.just

        pkgs.# For sync with Google Colab / Google Drive
        pkgs.rsync

        # Custom Python Enviroment
        # custom_python_env

          ];
        };
      });
}
