{ pkgs ? import <nixpkgs> { } }:
with pkgs;
mkShell {
    buildInputs = [
        # Pretty terminal experience
        zsh
        starship

        # Build tool
        just

        # Python packages
        python310
        jupyter
        python310Packages.jupyterlab
        python310Packages.pytorch
        python310Packages.torchvision
        python310Packages.matplotlib
        python310Packages.numpy
        python310Packages.seaborn
        python310Packages.wandb
        python310Packages.scikit-learn
        python310Packages.tqdm
        python310Packages.snakeviz
    ];

    shellHook = ''
    '';
}
