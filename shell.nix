{ pkgs ? import <nixpkgs> { } }:
with pkgs;
mkShell {
    buildInputs = [
        # Pretty terminal experience
        zsh
        starship

        python310
        jupyter

        # Python packages
        python310Packages.jupyterlab
        python310Packages.pytorch
        python310Packages.torchvision
        python310Packages.matplotlib
        python310Packages.numpy
        python310Packages.seaborn
        python310Packages.wandb
        python310Packages.scikit-learn
        python310Packages.tqdm
    ];

    # I want to use zsh as my dev shell
    shellHook = ''
        zsh
    '';
}
