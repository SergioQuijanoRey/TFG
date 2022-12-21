let
    # Pinning, because last version of the texlive enviroment is broken
    pkgs = import (builtins.fetchTarball {
        name = "NixOS_Stable_05_September_2022";
        url = "https://github.com/nixos/nixpkgs/archive/7d7622909a38a46415dd146ec046fdc0f3309f44.tar.gz";
        sha256 = "016dcmy9gkmrzd5gsmvz4rd6gh5phf2kg7jf8g2y276v5km50nwf";
    }) {};

    # Declare the latex enviroment (base enviroment plus additional packages)
    custom_tex_env = (pkgs.texlive.combine {
        # Base latex env
        inherit (pkgs.texlive) scheme-medium

        # Extra packages that we want
        amsmath
        hyperref
        cancel
        esvect

        # Packages that I need for my thesis template to compile
        koma-script
        xpatch
        cabin
        fontaxes
        inconsolata
        xurl
        upquote
        ;
    });
in
pkgs.mkShell {
    buildInputs = with pkgs; [
        # TODO -- remove this dependency
        zsh

        # Build tool for managing latex tasks
        just

        # Use our latex enviroment
        custom_tex_env
    ];

    shellHook = ''
    '';
}
