{ pkgs ? import <nixpkgs> { } }:
with pkgs;
let

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
mkShell {
    buildInputs = [
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
