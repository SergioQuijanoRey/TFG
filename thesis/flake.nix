{
    description = "Flake to define the Latex enviroment needed for writing the main doc";
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-22.11";
        utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

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

      in {

        # Packages that we use in `nix develop`
        devShell = pkgs.mkShell {
            buildInputs = [
                # Use our custom latex enviroment
                custom_tex_env

                # Build tool for managing latex tasks
                pkgs.just

                # Preferred terminal
                pkgs.zsh

            ];

            shellHook = ''
                # Log that we're in a custom enviroment
                echo "❄️  Running custom dev enviroment with python and other packages"
            '';
        };
      });
}
