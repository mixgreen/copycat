let
    pkgs = import <nixpkgs> {};
    artiq-full = import <artiq-full> { inherit pkgs; };
in
    pkgs.mkShell {
        buildInputs = [
            (pkgs.python3.withPackages(ps: [
                # ARTIQ environment
                artiq-full.artiq
                # DAX dependencies
                ps.numpy
                ps.scipy
                ps.pyvcd
                ps.natsort
                ps.pygit2
                ps.matplotlib
                ps.graphviz
                ps.h5py
                # Packages required for documentation
                ps.sphinx
                ps.sphinx_rtd_theme
            ]))
            pkgs.git  # Required to set the correct copyright year
        ];
    }
