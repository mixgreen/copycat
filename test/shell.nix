{ pkgs ? import <nixpkgs> {} }:

let
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
                ps.networkx
                # Packages required for testing
                ps.pytest
                ps.mypy
                ps.pycodestyle
                ps.coverage
                artiq-full.flake8-artiq
            ]))
            # Packages required for hardware testbenches
            artiq-full.binutils-or1k
            artiq-full.llvm-or1k
        ];
    }
