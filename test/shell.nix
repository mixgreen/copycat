{ pkgs ? import <nixpkgs> {} }:

let
  artiq-full = import <artiq-full> { inherit pkgs; };
  dax-full = import <dax-full> { inherit pkgs; };
in
  pkgs.mkShell {
    buildInputs = [
      (pkgs.python3.withPackages(ps: [
        # DAX dependencies (not using inputs.nix for compatibility with nix-env)
        artiq-full.artiq
        artiq-full.sipyco
        ps.numpy
        ps.scipy
        ps.pyvcd
        ps.natsort
        ps.pygit2
        ps.matplotlib
        ps.graphviz
        ps.h5py
        ps.networkx
        ps.sortedcontainers
        # Packages required for testing
        ps.pytest
        ps.mypy
        ps.pycodestyle
        ps.coverage
        dax-full.flake8-artiq
        dax-full.artiq-stubs
      ]))
      # Packages required for hardware testbenches
      artiq-full.binutils-or1k
      artiq-full.llvm-or1k
    ];
  }
