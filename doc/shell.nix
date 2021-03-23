{ pkgs ? import <nixpkgs> {} }:

let
  artiq-full = import <artiq-full> { inherit pkgs; };
in
  pkgs.mkShell {
    buildInputs = [
      (pkgs.python3.withPackages(ps: [
        # DAX dependencies
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
        # Packages required for documentation
        ps.sphinx
        ps.sphinx_rtd_theme
      ]))
      pkgs.git  # Required to set the correct copyright year
      pkgs.gnumake
    ];
  }
