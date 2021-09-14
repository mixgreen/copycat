{ pkgs ? import <nixpkgs> {}, artiqpkgs ? import <artiq-full> { inherit pkgs; } }:

let
  artiqDependencies = (with artiqpkgs;
    [ artiq sipyco ]
  );
  pkgsDependencies = (with pkgs.python3Packages;
    [ numpy scipy pyvcd natsort pygit2 matplotlib graphviz h5py networkx sortedcontainers ]
  );
in
  artiqDependencies ++ pkgsDependencies
