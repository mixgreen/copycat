{ pkgs ? import <nixpkgs> { } }:

let
  artiq-full = import <artiq-full> { inherit pkgs; };
  dax-full = import <dax-full> { inherit pkgs; };
  daxInputs = import ../inputs.nix { inherit pkgs; artiqpkgs = artiq-full; };
in
pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: (daxInputs ps) ++ [
      # Packages required for testing
      ps.pytest
      ps.mypy
      ps.pycodestyle
      ps.coverage
      dax-full.trap-dac-utils
      dax-full.flake8-artiq
      dax-full.artiq-stubs
    ]))
    # Packages required for hardware testbenches
    artiq-full.binutils-or1k
    artiq-full.llvm-or1k
  ];
}
