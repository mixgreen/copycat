{ pkgs ? import <nixpkgs> {} }:

let
  artiq-full = import <artiq-full> { inherit pkgs; };
  daxInputs = import ../inputs.nix { inherit pkgs; artiqpkgs = artiq-full; };
in
  pkgs.mkShell {
    buildInputs = daxInputs ++ [
      (pkgs.python3.withPackages(ps: [
        # Packages required for documentation
        ps.sphinx
        ps.sphinx_rtd_theme
      ]))
      pkgs.git  # Required to set the correct copyright year
      pkgs.gnumake
    ];
  }
