{ pkgs ? import <nixpkgs> { } }:

let
  artiq-full = import <artiq-full> { inherit pkgs; };
  dax-full = import <dax-full> { inherit pkgs; };
  daxInputs = import ../inputs.nix { inherit pkgs; artiqpkgs = artiq-full; inherit (dax-full) trap-dac-utils; };
in
pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: (daxInputs ps) ++ [
      # Packages required for documentation
      ps.sphinx
      ps.sphinx_rtd_theme
    ]))
    pkgs.git # Required to set the correct copyright year
    pkgs.gnumake
  ];
}
