{ pkgs ? import <nixpkgs> { }
}:
let
  dax = pkgs.callPackage ./default.nix { };
in
(pkgs.python3.withPackages (ps: [
  dax
])).env
