{ pkgs }:

let
  trapDacUtilsSrc = builtins.fetchGit { url = "https://gitlab.com/duke-artiq/trap-dac-utils.git"; };
  trap-dac-utils = pkgs.callPackage (import trapDacUtilsSrc) { inherit pkgs; };
in
trap-dac-utils
