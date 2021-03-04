{ stdenv, git, src }:
let
  dax-version = stdenv.mkDerivation {
    name = "dax-version";
    src = src;
    buildPhase = ''
      # mimic versioneer version string: remove version prefix ('v'),
      # replace first '-' with a '+' and other '-'es with a '.'
      VERSION=`${git}/bin/git describe --tags --always --dirty | sed '1s/^v//' | sed '0,/-/s//+/' | tr - .`
    '';
    installPhase = ''
      echo -n $VERSION > $out
    '';
  };
in
  builtins.readFile dax-version
