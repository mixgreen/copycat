{ pkgs ? import <nixpkgs> {}, artiqpkgs ? import <artiq-full> { inherit pkgs; } }:

with pkgs;
python3Packages.buildPythonPackage rec {
  pname = "dax";
  version = import ./version.nix { inherit stdenv git src; };

  src = nix-gitignore.gitignoreSource [ "*.nix" ] ./.;

  VERSIONEER_OVERRIDE = version;

  propagatedBuildInputs = (with artiqpkgs; [ artiq ])
    ++ (with python3Packages; [ numpy scipy pyvcd natsort pygit2 matplotlib graphviz h5py networkx ]);

  checkInputs = with python3Packages; [ pytestCheckHook ];

  inherit (python3Packages.pygit2) SSL_CERT_FILE;

  meta = with stdenv.lib; {
    description = "Duke ARTIQ Extensions (DAX)";
    maintainers = [ "Duke University" ];
    homepage = "https://gitlab.com/duke-artiq/dax";
    license = licenses.asl20;
  };
}
