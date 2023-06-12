{ pkgs ? import <nixpkgs> { }
, artiqpkgs ? import <artiq-full> { inherit pkgs; }
, daxVersion ? null
, trap-dac-utils ? import ./trap-dac-utils.nix { inherit pkgs; }
}:

with pkgs;
python3Packages.buildPythonPackage rec {
  pname = "dax";
  version = if daxVersion == null then import ./version.nix { inherit pkgs src; } else daxVersion;

  src = nix-gitignore.gitignoreSource [ "*.nix" ] ./.;

  VERSIONEER_OVERRIDE = version;
  inherit (python3Packages.pygit2) SSL_CERT_FILE;

  propagatedBuildInputs = import ./inputs.nix { inherit pkgs artiqpkgs trap-dac-utils; } python3Packages;

  checkInputs = with python3Packages; [ pytestCheckHook ];

  condaDependencies = [
    "python>=3.7"
    "trap-dac-utils"
    "artiq"
    "sipyco"
    "numpy"
    "scipy"
    "pyvcd"
    "natsort"
    "pygit2"
    "matplotlib"
    "python-graphviz"
    "h5py"
    "networkx"
    "sortedcontainers"
    "libffi=3.3" # Limit version to prevent broken environment
    "llvmlite<0.40.0" # Limit version, see https://gitlab.com/duke-artiq/dax/-/issues/144
  ];

  meta = with lib; {
    description = "Duke ARTIQ Extensions (DAX)";
    maintainers = [ "Duke University" ];
    homepage = "https://gitlab.com/duke-artiq/dax";
    license = licenses.mit;
  };
}
