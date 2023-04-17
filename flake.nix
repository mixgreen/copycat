{
  inputs = {
    artiqpkgs.url = git+https://github.com/m-labs/artiq?ref=release-7;
    nixpkgs.follows = "artiqpkgs/nixpkgs";
    sipyco.follows = "artiqpkgs/sipyco";
    flake8-artiq = {
      url = git+https://gitlab.com/duke-artiq/flake8-artiq.git;
      inputs.artiqpkgs.follows = "artiqpkgs";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    artiq-stubs = {
      url = git+https://gitlab.com/duke-artiq/artiq-stubs.git;
      inputs.artiqpkgs.follows = "artiqpkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake8-artiq.follows = "flake8-artiq";
    };
    trap-dac-utils = {
      url = git+https://gitlab.com/duke-artiq/trap-dac-utils.git;
      inputs.artiqpkgs.follows = "artiqpkgs";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, artiqpkgs, sipyco, flake8-artiq, artiq-stubs, trap-dac-utils }:
    let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
      # get major version from ARTIQ
      daxVersionBase = builtins.elemAt (builtins.splitVersion artiqpkgs.packages.x86_64-linux.artiq.version) 0;
      daxVersion = daxVersionBase + "." + (builtins.toString self.sourceInfo.revCount or 0) + "." + (self.sourceInfo.shortRev or "unknown");
      dax = (with pkgs; python3Packages.buildPythonPackage rec {
        pname = "dax";
        version = daxVersion;

        src = nix-gitignore.gitignoreSource [ "*.nix" ] ./.;

        VERSIONEER_OVERRIDE = version;
        inherit (python3Packages.pygit2) SSL_CERT_FILE;

        propagatedBuildInputs = (
          (with python3Packages; [ numpy scipy pyvcd natsort pygit2 matplotlib graphviz h5py networkx sortedcontainers ]) ++
          [ artiqpkgs.packages.x86_64-linux.artiq sipyco.packages.x86_64-linux.sipyco trap-dac-utils.packages.x86_64-linux.trap-dac-utils ]
        );

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
        ];

        meta = with lib; {
          description = "Duke ARTIQ Extensions (DAX)";
          maintainers = [ "Duke University" ];
          homepage = "https://gitlab.com/duke-artiq/dax";
          license = licenses.asl20;
        };
      });
    in
    rec {
      packages.x86_64-linux = {
        inherit dax;
        flake8-artiq = flake8-artiq.packages.x86_64-linux.flake8-artiq;
        artiq-stubs = artiq-stubs.packages.x86_64-linux.artiq-stubs;
        trap-dac-utils = trap-dac-utils.packages.x86_64-linux.trap-dac-utils;
        default = pkgs.python3.withPackages (ps: [ dax ]);
      };
      # shells for `nix develop`
      devShells.x86_64-linux = {
        default = pkgs.mkShell {
          name = "dax-dev-shell";
          buildInputs = [
            (pkgs.python3.withPackages (ps:
              # basic environment
              dax.propagatedBuildInputs ++
              # test dependencies
              (with ps; [ pytest mypy pycodestyle coverage autopep8 ]) ++
              ([ packages.x86_64-linux.flake8-artiq packages.x86_64-linux.artiq-stubs ])
            ))
            # required for compile/hardware testcases
            pkgs.unixtools.ping
            pkgs.lld_11
            pkgs.llvm_11
          ];
        };
        docs = pkgs.mkShell {
          name = "docs-dev-shell";
          buildInputs = [
            (pkgs.python3.withPackages (ps:
              # basic environment
              dax.propagatedBuildInputs ++
              # Packages required for documentation
              [ ps.sphinx ps.sphinx_rtd_theme ]
            ))
            pkgs.git # Required to set the correct copyright year
            pkgs.gnumake
          ];
        };
      };
      # enables use of `nix fmt`
      formatter.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.nixpkgs-fmt;
    };

  nixConfig = {
    extra-trusted-public-keys = [
      "nixbld.m-labs.hk-1:5aSRVA5b320xbNvu30tqxVPXpld73bhtOeH6uAjRyHc="
    ];
    extra-substituters = [ "https://nixbld.m-labs.hk" ];
  };
}
