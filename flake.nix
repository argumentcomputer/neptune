{
  description = "Neptune";
  inputs = {
    nixpkgs.url = github:nixos/nixpkgs;
    flake-utils = {
      url = github:numtide/flake-utils;
      inputs.nixpkgs.follows = "nixpkgs";
    };
    naersk = {
      url = github:nix-community/naersk;
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    , naersk
    }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      lib = nixpkgs.lib.${system};
      pkgs = nixpkgs.legacyPackages.${system};
      rustTools = import ./nix/rust.nix {
        nixpkgs = pkgs;
      };
      getRust =
        { channel ? "nightly"
        , date
        , sha256
        , targets ? [
          "wasm32-unknown-unknown"
          "wasm32-wasi"
          # "wasm32-unknown-emscripten"
        ]
        }: (rustTools.rustChannelOf {
          inherit channel date sha256;
        }).rust.override {
          inherit targets;
          extensions = [ "rust-src" "rust-analysis" ];
        };
      rust = getRust { date = "2022-03-15"; sha256 = "sha256-C7X95SGY0D7Z17I8J9hg3z9cRnpXP7FjAOkvEdtB9nE="; };
      # Get a naersk with the input rust version
      naerskWithRust = rust: naersk.lib."${system}".override {
        rustc = rust;
        cargo = rust;
      };
      # Naersk using the default rust version
      buildRustProject = pkgs.makeOverridable ({ rust, naersk ? naerskWithRust rust, ... } @ args: naersk.buildPackage ({
        buildInputs = with pkgs; [ ];
        targets = [ ];
        copyLibs = true;
        remapPathPrefix =
          true; # remove nix store references for a smaller output package
      } // args));

      # Convenient for running tests
      testRustProject = args: buildRustProject ({ doCheck = true; inherit root rust; } // args);
      # Load a nightly rust. The hash takes precedence over the date so remember to set it to
      # something like `lib.fakeSha256` when changing the date.
      crateName = "neptune";
      root = ./.;
      # This is a wrapper around naersk build
      # Remember to add Cargo.lock to git for naersk to work
      project = buildRustProject {
        inherit root rust;
      };
      wasm = project.override {
        targets = [ "wasm32-unknown-unknown" ];
        CC = "${pkgs.emscripten}/bin/emcc";
        copyBins = true;
        cargoBuildOptions = opt: opt ++ [ "--features wasm" "--no-default-features" ];
      };
    in
    {
      packages = {
        ${crateName} = project;
        "${crateName}-wasm" = wasm;
        "${crateName}-test" = testRustProject {};
      };

      defaultPackage = self.packages.${system}.${crateName};

      # `nix develop`
      devShell = pkgs.mkShell {
        inputsFrom = builtins.attrValues self.packages.${system};
        nativeBuildInputs = [ rust ];
        buildInputs = with pkgs; [
          rust-analyzer
          clippy
          rustfmt
          emscripten
        ];
      };
    });
}
