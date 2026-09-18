{ pkgs, lib, config, inputs, ... }:

{
  # 1. Environment Variables
  # Set a default IP for the PC to look for (you can override this in a .env file)
  env.ROBOT_IP = "192.168.1.50";

  # manylinux wheels (numpy, scipy, opencv, vosk...) are prebuilt against system
  # libs that nix's loader does not search for, so they fail at import, not install.
  # Empty on macOS: those wheels are Mach-O and DYLD ignores this anyway.
  env.LD_LIBRARY_PATH = lib.makeLibraryPath (lib.optionals pkgs.stdenv.isLinux [
    pkgs.zlib                # libz          -- numpy, scipy, opencv, pillow
    pkgs.stdenv.cc.cc.lib    # libstdc++/libgcc_s/libatomic -- almost everything
    pkgs.glib                # libglib/libgthread -- opencv
    pkgs.libGL               # libGL         -- opencv
    pkgs.xorg.libX11
    pkgs.xorg.libXext
    pkgs.xorg.libSM
    pkgs.xorg.libICE
    pkgs.xorg.libxcb         # opencv's GUI backend, linked even when unused
  ]);

  # 2. System Packages (C-Libraries and Tools)
  packages = [
    pkgs.git
    pkgs.zeromq       # The underlying C-library for fast networking
    pkgs.czmq         # High-level C bindings for ZeroMQ

    # portaudio for Linux aarch64 builds
    pkgs.portaudio
    # For building Ruckig on aarch64 Linux
    pkgs.cmake
  ];

  # 3. Python & Package Management
  languages.python = {
    enable = true;
    version = "3.13"; # Pin your version so your PC and Pi match exactly
    venv.enable = true;
    # Enable the ultra-fast 'uv' package manager
    uv.enable = true;
    uv.sync.enable = true; # Auto-installs pyproject.toml dependencies on boot
  };

  # 4. Pre-commit Hooks (Code Quality)
  # Automatically formats your code so you don't have to think about it
  git-hooks = {
    ruff.enable = true;       # Extremely fast linter
    black.enable = true;      # Standard Python formatter
  };

  # 5. Initialization Script
  # No PYTHONPATH munging needed: cleosdk is installed editable via uv (see
  # pyproject.toml [tool.uv.sources]), and running `python pi_controller/main.py`
  # puts pi_controller/ on sys.path automatically for its `from hal...` imports.
  enterShell = ''
    echo "🤖 Robot Development Environment Loaded"
    echo "Python Version: $(python --version)"
    echo "To run the Pi Controller: python pi_controller/main.py"
    echo "To run an SDK demo:        python -m cleosdk.demos.demo_move"
  '';
}