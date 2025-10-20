#!/usr/bin/env bash
set -euo pipefail
# /etc/bash.bashrc already sources oneAPI vars.

# Clone & build FLIT (first run only – ~2 min on a medium machine)
if [ ! -d "$HOME/flit" ]; then
  git clone --depth=1 https://github.com/lanl/flit.git "$HOME/flit"
  make -C "$HOME/flit/src" -j$(nproc)
fi

# Build RGM and its examples
make -C src     -j$(nproc)
make -C example -j$(nproc)