#!/usr/bin/env bash
set -euo pipefail
source /opt/intel/oneapi/setvars.sh

# Clone and build FLIT (first‑time only – 2‑3 min on medium machine)
if [ ! -d "$HOME/flit" ]; then
  git clone --depth=1 https://github.com/lanl/flit.git "$HOME/flit"
  pushd "$HOME/flit/src"
  make -j$(nproc)
  popd
fi

# Build RGM library
pushd src && make -j$(nproc) && popd

# Build the sample executables
pushd example && make -j$(nproc) && popd