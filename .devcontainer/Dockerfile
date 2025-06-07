# Jammy (22.04) Codespaces base – includes VS Code server & common utilities
FROM mcr.microsoft.com/devcontainers/base:0-ubuntu-22.04

# ---- Intel oneAPI Base & HPC toolkits (ifx / icx / icpx) ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget gnupg lsb-release build-essential make git ca-certificates pkg-config linux-headers-$(uname -r) && \
    wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
        gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
        tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        intel-basekit intel-hpckit && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Auto‑source compiler vars for every interactive shell
RUN echo "source /opt/intel/oneapi/setvars.sh" >> /etc/bash.bashrc