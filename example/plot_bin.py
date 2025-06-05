import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_bin(path: str, n1: int, n2: int, cmap: str = "gray") -> None:
    """Load little-endian 32-bit float data and display as image."""
    data = np.fromfile(path, dtype="<f4")
    if data.size != n1 * n2:
        raise ValueError(f"expected {n1*n2} values, got {data.size}")
    data = data.reshape(n1, n2)
    plt.imshow(data, cmap=cmap, aspect="auto")
    plt.title(path)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot binary image file")
    parser.add_argument("path", help="binary file to plot")
    parser.add_argument("--n1", type=int, default=256, help="first dimension")
    parser.add_argument("--n2", type=int, default=256, help="second dimension")
    args = parser.parse_args()
    plot_bin(args.path, args.n1, args.n2)
