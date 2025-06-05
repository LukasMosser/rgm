import numpy as np


def add_faults(
    image: np.ndarray,
    nf: int = 1,
    dip_range: tuple[float, float] = (60.0, 120.0),
    disp_range: tuple[float, float] = (-10.0, 10.0),
    seed: int | None = None,
) -> np.ndarray:
    """Insert simple linear faults into an image.

    This is a lightweight approximation of the Fortran fault generator.
    Faults are represented by straight lines with a constant displacement
    applied to one side of the line.
    """
    rng = np.random.default_rng(seed)
    result = np.copy(image)
    n1, n2 = result.shape
    for _ in range(nf):
        angle = np.radians(rng.uniform(*dip_range))
        disp = rng.uniform(*disp_range)
        x0 = rng.uniform(0, n2)
        z0 = rng.uniform(0, n1)
        k = int(round(disp))
        line = np.tan(angle) * (np.arange(n2) - x0) + z0
        for j, i in enumerate(line.astype(int)):
            if 0 <= i < n1:
                if k > 0:
                    seg = result[i:, j]
                    result[i:, j] = np.roll(seg, k)
                    result[i:i + k, j] = seg[0]
                elif k < 0:
                    seg = result[: i + 1, j]
                    result[: i + 1, j] = np.roll(seg, k)
                    result[i + k + 1 : i + 1, j] = seg[-1]
    return result
