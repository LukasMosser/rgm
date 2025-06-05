import numpy as np
from .utility import (
    ricker_wavelet,
    gauss_filt,
    noise_random_mix_2d,
)


class RGM2:
    """Simple 2-D random geological model generator."""

    def __init__(
        self,
        n1: int = 128,
        n2: int = 128,
        nf: int = 4,
        nl: int = 20,
        seed: int | None = None,
        f0: float = 150.0,
    ) -> None:
        self.n1 = n1
        self.n2 = n2
        self.nf = nf
        self.nl = nl
        self.seed = seed
        self.f0 = f0

    def generate(self) -> np.ndarray:
        """Generate a simplified 2-D model."""
        rng = np.random.default_rng(self.seed)
        # base reflectors
        refl = rng.normal(size=(self.n1, self.n2))
        refl = gauss_filt(refl, [20.0, 20.0])
        # convolve with Ricker wavelet along the first axis
        t = np.linspace(-0.1, 0.1, 81)
        wavelet = np.array([ricker_wavelet(tt, self.f0) for tt in t])
        wavelet /= np.sum(np.abs(wavelet))
        pad = len(wavelet) // 2
        padded = np.pad(refl, ((pad, pad), (0, 0)), mode="edge")
        conv = np.apply_along_axis(lambda m: np.convolve(m, wavelet, mode="valid"), 0, padded)
        # add some random mix noise
        noise = noise_random_mix_2d(self.n1, self.n2, seed=rng.integers(0, 2**32 - 1))
        image = conv + 0.25 * noise
        return image
