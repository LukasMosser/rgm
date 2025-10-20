import numpy as np

__all__ = [
    "sinc_wavelet",
    "gaussian_wavelet",
    "gaussian_deriv_wavelet",
    "gaussian_wavelet_hdur",
    "ricker_wavelet",
    "ricker_deriv_wavelet",
    "ormsby_wavelet",
    "noise_wavenumber_2d",
    "noise_wavenumber_3d",
    "noise_random_mix_2d",
    "noise_random_mix_3d",
    "random_mask_smooth_2d",
    "random_mask_smooth_3d",
    "elastic_reflection_coefs",
]


PI = np.pi


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(int(3 * sigma), 1)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def _convolve_along_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = [(0, 0)] * arr.ndim
    radius = len(kernel) // 2
    pad[axis] = (radius, radius)
    padded = np.pad(arr, pad, mode="edge")
    convolved = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis, padded)
    slices = [slice(radius, -radius) if i == axis else slice(None) for i in range(arr.ndim)]
    return convolved[tuple(slices)]


def gauss_filt(arr: np.ndarray, sigma) -> np.ndarray:
    sigma = np.atleast_1d(sigma).astype(float)
    if arr.ndim != sigma.size:
        sigma = np.resize(sigma, arr.ndim)
    out = arr.astype(float)
    for ax, sig in enumerate(sigma):
        if sig > 0:
            kern = _gaussian_kernel(sig)
            out = _convolve_along_axis(out, kern, ax)
    return out


def sinc_wavelet(t: float, f0: float) -> float:
    return np.sinc(PI * f0 * t / PI)


def gaussian_wavelet(t: float, f0: float) -> float:
    return np.exp(-(PI * f0 * t) ** 2)


def gaussian_wavelet_hdur(hdur: float, t: float) -> float:
    a = 1.0 / hdur ** 2
    return np.exp(-a * t ** 2) / (np.sqrt(PI) * hdur)


def gaussian_deriv_wavelet(t: float, f0: float) -> float:
    return -t * np.exp(-(PI * f0 * t) ** 2)


def ricker_wavelet(t: float, f0: float) -> float:
    val = (PI * f0 * t) ** 2
    return (1 - 2.0 * val) * np.exp(-val)


def ricker_deriv_wavelet(t: float, f0: float) -> float:
    val = (PI * f0 * t) ** 2
    return (2.0 * val - 3.0) * np.exp(-val) * t


def ormsby_wavelet(f: np.ndarray, t: float) -> float:
    f1, f2, f3, f4 = f
    return PI * (
        f4 ** 2 / (f4 - f3) * np.sinc(PI * f4 * t / PI) ** 2
        - f3 ** 2 / (f4 - f3) * np.sinc(PI * f3 * t / PI) ** 2
        - f2 ** 2 / (f2 - f1) * np.sinc(PI * f2 * t / PI) ** 2
        + f1 ** 2 / (f2 - f1) * np.sinc(PI * f1 * t / PI) ** 2
    )


def noise_wavenumber_2d(w: np.ndarray, level=0.5, smooth=(2.0, 2.0), seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.random(w.shape)
    r = gauss_filt(r, smooth)
    r = (r - r.min()) / (r.max() - r.min())
    r[r < level] = 0.0
    wt = np.fft.ifft2(np.fft.fft2(w) * r).real - w
    return wt


def noise_wavenumber_3d(w: np.ndarray, level=0.5, smooth=(2.0, 2.0, 2.0), seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.random(w.shape)
    r = gauss_filt(r, smooth)
    r = (r - r.min()) / (r.max() - r.min())
    r[r < level] = 0.0
    wt = np.fft.ifftn(np.fft.fftn(w) * r).real - w
    return wt


def noise_random_mix_2d(n1: int, n2: int, level=None, smooth=None, seed=None) -> np.ndarray:
    if level is None:
        level = [9.0, 3.0, 1.0]
    if smooth is None:
        smooth = [9.0, 3.0, 0.0]
    rng = np.random.default_rng(seed)
    w = np.zeros((n1, n2), dtype=float)
    for i, amp in enumerate(level):
        r = rng.normal(size=(n1, n2))
        r = gauss_filt(r, [smooth[i], smooth[i]])
        r -= r.mean()
        w += r / np.max(np.abs(r)) * amp
    w /= np.max(np.abs(w))
    return w


def noise_random_mix_3d(n1: int, n2: int, n3: int, level=None, smooth=None, seed=None) -> np.ndarray:
    if level is None:
        level = [9.0, 3.0, 1.0]
    if smooth is None:
        smooth = [9.0, 3.0, 0.0]
    rng = np.random.default_rng(seed)
    w = np.zeros((n1, n2, n3), dtype=float)
    for i, amp in enumerate(level):
        r = rng.normal(size=(n1, n2, n3))
        r = gauss_filt(r, [smooth[i], smooth[i], smooth[i]])
        r -= r.mean()
        w += r / np.max(np.abs(r)) * amp
    w /= np.max(np.abs(w))
    return w


def random_mask_smooth_2d(n1: int, n2: int, gs, mask_out: float, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(n1, n2))
    r = gauss_filt(r, gs)
    r = (r - r.min()) / (r.max() - r.min())
    ratio = 1.0
    bv = 1.0
    while ratio > mask_out:
        ratio = np.count_nonzero(r <= bv) / (n1 * n2)
        bv -= 0.001
    r = np.where(r <= bv, 0.0, 1.0)
    return r


def random_mask_smooth_3d(n1: int, n2: int, n3: int, gs, mask_out: float, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(n1, n2, n3))
    r = gauss_filt(r, gs)
    r = (r - r.min()) / (r.max() - r.min())
    ratio = 1.0
    bv = 1.0
    while ratio > mask_out:
        ratio = np.count_nonzero(r <= bv) / (n1 * n2 * n3)
        bv -= 0.001
    r = np.where(r <= bv, 0.0, 1.0)
    return r


def elastic_reflection_coefs(p, a1, b1, rho1, a2, b2, rho2):
    a = rho2 * (1 - 2 * b2 ** 2 * p ** 2) - rho1 * (1 - 2 * b1 ** 2 * p ** 2)
    b = rho2 * (1 - 2 * b2 ** 2 * p ** 2) + 2 * rho1 * b1 ** 2 * p ** 2
    c = rho1 * (1 - 2 * b1 ** 2 * p ** 2) + 2 * rho2 * b2 ** 2 * p ** 2
    d = 2 * (rho2 * b2 ** 2 - rho1 * b1 ** 2)
    i1 = np.arcsin(np.minimum(p * a1, 1.0))
    i2 = np.arcsin(np.minimum(p * a2, 1.0))
    j1 = np.arcsin(np.minimum(p * b1, 1.0))
    j2 = np.arcsin(np.minimum(p * b2, 1.0))
    e = b * np.cos(i1) / a1 + c * np.cos(i2) / a2
    f = b * np.cos(j1) / b1 + c * np.cos(j2) / b2
    g = a - d * np.cos(i1) / a1 * np.cos(j2) / b2
    h = a - d * np.cos(i2) / a2 * np.cos(j1) / b1
    dd = e * f + g * h * p ** 2
    if dd == 0:
        dd = 1e-10
    rpp = (((b * np.cos(i1) / a1) - c * np.cos(i2) / a2) * f - (a + d * np.cos(i1) / a1 * np.cos(j2) / b2) * h * p ** 2) / dd
    rss = -(((b * np.cos(j1) / b1) - c * np.cos(j2) / b2) * e - (a + d * np.cos(i2) / a2 * np.cos(j1) / b1) * g * p ** 2) / dd
    rps = -2 * np.cos(i1) / b1 * (a * b + c * d * np.cos(i2) / a2 * np.cos(j2) / b2) * p / dd
    rsp = -2 * np.cos(j1) / a1 * (a * b + c * d * np.cos(i2) / a2 * np.cos(j2) / b2) * p / dd
    return np.array([rpp, rps, rsp, rss])
