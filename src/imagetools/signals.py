import numpy as np
import pywt


# Functions taken from or adapted from the helper.py module
def fft1c(x):
    return (
        1
        / np.sqrt(np.prod(x.shape))
        * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))
    )


def ifft1c(y):
    return np.sqrt(np.prod(y.shape)) * np.fft.ifftshift(
        np.fft.ifft(np.fft.fftshift(y))
    )


def ComplexSoftThresh(y, lam):
    res = abs(y) - lam
    cst = (res > 0.0) * res * y / abs(y)
    return cst


def iterative_soft_thresholding(
    data_sampled, lam, n_iters, sample_freq, gt=None
):
    data = data_sampled.copy()
    mse_values = []
    for i in range(n_iters):
        signal = ifft1c(data)
        if not (gt is None):
            mse_values.append(np.linalg.norm(data - gt))
        signal = ComplexSoftThresh(signal, lam=lam)  # Threshold
        data = fft1c(signal)
        # Data consistency step for measurements
        data = data * (data_sampled == 0) + data_sampled
    return data, mse_values


def coeffs2img(LL, coeffs):
    LH, HL, HH = coeffs
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))


def img2coeffs(Wim, levels=4):
    LL, c = unstack_coeffs(Wim)
    coeffs = [c]
    for i in range(levels - 1):
        LL, c = unstack_coeffs(LL)
        coeffs.insert(0, c)
    coeffs.insert(0, LL)
    return coeffs


def unstack_coeffs(Wim):
    L1, L2 = np.hsplit(Wim, 2)
    LL, HL = np.vsplit(L1, 2)
    LH, HH = np.vsplit(L2, 2)
    return LL, [LH, HL, HH]


def dwt2(im):
    coeffs = pywt.wavedec2(im, wavelet="db4", mode="per", level=4)
    Wim, rest = coeffs[0], coeffs[1:]
    for levels in rest:
        Wim = coeffs2img(Wim, levels)
    return Wim


def idwt2(Wim):
    coeffs = img2coeffs(Wim, levels=4)
    return pywt.waverec2(coeffs, wavelet="db4", mode="per")
