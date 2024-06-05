import matplotlib.pyplot as plt
import numpy as np
import skimage

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2gray
from imagetools.signals import (
    ifft1c,
    fft1c,
    iterative_soft_thresholding,
    dwt2,
    idwt2,
)

# L1 and L2 Regression
# Import data
with open("data/y_line.txt") as f:
    file_list = f.read().split("\n")[:-1]
    y_noisy = np.array([float(x) for x in file_list])

with open("data/y_outlier_line.txt") as f:
    file_list = f.read().split("\n")[:-1]
    y_outlier = np.array([float(x) for x in file_list])

x = np.arange(len(y_noisy))

# L2 optimisation
a_noisy_l2 = np.sum(x * y_noisy) / np.sum(x * x)
b_noisy_l2 = np.mean(y_noisy) - a_noisy_l2 * np.mean(x)

a_outlier_l2 = np.sum(x * y_outlier) / np.sum(x * x)
b_outlier_l2 = np.mean(y_outlier) - a_outlier_l2 * np.mean(x)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(x, y_noisy)
ax[0].plot(x, a_noisy_l2 * x + b_noisy_l2)
ax[1].scatter(x, y_outlier)
ax[1].plot(x, a_outlier_l2 * x + b_outlier_l2)
plt.savefig("outputs/regression_l2.png")

# Create a sparse signal with small Gaussian noise.
rng = np.random.default_rng(seed=42)
signal = np.zeros(100)
signal[:5] = np.ones(5)
signal[5:10] = -np.ones(5)
signal = signal[rng.permutation(100)]
signal = signal + rng.normal(0, 0.05, 100)

# Data (y) is FT of signal (x)
data = fft1c(signal)

# Create sampling masks
sample_freq = 4
mask_random = rng.uniform(0, 1, 100)
# Using quantiles ensures the desired number of samples is taken
mask_random = mask_random < np.quantile(mask_random, 1.0 / sample_freq)
mask_unif = np.zeros(100)
start_index = rng.integers(0, sample_freq)
mask_unif[start_index::sample_freq] = 1

# Sample the true data
sample_random = mask_random * data
sample_unif = mask_unif * data

# IFT to get gt/noisy signal and normalise by sampling pdf at each point
# (which is constant)
signal_random = ifft1c(sample_random) * sample_freq
signal_unif = ifft1c(sample_unif) * sample_freq

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0, 0].plot(abs(data))
ax[0, 0].set_title("measurements")
ax[0, 1].plot(abs(sample_random))
ax[0, 2].plot(abs(sample_unif))
ax[1, 0].plot(abs(signal))
ax[1, 0].set_title("signal")
ax[1, 1].plot(abs(signal_random))
ax[1, 2].plot(abs(signal_unif))
plt.savefig("outputs/signal.png")

data_random_recon, mse_random = iterative_soft_thresholding(
    sample_random, lam=5e-3, n_iters=100, gt=signal
)
data_unif_recon, mse_unif = iterative_soft_thresholding(
    sample_unif, lam=1e-3, n_iters=100, gt=signal
)

signal_random_recon = ifft1c(data_random_recon) * sample_freq
signal_unif_recon = ifft1c(data_unif_recon) * sample_freq

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].plot(abs(signal_random_recon), label="random")
ax[0, 0].plot(abs(signal), label="gt")
ax[0, 0].legend()
ax[0, 1].plot(abs(signal_unif_recon), label="unif")
ax[0, 1].plot(abs(signal), label="gt")
ax[0, 1].legend()
ax[1, 0].plot(mse_random)
ax[1, 1].plot(mse_unif)
plt.savefig("outputs/signal_reconstruct.png")

# Image Compression via Wavelet Decomposition

# Read image
river_img = skimage.io.imread("data/river_side.jpeg")
river_img = rgb2gray(river_img)

# Wavelet transform, and then the inverse
river_img_dw = dwt2(river_img)
river_img_recon = idwt2(river_img_dw)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(river_img, cmap="grey")
ax[0].axis("off")
ax[1].imshow(river_img_recon, cmap="grey")
ax[1].axis("off")
ax[2].imshow(abs(river_img_recon - river_img))
ax[2].axis("off")
plt.savefig("outputs/river_img.png")

for fr in [0.2, 0.15, 0.1, 0.05, 0.025]:
    river_img_dw = dwt2(river_img)
    th = np.quantile(river_img_dw, 1 - fr)
    river_img_dw = np.clip(river_img_dw, th, None)
    river_img_recon = idwt2(river_img_dw)
    fig, ax = plt.subplots(1, 3, figsize=(15, 20))
    ax[0].imshow(river_img, cmap="grey")
    ax[0].axis("off")

    ax[1].imshow(river_img_recon, cmap="grey")
    ax[1].axis("off")
    ax[1].set_title(
        f"Top {fr*100}% psnr={psnr(river_img, river_img_recon):.2f}"
    )

    ax[2].imshow(abs(river_img_recon - river_img))
    ax[2].axis("off")
    plt.savefig(f"outputs/river_img_compressed_{fr:.3f}.png")
