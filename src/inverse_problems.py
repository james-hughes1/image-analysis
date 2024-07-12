"""!@file inverse_problems.py
@brief Script running code for questions 2.1, 2.2, 2.3, and 3.1.

@details This script solves various inverse problems related to signal and
image processing, employing various iterative strategies and signal
transforms.
@author Created by J. Hughes on 8th June 2024.
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage

from skimage.color import rgb2gray
from imagetools.signals import (
    ifft1c,
    fft1c,
    iterative_soft_thresholding,
    dwt2,
    idwt2,
)
from imagetools.ml import gradient_descent
from imagetools.plotting import plot_image

# --- L1 and L2 Regression --- #
print("Performing regression tasks...")
# Import data
with open("data/y_line.txt") as f:
    file_list = f.read().split("\n")[:-1]
    y_noisy = np.array([float(x) for x in file_list])

with open("data/y_outlier_line.txt") as f:
    file_list = f.read().split("\n")[:-1]
    y_outlier = np.array([float(x) for x in file_list])

x = np.arange(len(y_noisy))

# L2 optimisation
a_noisy_l2 = np.sum((x - np.mean(x)) * (y_noisy - np.mean(y_noisy))) / np.sum(
    (x - np.mean(x)) ** 2
)
b_noisy_l2 = np.mean(y_noisy) - a_noisy_l2 * np.mean(x)

a_outlier_l2 = np.sum(
    (x - np.mean(x)) * (y_outlier - np.mean(y_outlier))
) / np.sum((x - np.mean(x)) ** 2)
b_outlier_l2 = np.mean(y_outlier) - a_outlier_l2 * np.mean(x)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(x, y_noisy)
ax[0].plot(x, a_noisy_l2 * x + b_noisy_l2)
ax[0].set_title(f"a={a_noisy_l2:.4f} b={b_noisy_l2:.4f}")
ax[1].scatter(x, y_outlier)
ax[1].plot(x, a_outlier_l2 * x + b_outlier_l2)
ax[1].set_title(f"a={a_outlier_l2:4f} b={b_outlier_l2:.4f}")
plt.tight_layout()
plt.savefig("report/figures/regression_l2.png")

# L1 optimisation

B_coefs = []

for i, y in enumerate([y_noisy, y_outlier]):
    obj_l1 = lambda B: np.sum(np.abs(B[0] * x + B[1] - y))
    grad_l1 = lambda B: np.array(
        [
            np.dot(x, np.sign(B[0] * x + B[1] - y)),
            np.sum(np.sign(B[0] * x + B[1] - y)),
        ]
    )

    # Perform gradient descent
    B0 = np.array([0.1, 0.1])
    lr = 0.0001
    B = gradient_descent(obj_l1, grad_l1, B0, 0, 0.01, lr, 100, "gd_l1")
    B_coefs.append(B)

[B_noisy, B_outlier] = B_coefs

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(x, y_noisy)
ax[0].plot(x, B_noisy[0] * x + B_noisy[1])
ax[0].set_title(f"a={B_noisy[0]:.4f} b={B_noisy[1]:.4f}")
ax[1].scatter(x, y_outlier)
ax[1].plot(x, B_outlier[0] * x + B_outlier[1])
ax[1].set_title(f"a={B_outlier[0]:.4f} b={B_outlier[1]:.4f}")
plt.tight_layout()
plt.savefig("report/figures/regression_l1.png")

print("Done. Saved in report/figures/.")

# --- Sparse Signal Reconstruction --- #
print("Performing sparse signal reconstruction via ISTA...")
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

# Plot original samples and estimated signals
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0, 0].plot(abs(data))
ax[0, 0].set_title("Measurements")
ax[0, 1].plot(abs(sample_random))
ax[0, 1].set_title("Random Sample")
ax[0, 2].plot(abs(sample_unif))
ax[0, 2].set_title("Equidistant Sample")
ax[1, 0].plot(abs(signal))
ax[1, 0].set_title("Signal")
ax[1, 1].plot(abs(signal_random))
ax[1, 1].set_title("Signal from Random Sample")
ax[1, 2].plot(abs(signal_unif))
ax[1, 2].set_title("Signal from Equidistant Sample")
plt.tight_layout()
plt.savefig("report/figures/signal.png")

# Perform ISTA on both subsampled measurements
data_random_recon, mse_random = iterative_soft_thresholding(
    sample_random, lam=2e-2, n_iters=100, gt=data
)
data_unif_recon, mse_unif = iterative_soft_thresholding(
    sample_unif, lam=2e-2, n_iters=100, gt=data
)

signal_random_recon = ifft1c(data_random_recon) * sample_freq
signal_unif_recon = ifft1c(data_unif_recon) * sample_freq

# Plot estimated signals
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].plot(abs(signal_random_recon), label="random")
ax[0, 0].plot(abs(signal), label="gt")
ax[0, 0].legend()
ax[0, 0].set_title("Reconstructed Signal: Random")
ax[0, 1].plot(abs(signal_unif_recon), label="unif")
ax[0, 1].plot(abs(signal), label="gt")
ax[0, 1].legend()
ax[0, 1].set_title("Reconstructed Signal: Equidistant")
ax[1, 0].plot(mse_random)
ax[1, 0].set(
    title="L2 Loss of Data Reconstruction", xlabel="Iteration", ylabel="Loss"
)
ax[1, 1].plot(mse_unif)
ax[1, 1].set(
    title="L2 Loss of Data Reconstruction", xlabel="Iteration", ylabel="Loss"
)
plt.tight_layout()
plt.savefig("report/figures/signal_reconstruct.png")

print("Done. Saved in report/figures/.")

# --- Image Compression via Wavelet Decomposition --- #
print("Performing image compression task...")
# Read image
river_img = skimage.io.imread("data/river_side.jpeg")
river_img = rgb2gray(river_img)

# Wavelet transform, and then invert
river_img_dw = dwt2(river_img)
river_img_recon = idwt2(river_img_dw)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

plot_image(ax[0], river_img, "Original", "grey")
plot_image(ax[1], river_img_recon, "Reconstruction", "grey", gt=river_img)
plot_image(
    ax[2], abs(river_img_recon - river_img), "Absolute difference", "grey"
)

plt.tight_layout()
plt.savefig("report/figures/river_img.png")

# Show the effect of thresholding the wavelet decomposition by the top 15% of
# values.
river_img_dw = dwt2(river_img)
th = np.quantile(abs(river_img_dw), 0.85)
river_img_dw_th = abs(river_img_dw) > th

plotrange = (
    np.quantile(abs(river_img_dw), 0.005),
    np.quantile(abs(river_img_dw), 0.995),
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
aximg0 = ax[0].imshow(abs(river_img_dw), vmin=plotrange[0], vmax=plotrange[1])
ax[0].set_title("Wavelet Transform")
fig.colorbar(aximg0, ax=ax[0])
aximg1 = ax[1].imshow(abs(river_img_dw_th), cmap="grey")
ax[1].set_title("Wavelet Transform: Top 15%")
plt.tight_layout()
plt.savefig("report/figures/wavelet_transform.png")

# Plot reconstructions with different coefficient thresholds
for fr in [0.2, 0.15, 0.1, 0.05, 0.025, 0.01]:
    river_img_dw = dwt2(river_img)
    th = np.quantile(abs(river_img_dw), 1 - fr)
    river_img_dw = river_img_dw * (abs(river_img_dw) > th)
    river_img_recon = idwt2(river_img_dw)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    plot_image(ax[0], river_img, "Original", "grey")
    plot_image(
        ax[1],
        river_img_recon,
        f"Reconstruction (Top {fr*100}% of WT)",
        "grey",
        gt=river_img,
    )
    plot_image(
        ax[2], abs(river_img_recon - river_img), "Absolute difference", "grey"
    )

    plt.tight_layout()
    plt.savefig(f"report/figures/river_img_compressed_{fr:.3f}.png")

print("Done. Saved in report/figures/.")

# --- Gradient Descent Convergence --- #
print("Performing gradient descent task...")
# Define specific objective function, its gradient, and gradient descent step
obj = lambda x: 0.5 * (x[0] ** 2) + (x[1] ** 2)
grad = lambda x: np.array([x[0], 2.0 * x[1]])

# Perform gradient descent
x0 = np.array([1.0, 1.0])
lr = 0.5
eps = 0.01
obj_min = 0.0
n_iters = 1000
gradient_descent(obj, grad, x0, obj_min, eps, lr, n_iters, "gradient_descent")

print("Done. Saved in report/figures/.")
