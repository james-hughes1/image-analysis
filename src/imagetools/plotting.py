"""!@file plotting.py
@brief Module building on matplotlib to provide image-specific plotting
functions.

@author Created by J. Hughes on 8th June 2024.
"""

import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def plot_image(ax_sp, img, title, cmap, gt=None):
    """!@brief Plots an image and compares to GT

    @param ax_sp Specified matplotlib.pyplot.Axes object
    @param img Image to be displayed
    @param title Image title string
    @param cmap String defining matplotlib colormap for image
    @param gt Ground truth measurements to compare to
    """
    ax_sp.imshow(img, cmap=cmap)
    ax_sp.set_xticks([])
    ax_sp.set_yticks([])
    ax_sp.set_title(title)
    if not (gt is None):
        data_range = np.max(gt) - np.min(gt)
        psnr_fbp = psnr(gt, img, data_range=data_range)
        ssim_fbp = ssim(gt, img, data_range=data_range)
        ax_sp.set(xlabel=f"PSNR: {psnr_fbp:.2f} dB, SSIM: {ssim_fbp:.2f}")
