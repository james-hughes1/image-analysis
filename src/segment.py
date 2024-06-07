import numpy as np
import matplotlib.pyplot as plt
import skimage
import itertools

from imagetools.ml import KMeans_Custom

from skimage.filters import threshold_otsu, rank, unsharp_mask
from skimage.morphology import (
    disk,
    binary_opening,
    remove_small_objects,
    label,
    binary_closing,
)
from skimage.restoration import denoise_tv_bregman
from skimage.color import rgba2rgb, rgb2gray
from skimage.segmentation import chan_vese
from skimage.measure import regionprops


# CT Lung segmentation
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Import
ct_img = skimage.io.imread("data/CT.png")
ct_img = rgb2gray(rgba2rgb(ct_img))
ax[0, 0].imshow(ct_img)

# Otsu's method segmentation
ct_threshold = threshold_otsu(ct_img)
ct_segmented = ct_img > ct_threshold
ax[0, 1].imshow(ct_segmented)

# Post-processing
se = disk(4)
ct_mask = remove_small_objects(binary_opening(ct_segmented, se))
ax[0, 2].imshow(ct_mask)

# Show segmented lungs
ct_labelled = label(ct_mask == 0)

ct_props = regionprops(ct_labelled)
lung_idx = sorted(list(range(3)), key=lambda i: ct_props[i].area)[:2]

ct_labelled = 0.7 * (ct_labelled == lung_idx[0] + 1) + (
    ct_labelled == lung_idx[1] + 1
)
ax[1, 0].imshow(ct_labelled, cmap="jet")

# Show original image with segmentation mask
ax[1, 1].imshow(0.1 * ct_img + 0.9 * ct_img * (ct_labelled > 0))

for p in itertools.product(list(range(2)), list(range(3))):
    ax[p].axis("off")

plt.tight_layout()
plt.savefig("report/figures/ct_segmentation.png")

# Coins segmentation
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Import
coins_img = skimage.io.imread("data/coins.png")
coins_img = rgb2gray(rgba2rgb(coins_img))
coins_img = (coins_img * 255).astype("uint8")

# Mark desired coins in order to choose masks later
coins_marked = coins_img.copy()
coins_marked[50:55, 40:45] = 255
coins_marked[125:130, 100:105] = 255
coins_marked[200:205, 160:165] = 255
coins_marked[275:280, 240:245] = 255

ax[0, 0].imshow(coins_marked)

# Pre-processing
coins_pre = rank.median(coins_img, np.ones((1, 7)))
coins_pre = unsharp_mask(coins_pre, radius=10)
ax[0, 1].imshow(coins_pre)

# Chan-vese segmentation
coins_segmented = 1 - chan_vese(coins_pre, mu=0.1)
se = disk(3)
coins_segmented = binary_closing(coins_segmented, se)
ax[0, 2].imshow(coins_segmented)

coins_labelled = label(coins_segmented)

desired_coins_idx = [
    coins_labelled[50, 40],
    coins_labelled[125, 100],
    coins_labelled[200, 160],
    coins_labelled[275, 240],
]

# Show desired coins
coins_labelled = np.isin(coins_labelled, desired_coins_idx)
ax[1, 0].imshow(coins_labelled)

# Show original image with segmentation mask
ax[1, 1].imshow(0.2 * coins_img + 0.8 * coins_img * coins_labelled)

for p in itertools.product(list(range(2)), list(range(3))):
    ax[p].axis("off")

plt.tight_layout()
plt.savefig("report/figures/coins_segmentation.png")

# Flowers segmentation
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Import
flowers_img = skimage.io.imread("data/noisy_flower.jpg")
flowers_img = rgba2rgb(flowers_img)
ax[0, 0].imshow(flowers_img)

# Pre-processing
flowers_pre = denoise_tv_bregman(flowers_img, weight=0.5)
ax[0, 1].imshow(flowers_pre)

# KMeans segmentation
km = KMeans_Custom(K=6)

flowers_data = flowers_pre[::8, ::8, :].reshape(-1, 3)
km.fit(flowers_data, verbose=1)

flowers_flat = flowers_pre.reshape(-1, 3)
flowers_segmented = km.predict_cluster(flowers_flat)
flowers_segmented = flowers_segmented.reshape(flowers_pre.shape[:-1])
ax[0, 2].imshow(flowers_segmented, cmap="jet")

# Post-processing
flowers_purple = flowers_segmented == 0
se = disk(2)
flowers_purple = binary_closing(
    remove_small_objects(flowers_purple, min_size=40), se
)
ax[1, 0].imshow(flowers_purple)

# Show original image with segmentation mask
ax[1, 1].imshow(
    0.2 * flowers_img + 0.8 * flowers_img * (flowers_purple[:, :, None])
)

for p in itertools.product(list(range(2)), list(range(3))):
    ax[p].axis("off")

plt.tight_layout()
plt.savefig("report/figures/flowers_segmentation.png")
