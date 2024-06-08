"""!@file lgd.py
@brief Script for training LGD.

@details Reconstructs CT images using FBP and ADMM, and compares to a
data-driven LGD algorithm. Can be run in 'demo' mode, just performing the last
10 epochs of training from a checkpoint, or 'full' mode which runs all 2000
training epochs.
@author Created by J. Hughes on 8th June 2024.
"""

# import libraries for CT and deep learning
import numpy as np
import astra

import torch
import torch.nn as nn
import odl
import odl.contrib.torch as odl_torch
import matplotlib.pyplot as plt
import argparse

from imagetools.plotting import plot_image

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--mode", type=str, choices=["demo", "full"], required=True
)
args = parser.parse_args()

astra.test()

# --- Set up the forward operator (ray transform) in ODL --- #
# Reconstruction space: functions on the rectangle [-20, 20]^2
img_size = 256  # discretized with 256 samples per dimension
reco_space = odl.uniform_discr(
    min_pt=[-20, -20],
    max_pt=[20, 20],
    shape=[img_size, img_size],
    dtype="float32",
)

# Make a parallel beam geometry with flat detector
num_angles = 30
geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=num_angles)

# Create the forward operator, adjoint operator, and the FBO operator in ODL
fwd_op_odl = odl.tomo.RayTransform(reco_space, geometry)
fbp_op_odl = odl.tomo.fbp_op(
    fwd_op_odl, filter_type="Ram-Lak", frequency_scaling=0.6
)
adj_op_odl = fwd_op_odl.adjoint

# Create phantom and noisy projection data in ODL
phantom_odl = odl.phantom.shepp_logan(reco_space, modified=True)
data_odl = fwd_op_odl(phantom_odl)
data_odl += odl.phantom.white_noise(fwd_op_odl.range) * np.mean(data_odl) * 0.1
fbp_odl = fbp_op_odl(data_odl)

# Convert the image and the sinogram to numpy arrays
phantom_np = phantom_odl.__array__()
fbp_np = fbp_odl.__array__()
data_np = data_odl.__array__()
print("sinogram size = {}".format(data_np.shape))

# Display ground-truth and FBP images
fig, ax = plt.subplots(1, 3, figsize=(9, 6))

plot_image(ax[0], phantom_np.T, "ground-truth", "bone")
plot_image(ax[1], data_np, "sinogram", "bone")
plot_image(ax[2], fbp_np.T, "FBP", "bone", gt=phantom_np.T)

plt.savefig("report/figures/fbp.png")

# Let's solve the TV reconstruction problem using linearized ADMM

# In this example we solve the optimization problem:
# min_x  f(x) + g(Lx) = ||A(x) - y||_2^2 + lam * ||grad(x)||_1,
# Where ``A`` is a parallel beam ray transform, ``grad`` is the spatial
# gradient and ``y`` given noisy data.

# The problem is rewritten in decoupled form as: min_x g(L(x)) with a
# separable sum ``g`` of functionals and the stacked operator ``L``:

#     g(z) = ||z_1 - g||_2^2 + lam * ||z_2||_1,

#                ( A(x)    )
#     z = L(x) = ( grad(x) ).

# Gradient operator for the TV part
grad = odl.Gradient(reco_space)

# Stacking of the two operators
L = odl.BroadcastOperator(fwd_op_odl, grad)

# Data matching and regularization functionals
data_fit = odl.solvers.L2NormSquared(fwd_op_odl.range).translated(data_odl)
lam = 0.015
reg_func = lam * odl.solvers.L1Norm(grad.range)
g = odl.solvers.SeparableSum(data_fit, reg_func)

# We don't use the f functional, setting it to zero
f = odl.solvers.ZeroFunctional(L.domain)

# --- Select parameters and solve using ADMM --- #

# Estimated operator norm, add 10 percent for some safety margin
op_norm = 1.1 * odl.power_method_opnorm(L, maxiter=20)

niter = 200  # Number of iterations
sigma = 2.0  # Step size for g.proximal
tau = sigma / op_norm**2  # Step size for f.proximal

# Optionally pass a callback to the solver to display intermediate results
callback = odl.solvers.CallbackPrintIteration(
    step=10
) & odl.solvers.CallbackShow(step=10)

# Choose a starting point
x_admm_odl = L.domain.zero()

# Run the algorithm
odl.solvers.admm_linearized(
    x_admm_odl, f, g, L, tau, sigma, niter, callback=None
)
x_admm_np = x_admm_odl.__array__()

# Let's display the image reconstructed by ADMM and compare it with FBP
fig, ax = plt.subplots(1, 3, figsize=(9, 6))

plot_image(ax[0], phantom_np.T, "ground-truth", "bone")
plot_image(ax[1], fbp_np.T, "FBP", "bone", gt=phantom_np.T)
plot_image(ax[2], x_admm_np.T, "TV", "bone", gt=phantom_np.T)

plt.savefig("report/figures/admm.png")

# Now, we will "train" learned gradient descent (LGD) network on this specific
# image to show the potential of data-driven reconstruction.
# Keep in mind that this example is unrealistic, in the sense that in practice
# you will train your learned reconstruction network on a dataset containing
# many images, and then test the performance on a new test image (that was not
# a part of your training dataset).

# Find processing device, preferring GPU if possible.
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


# First, let's create a CNN that replaces the prox operator in PGD.
class prox_net(nn.Module):
    def __init__(
        self, n_in_channels=2, n_out_channels=1, n_filters=32, kernel_size=3
    ):
        super(prox_net, self).__init__()
        self.pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(
            n_in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=self.pad,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            n_filters,
            n_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=self.pad,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            n_filters,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=self.pad,
            bias=True,
        )

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x, u):
        """################## YOUR CODE HERE ####################"""
        # Note: here the two inputs denote the current iterate and the gradient
        dx = torch.cat((x, u), 0)
        dx = self.conv1(dx)
        dx = self.act1(dx)
        dx = self.conv2(dx)
        dx = self.act2(dx)
        dx = self.conv3(dx)

        return dx


# Let's compute a reasonable initial value for the step-size as
# step_size = 1/L, where L is the spectral norm of the forward operator.
op_norm = 1.1 * odl.power_method_opnorm(fwd_op_odl)
step_size = 1 / op_norm


class LGD_net(nn.Module):
    def __init__(self, niter=5, step_size=step_size):
        super(LGD_net, self).__init__()
        self.niter = niter
        self.prox = nn.ModuleList(
            [prox_net().to(device) for i in range(self.niter)]
        )
        self.step_size = nn.Parameter(
            step_size * torch.ones(self.niter).to(device)
        )
        self.fwd_op_torch = odl_torch.OperatorModule(fwd_op_odl)
        self.adj_op_torch = odl_torch.OperatorModule(adj_op_odl)

    def forward(self, y, x_init):
        x = x_init
        """ ################## YOUR CODE HERE #################### """
        # Note: the gradient at a given x is A^T(Ax-y).
        for k in range(self.niter):
            prox_net_output = self.prox[k](
                x, self.adj_op_torch(self.fwd_op_torch(x) - y)
            )
            x = x + self.step_size[k] * prox_net_output
        return x


lgd_net = LGD_net().to(device)  # realize the network and export it to GPU
num_learnable_params = sum(
    p.numel() for p in lgd_net.parameters() if p.requires_grad
)
print("number of model parameters = {}".format(num_learnable_params))
y = (
    torch.from_numpy(data_np).to(device).unsqueeze(0)
)  # Noisy sinogram data as a torch tensor

# Initialization for the LGD net. Note that the input to a torch 2D CNN must
# be of size (num_batches x height x width).
x_init = (
    torch.from_numpy(
        fbp_op_odl(y.detach().cpu().numpy().squeeze()).__array__()
    )
    .to(device)
    .unsqueeze(0)
)

ground_truth = (
    torch.from_numpy(phantom_np).to(device).unsqueeze(0)
)  # Target ground-truth as a torch tensor

# Define the loss and the optimizer
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lgd_net.parameters(), lr=1e-4)
num_epochs = 2000

# Determine whether to continue training or create new model.
if args.mode == "demo":
    print(
        "Loading model from /outputs/weights_1990.pth and training for 10"
        " epochs."
    )
    ckpt = torch.load("outputs/weights_1990.pth", map_location=device)
    lgd_net.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_idx = 1989
    losses = ckpt["losses"]
    save_interval = 1
    verbose_interval = 1

else:
    start_idx = 0
    losses = []
    save_interval = 10
    verbose_interval = 100

# Training loop
for epoch in range(start_idx, num_epochs):
    optimizer.zero_grad()
    """################## YOUR CODE HERE #################### """
    # You need to compute the reconstructed image by applying a forward pass
    # through lgd_net, compute the loss, call the backward function on loss,
    # and update the parameters.
    recon = lgd_net(y, x_init)
    loss = mse_loss(recon, ground_truth)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % save_interval == 0:
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": lgd_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "losses": losses,
        }
        checkpoint_filepath = f"outputs/weights_{epoch+1:03d}.pth"
        torch.save(checkpoint, checkpoint_filepath)

    if (epoch + 1) % verbose_interval == 0:
        print(f"epoch = {epoch+1}, loss = {loss.item():.5f}")

lgd_recon_np = (
    recon.detach().cpu().numpy().squeeze()
)  # Convert the LGD reconstruction to numpy format
###############################################

# Let's display the reconstructed images by LGD and compare it with FBP and
# ADMM

fig, ax = plt.subplots(2, 2, figsize=(8, 8))

plot_image(ax[0, 0], phantom_np.T, "ground-truth", "bone")
plot_image(ax[0, 1], fbp_np.T, "FBP", "bone", gt=phantom_np.T)
plot_image(ax[1, 0], x_admm_np.T, "TV", "bone", gt=phantom_np.T)
plot_image(ax[1, 1], lgd_recon_np.T, "LGD", "bone", gt=phantom_np.T)

plt.savefig("report/figures/lgd.png")
