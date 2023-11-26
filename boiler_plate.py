import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np
import math
from tqdm.auto import tqdm
import functools
import argparse
from packaging import version as pver
import json
import cv2

# mp.set_start_method('spawn', force=True)
# mp.set_sharing_strategy('file_system')

device = torch.device("cpu")
model_weights = torch.load("foxnerf/checkpoints/ngp_ep0307.pth", map_location=device)[
    "model"
]


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    """
    Converting the camera pose from the nerf format to the ngp format
    """
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose


def custom_meshgrid(*args):
    """
    Custom Meshgrid
    """
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def read_transform(transform_path):
    """
    Reading the camera poses and instrinsics from the json file
    """

    with open(transform_path, "r") as f:
        transform = json.load(f)

    H = None
    W = None
    if "h" in transform and "w" in transform:
        H = transform["h"]
        W = transform["w"]

    # The camera frames
    frames = transform["frames"]
    poses = []

    for frame in frames:
        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        # Scaling the camera pose
        pose = nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0])
        poses.append(pose)

    poses = torch.from_numpy(np.stack(poses, axis=0))  # [N, 4, 4]

    # calculate mean radius of all camera poses
    radius = poses[:, :3, 3].norm(dim=-1).mean(0).item()

    # Camera pose and intrinsics
    poses = poses.to(device)
    fl_x = transform["fl_x"]
    fl_y = transform["fl_y"]
    cx = transform["cx"]
    cy = transform["cy"]
    intrinsics = np.array([fl_x, fl_y, cx, cy])

    return poses, intrinsics, radius, int(H), int(W)


def generate_rays(poses, intrinsics, H, W, num_rays=-1, error_map=None, patch_size=1):
    """
    Calculating rays from the camera poses and instrinsics. Takes the follwing arguments:
    In case of rendering, we take N = -1, resulting in 2,073,600 rays i.e. for each pixel,
    we are taking a sample along the ray.
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    # Custom mesh grid
    i, j = custom_meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
    )  # float
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}
    N = num_rays
    if N > 0:
        # NOTE: Not used in final implementation.

        N = min(N, H * W)
        print(f"[get_rays] N = {N}")
        # if use patch-based sampling, ignore error_map
        if patch_size > 1:
            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner pixels... but I don't have other ideas.
            num_patch = N // (patch_size**2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(
                torch.arange(patch_size, device=device),
                torch.arange(patch_size, device=device),
            )
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            # Random Sampling for calculating the sampled points along the ray
            # NOTE: not used in the final implementation
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:
            # NOTE: Not used in the final implementation
            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(
                error_map.to(device), N, replacement=False
            )  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = (
                inds_coarse // 128,
                inds_coarse % 128,
            )  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (
                (inds_x * sx + torch.rand(B, N, device=device) * sx)
                .long()
                .clamp(max=H - 1)
            )
            inds_y = (
                (inds_y * sy + torch.rand(B, N, device=device) * sy)
                .long()
                .clamp(max=W - 1)
            )
            inds = inds_x * W + inds_y

            results["inds_coarse"] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results["inds"] = inds

    else:
        # For a standard image, we take H = 1920, W = 1080, resulting in 2,073,600 rays
        # NOTE: This is the final implementation in case of rendering.
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    # Calculating rays direction vectors for each pixel using camera focus point and principal axis
    # and normalizing the direction vectors
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    # Calculating the rays origin, asumming a pinhole camera
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d
    results["H"] = H
    results["W"] = W

    return results


def compute_near_far(rays_o, rays_d, aabb):
    """
    Calculating the near and far intersection times with the axis-aligned bounding box
    """
    t_min = (aabb[:3] - rays_o) / rays_d
    t_max = (aabb[3:] - rays_o) / rays_d

    nears = torch.max(torch.min(t_min, t_max), dim=1)[0]
    fars = torch.min(torch.max(t_min, t_max), dim=1)[0]

    return nears, fars


def cal_xyzs(rays_o, rays_d, aabb_infer, num_steps=16):
    """
    For each ray, we sample 128 points along its direction
    Args:
        rays_o: [N, 3]
        rays_d: [N, 3]
        aabb_infer: [6]
        num_steps: int
    """

    pos0_ray_o = rays_o
    pos0_ray_d = rays_d
    N = pos0_ray_o.shape[0]

    # calculate nears and fars for axis aligned bounding box for each ray
    nears, fars = compute_near_far(pos0_ray_o, pos0_ray_d, aabb_infer)
    nears = nears.unsqueeze_(-1)
    fars = fars.unsqueeze_(-1)

    # uniform sampling along the z axis
    z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
    z_vals = z_vals.expand((N, num_steps))  # [N, T]
    z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

    sample_dist = (fars - nears) / num_steps

    # generate xyzs based on uniform sampling along the z axis
    xyzs = pos0_ray_o.unsqueeze(-2) + pos0_ray_d.unsqueeze(-2) * z_vals.unsqueeze(
        -1
    )  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
    # Make sure the xyzs are within the aabb
    xyzs = torch.min(torch.max(xyzs, aabb_infer[:3]), aabb_infer[3:])  # a manual clip.

    # After that, we can also use density to accelerate the rendering process by selectively choosing the number of points.
    # Not implemented in the final implementation due to time constraints.

    return xyzs, z_vals, fars, nears, sample_dist


def fast_hash_op(pos_grid):
    """
    Fast hashing function for calculating the hash encodings. Given a position grid, it calculates the spatial hash
    encoding for each point in the grid using the bitwise xor operation with prime numbers, choosen in the Instant NGP paper.

    The function has been parallelized over the number of samples per ray and number of points for effective computation resulting in
    pos_grid being a tensor of shape [2**3,num_samples_per_ray, 3]

    Args:
        pos_grid: [8 ,num_samples_per_ray, 3]
    Returns:
        result: [8, num_samples_per_ray]
    """
    pos_grid = pos_grid.int()
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    result = torch.zeros(pos_grid.shape[0], pos_grid.shape[1], dtype=torch.int32).to(
        device
    )
    for i in range(3):
        # Calculating bitwise xor with prime numbers along spatial dimensions
        result = torch.bitwise_xor(result, pos_grid[:, :, i] * primes[i])
    return result


def get_grid_index_op(C, D, align_corners, hashmap_size, resolution, pos_grid, ch=0):
    """
    Given a position grid local, it calculates the corresponding index in the hashmap for each
    point in the grid. For low levels, i.e. coarse resolution, we don't have need to calculate the
    hash encoding for each point as there is one to one mapping between the position grid and the
    hash encoding. But for higher levels, we need to calculate the hash encoding for each point in
    the grid.

    The function has been parallelized over the number of samples per ray and number of points for effective computation resulting in
    pos_grid being a tensor of shape [8,num_samples_per_ray, 3]

    Args:
        pos_grid: [8, num_samples_per_ray, 3]
    Returns:
        index: [8, num_samples_per_ray]

    """

    # Calcularing the stride for each axis in the position grid
    stride = torch.pow(resolution + 1, torch.arange(3)).to(device)
    stride = stride.unsqueeze(0).expand(
        (pos_grid.shape[0], pos_grid.shape[1], -1)
    )  # [8,3]

    # Calculating the index for each point in the position grid using the stride and pos grid local
    index = torch.empty([8, 3], dtype=torch.int64, device=device)
    index = pos_grid * stride  # [8,3]

    mask = (stride <= hashmap_size).to(device)
    index = torch.where(mask, index, torch.zeros_like(index))

    index = index.sum(dim=-1)  # [8]
    stride = torch.tensor(
        [[(resolution + 1) ** 3] * pos_grid.shape[1]] * pos_grid.shape[0]
    ).to(device)

    mask = (stride > hashmap_size).to(device)  # [8]
    index = index.int()

    # Calculating the hash encoding for each point in the position grid
    # until and unless the stride is greater than the hashmap size, there is one to one mapping
    # between the position grid and the hash encoding and there is no need to calculate the hash encoding.
    index = torch.where(mask, fast_hash_op(pos_grid), index)
    del mask, stride

    # Return the final index for each point in the position grid
    return (index % hashmap_size) * C + ch


def compute_hash_op(xyzs, offsets, embeddings):
    """
    Given a point in xyzs, it calculates the hash encoding for that point by first calculating the
    position grid local and then calculating the index for the position grid local in the hashmap.
    Then using d-linear interpolation, it calculates the hash encoding for the point.

    The function has been parallelized over the number of samples per ray for effective computation resulting in
    xyzs being a tensor of shape [num_samples_per_ray, 3]

    Args:
        xyzs: [num_samples_per_ray, 3]

    Returns:
        ans: [num_samples_per_ray, num_levels, C]
    
    """

    # Parameters for calculating the hash encoding
    # number of levels in the hashmap, base resolution, desired resolution, align_corners, per_level_scale
    num_levels = 16
    base_resolution = 16
    desired_resolution = 4096
    align_corners = False
    per_level_scale = np.exp2(
        np.log2(desired_resolution / base_resolution) / (num_levels - 1)
    )
    bound = 2.0

    # Normalize xyzs to [0,1]
    xyzs = (xyzs + bound) / (2 * bound)  # map to [0, 1]
    xyzs = xyzs.contiguous()

    B, D = xyzs.shape  # batch size, coord dim
    L = offsets.shape[0] - 1  # level
    C = 2  # embedding dim for each level
    S = np.log2(
        per_level_scale
    )  # resolution multiplier at each level, apply log2 for later CUDA exp2f
    H = base_resolution  # base resolution

    # len_sample denotes the number of samples per ray
    len_sample = xyzs.shape[0]
    ans = torch.empty(len_sample, num_levels, C).to(device)
    inputs = xyzs

    # Iterating over all the levels in the hashmap from coarse to fine
    for level in range(num_levels):
        # Calculating the hashmap size, scale and resolution for each level
        # using the offsets in the model weights
        hashmap_size = offsets[level + 1] - offsets[level]
        scale = 2 ** (S * level) * H - 1.0
        resolution = math.ceil(scale) + 1

        # Calculating the position grid local for each point in xyzs
        pos = inputs * scale + 0.5
        pos_grid = torch.floor(pos)
        pos -= pos_grid

        # Calculating position grid local for each point in xyzs, also weight of each point in the position grid local
        # to that of the point. Here, 'w' denotes the weight of each point in the position grid local equal to x_i - floor(x_i)
        # d-linear interpolation is used to prevent the blocky nature of interpolation and for smoothing out.
        # val0 and val1 are used to calculate the voxel points in the position grid
        val0 = 1 << torch.arange(D)
        val0 = val0.expand((1 << D, pos.shape[0], -1))
        val1 = torch.arange(1 << D)
        val1 = val1.repeat((1 << D) // val1.shape[0], pos.shape[0] * D).reshape(
            1 << D, pos.shape[0], D
        )
        mask = (val0 & val1) == 0
        w = torch.where(mask, 1 - pos, pos)
        w = w.prod(dim=-1)

        pos_grid_local = torch.where(mask, pos_grid, pos_grid + 1)

        # Getting the grid index for each point in the position grid local
        # [num_samples_per_ray,8, 3]
        index = get_grid_index_op(
            C, D, align_corners, hashmap_size, resolution, pos_grid_local
        )

        # per, point we have two local features, so for all the 8 points in the position grid local
        # we calculate the embeddings and then take the weighted sum of the embeddings i.e. d-linear interpolation 
        # based on the relative position of x within it's hypercube. Here, w denotes the weight
        # to calculate the hash encoding for the point
        local_feature0 = (
            w * embeddings[(int(offsets[level]) * int(C) + index).long()]
        ).sum(dim=0)

        local_feature1 = (
            w * embeddings[(int(offsets[level]) * int(C) + index + 1).long()]
        ).sum(dim=0)

        ans[:, level, :] = torch.cat(
            [local_feature0[:, None], local_feature1[:, None]], dim=-1
        ).to(device)

    del (
        inputs,
        xyzs,
        pos,
        pos_grid,
        pos_grid_local,
        index,
        w,
        local_feature0,
        local_feature1,
        mask,
        val0,
        val1,
    )

    return ans


class sigma_net(nn.Module):
    def __init__(self) -> None:
        """
        Neural network to predict the density and geometric features for each sampled point along the ray
        Takes in the hash encoding of the point as input
        """
        super().__init__()
        self.linear1 = nn.Linear(32, 64, bias=False)
        self.linear2 = nn.Linear(64, 16, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        return x


def SHencoder(inputs):
    """
    This functions takes in the directions of the rays and calculates the spherical harmonics
    encoding for each direction.

    For instant NGP, we used the 4 degree spherical harmonics encoding.
    """

    input_dim = 3
    degree = 4
    output_dim = 16

    inputs = inputs.reshape(-1, input_dim)
    B = inputs.shape[0]
    outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

    D = input_dim
    C = degree

    x, y, z = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    # Calculate the spherical harmonics encoding for each direction
    xy = x * y
    xz = x * z
    yz = y * z
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xyz = xy * z

    x4 = x2 * x2
    y4 = y2 * y2
    z4 = z2 * z2
    x6 = x4 * x2
    y6 = y4 * y2
    z6 = z4 * z2

    outputs[:, 0] = 0.28209479177387814  # 1/(2*sqrt(pi))
    outputs[:, 1] = -0.48860251190291987 * y  # -sqrt(3)*y/(2*sqrt(pi))
    outputs[:, 2] = 0.48860251190291987 * z  # sqrt(3)*z/(2*sqrt(pi))
    outputs[:, 3] = -0.48860251190291987 * x  # -sqrt(3)*x/(2*sqrt(pi))
    outputs[:, 4] = 1.0925484305920792 * xy  # sqrt(15)*xy/(2*sqrt(pi))
    outputs[:, 5] = -1.0925484305920792 * yz  # -sqrt(15)*yz/(2*sqrt(pi))
    outputs[:, 6] = (
        0.94617469575755997 * z2 - 0.31539156525251999
    )  # sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
    outputs[:, 7] = -1.0925484305920792 * xz  # -sqrt(15)*xz/(2*sqrt(pi))
    outputs[:, 8] = (
        0.54627421529603959 * x2 - 0.54627421529603959 * y2
    )  # sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    outputs[:, 9] = 0.59004358992664352 * y * (-3.0 * x2 + y2)
    outputs[:, 10] = 2.8906114426405538 * xy * z
    outputs[:, 11] = 0.45704579946446572 * y * (1.0 - 5.0 * z2)
    outputs[:, 12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0)
    outputs[:, 13] = 0.45704579946446572 * x * (1.0 - 5.0 * z2)
    outputs[:, 14] = 1.4453057213202769 * z * (x2 - y2)
    outputs[:, 15] = 0.59004358992664352 * x * (-x2 + 3.0 * y2)

    return outputs


class color_net(nn.Module):
    def __init__(self) -> None:
        """
        Neural network to predict the color for each sampled point along the ray
        Takes in the spherical harmonics of the ray and the geometric features as input
        """
        super().__init__()
        self.linear1 = nn.Linear(31, 64, bias=False)
        self.linear2 = nn.Linear(64, 64, bias=False)
        self.linear3 = nn.Linear(64, 3, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.relu(x, inplace=True)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x


def color(model_weights, dirs, geo_feat):
    """
    Predicts the color for each sampled point along the ray
    Consists of two parts:
        1. SH encoding of the ray
        2. Prediction of color using the SH encoding and the geometric features
    """

    sh_encoding = SHencoder(dirs)

    input = torch.cat([sh_encoding, geo_feat], dim=-1)
    model = color_net()
    model.linear1.weight.data = model_weights["color_net.0.weight"]
    model.linear2.weight.data = model_weights["color_net.1.weight"]
    model.linear3.weight.data = model_weights["color_net.2.weight"]

    model = model.eval()
    with torch.no_grad():
        color = model(input)
    return color

def sample_pdf(bins, weights, n_samples, det=False):
    """
    This implementation is from NeRF
    bins: [B, T], old_z_vals
    weights: [B, T - 1], bin weights.
    return: [B, n_samples], new_z_vals
    """

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples



def upsample_zvals(z_vals, density_outputs, rays_o, rays_d, aabb, upsample_steps=128):
    """
    Upsamples the z-values and returns new XYZ coordinates and z-values.
    Used in case of determining xyzs and z_vals for fine level sampling.

    Args:
        z_vals (torch.Tensor): The z-values of the input rays. Shape: [N, T].
        density_outputs (dict): Dictionary containing density outputs.
        rays_o (torch.Tensor): The origin coordinates of the rays. Shape: [N, 3].
        rays_d (torch.Tensor): The direction vectors of the rays. Shape: [N, 3].
        aabb (torch.Tensor): The axis-aligned bounding box. Shape: [6].
        upsample_steps (int): The number of steps to upsample the z-values. Default: 128.

    Returns:
        torch.Tensor: The new XYZ coordinates. Shape: [N, t, 3].
        torch.Tensor: The new z-values. Shape: [N, t].
    """
    if upsample_steps > 0:
        density_scale = 1.0
        with torch.no_grad():

            deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
            deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)  # [N, T]

            alphas = 1 - torch.exp(-deltas * density_scale * density_outputs['sigma'].squeeze(-1))  # [N, T]
            alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+1]
            weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T]

            # sample new z_vals
            z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1])  # [N, T-1]
            new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=True).detach()  # [N, t]

            new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
            new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        return new_xyzs, new_z_vals



if __name__ == "__main__":
    # add command line argument to take img to render
    parser = argparse.ArgumentParser(description="Instant NGP renderer from scratch")
    parser.add_argument("--img", type=int, help="The image number", default=0)

    # Reading the camera poses and extracting the instrinsics from the json file
    poses, intrinsics, radius, H, W = read_transform("data/fox/test_transform.json")
    print("Successfully read the camera poses and instrinsics")

    # Generating the rays from the camera poses and instrinsics
    # For a standard image, we take H = 1920, W = 1080, resulting in 2,073,600 rays
    results = generate_rays(
        poses, intrinsics, H, W, num_rays=-1, error_map=None, patch_size=1
    )
    print("Generated Rays")

    rays_o = results["rays_o"]
    rays_d = results["rays_d"]

    # Take a center patch of ray_o and ray_d of size 100x100 pixels
    patch_size = 100
    new_rays_o = torch.empty(3,patch_size*patch_size, 3)
    new_rays_d = torch.empty(3,patch_size*patch_size, 3)
    for i in range(patch_size):
        val = i
        new_rays_o[:,val*patch_size:(val+1)*patch_size,:] = rays_o[:, (960*1080)+1080*i+540-int(patch_size/2): (960*1080)+1080*i+540+int(patch_size/2),:]
        new_rays_d[:,val*patch_size:(val+1)*patch_size,:] = rays_d[:, (960*1080)+1080*i+540-int(patch_size/2): (960*1080)+1080*i+540+int(patch_size/2),:]
    
    rays_o = new_rays_o
    rays_d = new_rays_d

    num_levels = 16
    C = 2
    density_scale = 1
    N = rays_o.shape[1]
    img = parser.parse_args().img

    # For each ray we sample 128 points along its direction
    num_steps = 128

    # Calculating the xyzs, deltas, z_vals, fars and nears for a given ray
    # Resulting vector size is [num_rays, num_samples_per_ray, 3]
    aabb_infer = model_weights["aabb_infer"]
    xyzs, z_vals, fars, nears, sample_dist = cal_xyzs(
        rays_o[img], rays_d[img], aabb_infer, num_steps=num_steps
    )

    torch.save([xyzs, z_vals, fars, nears, sample_dist], f"xyzs_{img}.pt")
    
    # Calculating the hash encodings for each xyzs
    # Here, we make use of torch.multiprocessing to parallelize the computation of hash encodings
    # over the CPU. We parallelize over the number of samples per ray. This parallelization is not
    # optimal and can be improved by parallelizing over the number of rays as well. But due to time and
    # resource constraints, I have not implemented it.
    samples = xyzs.shape[0]
    limit = 32
    ans = []

    # Reading the offsets and embeddings from the model weights
    # and sharing them over the CPU for parallelization
    offsets = model_weights["encoder.offsets"]
    embeddings = model_weights["encoder.embeddings"].flatten()
    offsets.share_memory_()
    embeddings.share_memory_()

    hash_encodings = torch.empty(samples, num_steps, num_levels, C).to(device)
    # Parallelizing the computation of hash encodings
    for i in tqdm(range(samples // limit + 1)):
        with mp.Pool(16) as pool:
            if i * limit == min((i + 1) * limit, samples):
                break
            val = xyzs[i * limit : min((i + 1) * limit, samples), :, :]
            hash_encodings[i * limit : min((i + 1) * limit, samples), :,:,:] = torch.stack(list(
                pool.map(
                    functools.partial(
                        compute_hash_op, offsets=offsets, embeddings=embeddings
                    ),
                    val,
                )
            ))
        del pool

    # [num_rays, num_samples_per_ray, num_levels, C=2]
    torch.save(hash_encodings, f"hash_encodings_{img}_{num_steps}.pt")
    print("Successfully saved hash encodings")

    # [num_rays, num_samples_per_ray, num_levels, C=2] -> [num_rays*num_samples_per_ray, num_levels=16*C=2]
    hash_encodings = hash_encodings.view(
        hash_encodings.shape[0] * hash_encodings.shape[1], -1
    )
    print("Hash Encodings done")

    # Calculating the density and geometric features for each xyzs
    # Creating the model and loading the model weights
    sigma = sigma_net()
    sigma.linear1.weight.data = model_weights["sigma_net.0.weight"]
    sigma.linear2.weight.data = model_weights["sigma_net.1.weight"]
    sigma = sigma.eval()

    # Forward pass through the model over the hash encodings calculated
    with torch.no_grad():
        print("Running density prediction")
        sigma_net_pred = sigma(hash_encodings)
        density = torch.exp(sigma_net_pred[..., 0])
        geo_feat = sigma_net_pred[..., 1:]
        density_outputs = {"sigma": density, "geo_feat": geo_feat}

    for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)
    

    # As we know, NeRF uses coarse to fine sampling for calculating the density and color for each point along the ray
    # So based on the coarse sampling done above, we now do fine sampling for each ray
    # We calculate the hash encodings for each point in the fine sampling and then use the model to predict the density
    # and geometric features for each point in the fine sampling
    
    # Finding new xyzs and new z_vals for each ray
    upsample_steps = 128
    new_xyzs, new_z_vals = upsample_zvals(z_vals,density_outputs, rays_o[img], rays_d[img],aabb_infer, upsample_steps = upsample_steps)
    
    samples = new_xyzs.shape[0]

    new_hash_encodings = torch.empty(samples, upsample_steps, num_levels, C).to(device)
    # Parallelizing the computation of hash encodings
    for i in tqdm(range(samples // limit + 1), desc="Cal new densities"):
        with mp.Pool(16) as pool:
            if i * limit == min((i + 1) * limit, samples):
                break
            val = new_xyzs[i * limit : min((i + 1) * limit, samples), :, :]
            new_hash_encodings[i * limit : min((i + 1) * limit, samples), :,:,:] = torch.stack(list(
                pool.map(
                    functools.partial(
                        compute_hash_op, offsets=offsets, embeddings=embeddings
                    ),
                    val,
                )
            ))
        del pool
    
    new_hash_encodings = new_hash_encodings.view(
        new_hash_encodings.shape[0] * new_hash_encodings.shape[1], -1
    )

    with torch.no_grad():
        print("Running density prediction")
        new_sigma_net_pred = sigma(new_hash_encodings)
        new_density = torch.exp(sigma_net_pred[..., 0])
        new_geo_feat = sigma_net_pred[..., 1:]
        new_density_outputs = {"sigma": new_density, "geo_feat": new_geo_feat}

    for k, v in new_density_outputs.items():
        new_density_outputs[k] = v.view(N, upsample_steps, -1)

    z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
    z_vals, z_index = torch.sort(z_vals, dim=1)

    xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
    xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

    for k in density_outputs:
        tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
        density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

    deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
    deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
    alphas = 1 - torch.exp(-deltas * density_scale * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
    alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
    weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

    for k, v in density_outputs.items():
        density_outputs[k] = v.view(-1, v.shape[-1])

    N = xyzs.shape[0]
    num_steps = xyzs.shape[1]

    torch.save(density_outputs, f"density_outputs_{img}_{num_steps}.pt")
    print("Density prediction done")

    # Now we calculate the color for each sampled point along each ray
    rays_dir = rays_d[img]
    rays_dir = rays_dir[:, None, :].expand(-1, num_steps, -1)

    # rgbs = color(model_weights, rays_dir.reshape(-1, 3), density_outputs['geo_feat'])
    batch_size = 4096
    rgbs = []
    for i in tqdm(range(0, N, batch_size)):
        rgbs.append(color(model_weights, rays_dir[i : i + batch_size], density_outputs['geo_feat'][i*num_steps: (i + batch_size)*num_steps]))
    rgbs = torch.cat(rgbs, dim=0)
    torch.save(rgbs, f"rgb_{img}_{num_steps}.pt")
    print("Color prediction done and saved")

    # Now for each ray we calculate the alpha and weights, which will be used to
    # determine the final color and depth
    density_scale = 1
    density = density_outputs["sigma"].view(N, num_steps, -1)
    rgbs = rgbs.reshape(N, num_steps, 3)
    
    ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)

    # calculate weighted depth
    depth = torch.sum(weights * ori_z_vals, dim=-1)

    # calculate weighted color
    image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)

    # save the image and depth tensors
    torch.save(image, f"image_{img}_{num_steps}.pt")
    torch.save(depth, f"depth_{img}_{num_steps}.pt")
    print("Image and depth tensors calculated and saved")

    # Now finally reshaping the tensors and saving it using opencv
    image = image.reshape(patch_size, patch_size, 3)
    depth = depth.reshape(patch_size,patch_size)

    # Normalize the image to 255 and save it
    image = image.cpu().numpy()
    image = image / image.max()
    image = image * 255
    image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"image_{img}.png", image)

    # normalize the depth to 255 and save it
    depth = depth.cpu().numpy()
    depth = depth / depth.max()
    depth = depth * 255
    depth = depth.astype(np.uint8)
    cv2.imwrite(f"depth_{img}.png", depth)
