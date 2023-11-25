# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import numpy as np
import math
from tqdm.auto import tqdm
import functools
import argparse
# mp.set_start_method('spawn', force=True)
# %%

device = torch.device('cpu')
model_weights = torch.load('foxnerf/checkpoints/ngp_ep0307.pth', map_location=device)['model']

rays_o = torch.load('rays_o.pt', map_location=device)
rays_d = torch.load('rays_d.pt', map_location=device)

# %%
def compute_near_far(rays_o, rays_d, aabb):
    # Calculate intersection times with planes
    t_min = (aabb[:3] - rays_o) / rays_d
    t_max = (aabb[3:] - rays_o) / rays_d

    # Calculate near and far intersection times
    nears = torch.max(torch.min(t_min, t_max), dim=1)[0]
    fars = torch.min(torch.max(t_min, t_max), dim=1)[0]

    return nears, fars

def cal_xyzs(rays_o, rays_d, aabb_infer, num_steps=16):
    pos0_ray_o = rays_o
    pos0_ray_d = rays_d
    N = pos0_ray_o.shape[0]

    nears, fars = compute_near_far(pos0_ray_o, pos0_ray_d, aabb_infer)
    nears = nears.unsqueeze_(-1)
    fars = fars.unsqueeze_(-1)

    # print(nears.shape, fars.shape)

    z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
    z_vals = z_vals.expand((N, num_steps)) # [N, T]
    # print(z_vals.shape)
    z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

    sample_dist = (fars - nears) / num_steps

    # generate xyzs
    xyzs = pos0_ray_o.unsqueeze(-2) + pos0_ray_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
    xyzs = torch.min(torch.max(xyzs, aabb_infer[:3]), aabb_infer[3:]) # a manual clip.

    # After that, we can also use density to accelerate the rendering process by selectively choosing the number of points.
    
    # Currently not implementing the mask one.
    deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
    deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
    
    return xyzs, deltas, z_vals, fars, nears

# prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
# NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
aabb_infer = model_weights['aabb_infer']

xyzs = cal_xyzs(rays_o[0], rays_d[0], aabb_infer)
# xyzs = xyzs[:2000,:,:]

def fast_hash_op(pos_grid):
    pos_grid = pos_grid.int()
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    result = torch.zeros(pos_grid.shape[0], pos_grid.shape[1], dtype=torch.int32).to(device)
    for i in range(3):
        result = torch.bitwise_xor(result, pos_grid[:, :, i] * primes[i])
    return result

def get_grid_index_op(C, D,align_corners, hashmap_size, resolution, pos_grid, ch=0):
    stride = torch.pow(resolution+1, torch.arange(3)).to(device)
    stride = stride.unsqueeze(0).expand((pos_grid.shape[0],pos_grid.shape[1], -1)) #[8,3]
    
    index = torch.empty([8,3],dtype=torch.int64, device=device)
    index = pos_grid * stride #[8,3]

     
    mask = (stride<=hashmap_size).to(device)
    index = torch.where(mask, index, torch.zeros_like(index))
    
    index = index.sum(dim=-1) #[8]
    stride = torch.tensor([[(resolution+1)**3]*pos_grid.shape[1]]*pos_grid.shape[0]).to(device)

    mask = (stride>hashmap_size).to(device) #[8]
    index = index.int()
    index = torch.where(mask, fast_hash_op(pos_grid), index)
    del mask, stride
    return (index%hashmap_size)*C + ch

def compute_hash_op(xyzs, offsets, embeddings):
    input_dim = 3
    multires = 6
    degree = 4
    num_levels = 16
    level_dim = 2
    base_resolution = 16
    log2_hashmap_size = 19
    desired_resolution = 4096
    align_corners = False
    per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))
    interpolation = 'linear'
    bound = 2.0

    # model_weights = torch.load('foxnerf/checkpoints/ngp_ep0307.pth', map_location=torch.device('cpu'))['model']
    
    # Normalize xyzs to [0,1]
    # xyz = (self.xyzs[self.temp_hit[sn]] - self.xyz_min) / (self.xyz_delta)
    xyzs = (xyzs + bound) / (2 * bound) # map to [0, 1]
    xyzs = xyzs.contiguous()

    B, D = xyzs.shape # batch size, coord dim
    L = offsets.shape[0] - 1 # level
    C = 2 # embedding dim for each level
    S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
    H = base_resolution # base resolution

    # embeddings = embeddings.flatten()

    len_sample = xyzs.shape[0]
    ans = torch.empty(len_sample, num_levels, C).to(device)


    # for sample_num in range(len_sample):
    #     inputs = xyzs[sample_num, :]

    inputs = xyzs

    for level in range(num_levels):
        hashmap_size = offsets[level + 1] - offsets[level]
        scale = 2**(S * level)*H - 1.0
        resolution = math.ceil(scale) + 1
    
        pos = inputs*scale + 0.5
        pos_grid = torch.floor(pos)
        pos -= pos_grid

        val0 = (1<<torch.arange(D))
        val0 = val0.expand((1<<D,pos.shape[0], -1))

        val1 = torch.arange(1<<D)
        val1 = val1.repeat((1<<D)//val1.shape[0], pos.shape[0]*D).reshape(1<<D, pos.shape[0], D)

        mask = ((val0 & val1)==0)
        # mask = mask.unsqueeze(0).expand((len_sample, -1, -1)).to(device)
        
        w = torch.where(mask, 1-pos, pos)
        w = w.prod(dim=-1)
        pos_grid_local = torch.where(mask, pos_grid, pos_grid + 1)
        # print(pos_grid_local.shape, pos_grid_local)
        # 8,1
        index = get_grid_index_op(C, D, align_corners, hashmap_size, resolution, pos_grid_local)

        local_feature0 = (w*embeddings[(int(offsets[level])*int(C) + index).long()]).sum(dim=0)
        local_feature1 = (w*embeddings[(int(offsets[level])*int(C) + index + 1).long()]).sum(dim=0)

        # ans[sample_num, level, :] = torch.tensor([local_feature0, local_feature1]).to(device)

        ans[:, level, :] = torch.cat([local_feature0[:,None], local_feature1[:, None]], dim=-1).to(device)
    del inputs,xyzs, pos, pos_grid, pos_grid_local, index, w, local_feature0, local_feature1, mask, val0, val1
    return ans
    
# %%
class sigma_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(32, 64, bias=False)
        self.linear2 = nn.Linear(64, 16, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        return x


# %%
def SHencoder(inputs):
    input_dim = 3
    degree = 4
    output_dim = 16

    inputs = inputs.view(-1, input_dim)
    B = inputs.shape[0]
    outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

    D = input_dim
    C = degree

    x,y,z = inputs[:, 0], inputs[:, 1], inputs[:, 2]

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

    outputs[:,0] = 0.28209479177387814  # 1/(2*sqrt(pi))
    outputs[:,1] = -0.48860251190291987 * y  # -sqrt(3)*y/(2*sqrt(pi))
    outputs[:,2] = 0.48860251190291987 * z  # sqrt(3)*z/(2*sqrt(pi))
    outputs[:,3] = -0.48860251190291987 * x  # -sqrt(3)*x/(2*sqrt(pi))
    outputs[:,4] = 1.0925484305920792 * xy  # sqrt(15)*xy/(2*sqrt(pi))
    outputs[:,5] = -1.0925484305920792 * yz  # -sqrt(15)*yz/(2*sqrt(pi))
    outputs[:,6] = 0.94617469575755997 * z2 - 0.31539156525251999  # sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
    outputs[:,7] = -1.0925484305920792 * xz  # -sqrt(15)*xz/(2*sqrt(pi))
    outputs[:,8] = 0.54627421529603959 * x2 - 0.54627421529603959 * y2  # sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    outputs[:,9] = 0.59004358992664352 * y * (-3.0 * x2 + y2)
    outputs[:,10] = 2.8906114426405538 * xy * z
    outputs[:,11] = 0.45704579946446572 * y * (1.0 - 5.0 * z2)
    outputs[:,12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0)
    outputs[:,13] = 0.45704579946446572 * x * (1.0 - 5.0 * z2)
    outputs[:,14] = 1.4453057213202769 * z * (x2 - y2)
    outputs[:,15] = 0.59004358992664352 * x * (-x2 + 3.0 * y2)
    
    return outputs

# %%
class color_net(nn.Module):
    def __init__(self) -> None:
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

# %%
def color(model_weights, dirs, density_outputs):
    sh_encoding = SHencoder(dirs)
    geo_feat = density_outputs['geo_feat']

    input = torch.cat([sh_encoding, geo_feat], dim=-1)
    model = color_net()
    model.linear1.weight.data = model_weights['color_net.0.weight']
    model.linear2.weight.data = model_weights['color_net.1.weight']
    model.linear3.weight.data = model_weights['color_net.2.weight']

    model = model.eval()
    with torch.no_grad():
        color = model(input)
    return color

# def compute_hash_for_sample(xyzs):
#     hash_encompute_hash(xyzs)
#     return hash_encodings_sample

# def compute_hash_mp(xyzs):
#     with mp.Pool(32) as pool:
#         hash_encodings_list = list(tqdm(pool.imap_unordered(compute_hash, xyzs), total=xyzs.shape[0]))
#     return torch.stack(hash_encodings_list)

    

# %%


# %%
# For testing purposes

# sample = torch.tensor([[0.8488,  0.5253, -0.0605]])
# sample_density = {
#     'geo_feat': torch.tensor([[-0.4214,  0.3730, -0.2864, -0.0939,  0.7633, -0.1327,  0.9101, -1.0568,
#         -0.2119, -0.8055,  0.5877,  0.1208,  0.3580, -0.6475,  0.6442]])
# }

# color(model_weights, sample, sample_density) 
# # 0.4864, 0.3541, 0.2778



if __name__=='__main__':
    # %%

    # add command line argument to take img
    parser = argparse.ArgumentParser(description='IDK')

    # Add the arguments
    parser.add_argument('img', type=int, help='The image number', default=0)


    num_levels = 128
    C = 2
    img = 0

    offsets = model_weights['encoder.offsets']
    embeddings = model_weights['encoder.embeddings'].flatten()

    offsets.share_memory_()
    embeddings.share_memory_()

    # def compute_hash_for_sample(sample):
    #     return compute_hash_op(xyzs[sample,:,:])

    # with mp.Pool(mp.cpu_count()) as pool:
    #     hash_encodings = list( tqdm(pool.imap(functools.partial(compute_hash_op, offsets=offsets, embeddings = embeddings), xyzs), total=xyzs.shape[0]))


    # xyzs = xyzs[:5000]
    num_steps = 16

    # rays_o = rays_o[:,:2000,:]
    # rays_d = rays_d[:,:2000,:]
    xyzs, deltas, z_vals, fars, nears = cal_xyzs(rays_o[img], rays_d[img], aabb_infer, num_steps=num_steps)
    torch.save([xyzs, deltas, z_vals, fars, nears], f'xyzs_{img}.pt')
    # xyzs = xyzs[:2000,:,:]
    samples = xyzs.shape[0]
    limit = 5120
    ans = []
    for i in tqdm(range(samples//limit+1)):
        with mp.Pool(16) as pool:
            if(i*limit==min((i+1)*limit, samples)):
                break
            val = xyzs[i*limit:min((i+1)*limit, samples),:,:]
            hash_encodings = list(pool.map(functools.partial(compute_hash_op, offsets=offsets, embeddings = embeddings), val))
            hash_encodings = torch.stack(hash_encodings)
            ans.append(hash_encodings)
        del hash_encodings, pool

    hash_encodings = torch.cat(ans, dim=0)
    torch.save(hash_encodings, f'hash_encodings_{img}_{num_steps}.pt')
    print("Successfully saved hash encodings")
    hash_encodings = hash_encodings.view(hash_encodings.shape[0]*hash_encodings.shape[1],-1)

    sigma = sigma_net()
    sigma.linear1.weight.data = model_weights['sigma_net.0.weight']
    sigma.linear2.weight.data = model_weights['sigma_net.1.weight']
    sigma = sigma.eval()

    # %%
    with torch.no_grad():
        print("Running density prediction")
        print("Hash Encoding Shape: ", hash_encodings.shape)
        sigma_net_pred = sigma(hash_encodings)
        print(sigma_net_pred.shape)
        density = torch.exp(sigma_net_pred[...,0])
        geo_feat = sigma_net_pred[...,1:]
        density_outputs = {
            'density': density,
            'geo_feat': geo_feat
        }
    print(density.shape, geo_feat.shape)

    # %%
    xyzs.shape

    # %%
    N = xyzs.shape[0]
    num_steps = xyzs.shape[1]
    # for k, v in density_outputs.items():
    #     density_outputs[k] = v.view(N, num_steps, -1)

    for k, v in density_outputs.items():
                density_outputs[k] = v.view(-1, v.shape[-1])

    rays_dir = rays_d[img]
    rays_dir = rays_dir[:,None,:].expand(-1, num_steps, -1)

    rgbs = color(model_weights, rays_dir.reshape(-1,3), density_outputs)

    torch.save(rgbs, f'rgb_{img}_{num_steps}.pt')
    torch.save(density_outputs, f'density_outputs_{img}_{num_steps}.pt')

    density_scale = 1

    density = density_outputs['density'].view(N, num_steps, -1)
    rgbs = rgbs.reshape(N, num_steps, 3)
    alphas = 1 - torch.exp(-deltas * density_scale * density.squeeze(-1)) # [N, T+t]
    alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
    weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

    weights_sum = weights.sum(dim=-1)

    ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
    depth = torch.sum(weights * ori_z_vals, dim=-1)

    # calculate color
    image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)

    torch.save(image, f'image_{img}_{num_steps}.pt')
    torch.save(depth, f'depth_{img}_{num_steps}.pt')

    image = image.reshape(1920,1080, 3)
    depth = depth.reshape(1920,1080)

    # Normalize the image to 255 and show it
    image = image.cpu().numpy()
    image = image / image.max()
    image = image * 255
    image = image.astype(np.uint8)
    import cv2
    cv2.imwrite(f'image_{img}.png', image)

    # Save depth
    depth = depth.cpu().numpy()
    depth = depth / depth.max()
    depth = depth * 255
    depth = depth.astype(np.uint8)
    cv2.imwrite(f'depth_{img}.png', depth)
    # %%
