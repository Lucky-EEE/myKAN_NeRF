import torch
import torch.nn as nn
import torch.nn.functional as F
import kan


class NeRF(nn.Module):
    def __init__(self, num_coarse_samples=64, num_fine_samples=128, near=0.1, far=100.0, width=[15, 15, 15, 15, 15]):
        super(NeRF, self).__init__()
        self.transformer = kan.KAN(width=width)
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.near = near
        self.far = far

    def positional_encoding(self, xyz, viewdirs=None):
        freq = 10
        xyz_enc = xyz
        for i in range(6):
            xyz_enc = torch.cat([xyz_enc, torch.sin(2.0 ** i * torch.pi * xyz),
                                 torch.cos(2.0 ** i * torch.pi * xyz)], dim=-1)
        if viewdirs is not None:
            viewdirs_enc = viewdirs
            for i in range(4):
                viewdirs_enc = torch.cat([viewdirs_enc, torch.sin(2.0 ** i * torch.pi * viewdirs),
                                          torch.cos(2.0 ** i * torch.pi * viewdirs)], dim=-1)
            return xyz_enc, viewdirs_enc
        return xyz_enc, None

    def render(self, xyz, viewdirs=None):
        xyz_enc, viewdirs_enc = self.positional_encoding(xyz, viewdirs)
        features = self.transformer(xyz_enc)
        rgb = torch.sigmoid(features[:, :3])
        sigma = F.relu(features[:, 3])
        return rgb, sigma

    def forward(self, rays_origin, rays_direction, viewdirs=None):
        z_vals, points = self.sample_along_rays(rays_origin, rays_direction)
        rgb, sigma = self.render(points, viewdirs)
        return self.volume_rendering(rgb, sigma, z_vals, rays_direction)

    def sample_along_rays(self, origins, directions):
        # Stratified sampling
        bins = torch.linspace(self.near, self.far, self.num_coarse_samples + 1).to(origins.device)
        z_vals = bins[:-1] + (bins[1:] - bins[:-1]) * torch.rand_like(bins[:-1])
        points = origins[:, None, :] + directions[:, None, :] * z_vals[:, :, None]
        return z_vals, points

    def volume_rendering(self, rgb, sigma, z_vals, directions):
        delta_z = z_vals[..., 1:] - z_vals[..., :-1]
        alpha = 1.0 - torch.exp(-sigma * delta_z)
        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
                                        dim=-1)[:, :-1]
        rgb_rendered = torch.sum(weights[..., None] * rgb, dim=-2)
        return rgb_rendered