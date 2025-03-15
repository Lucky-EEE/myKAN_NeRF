import torch
import torch.nn as nn
import torch.nn.functional as F
import kan

class NeRF(nn.Module):
    def __init__(self, num_coarse_samples=64, num_fine_samples=128, near=0.1, far=100.0):
        super(NeRF, self).__init__()
        width = [63, 128, 128, 64, 4]  # 调整网络容量
        self.transformer = kan.KAN(width=width)
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.near = near
        self.far = far

    def positional_encoding(self, xyz, viewdirs=None):
        xyz_enc = xyz
        for i in range(4):
            xyz_enc = torch.cat([xyz_enc, torch.sin(2.0 ** i * torch.pi * xyz),
                                 torch.cos(2.0 ** i * torch.pi * xyz)], dim=-1)
        if viewdirs is not None:
            viewdirs_enc = viewdirs
            for i in range(2):
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
        rgb_rendered = self.volume_rendering(rgb, sigma, z_vals, rays_direction)
        return rgb_rendered, sigma, points  # 返回 sigma 和 points 用于正则化

    def sample_along_rays(self, origins, directions):
        z_vals_coarse, points_coarse = self._stratified_sample(origins, directions, self.num_coarse_samples)
        rgb_coarse, sigma_coarse = self.render(points_coarse)
        weights = self._compute_weights(sigma_coarse, z_vals_coarse)
        z_vals_fine = self._importance_sample(z_vals_coarse, weights, self.num_fine_samples)
        points_fine = origins[:, None, :] + directions[:, None, :] * z_vals_fine[:, :, None]
        return z_vals_fine, points_fine

    def _stratified_sample(self, origins, directions, num_samples):
        bins = torch.linspace(self.near, self.far, num_samples + 1).to(origins.device)
        z_vals = bins[:-1] + (bins[1:] - bins[:-1]) * torch.rand_like(bins[:-1])
        points = origins[:, None, :] + directions[:, None, :] * z_vals[:, :, None]
        return z_vals, points

    def _compute_weights(self, sigma, z_vals):
        delta_z = z_vals[..., 1:] - z_vals[..., :-1]
        alpha = 1.0 - torch.exp(-sigma[..., :-1] * delta_z)
        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
        return weights

    def _importance_sample(self, z_vals_coarse, weights, num_samples):
        z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
        z_vals_fine = torch.zeros(z_vals_coarse.shape[0], num_samples).to(z_vals_coarse.device)
        for i in range(z_vals_coarse.shape[0]):
            z_vals_fine[i] = torch.multinomial(weights[i], num_samples, replacement=True).float() / weights.shape[-1] * (self.far - self.near) + self.near
        return z_vals_fine

    def volume_rendering(self, rgb, sigma, z_vals, directions):
        delta_z = z_vals[..., 1:] - z_vals[..., :-1]
        alpha = 1.0 - torch.exp(-sigma * delta_z)
        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_rendered = torch.sum(weights[..., None] * rgb, dim=-2)
        return rgb_rendered

# 示例训练循环
model = NeRF()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    rgb_rendered, sigma, points = model(rays_origin, rays_direction)
    recon_loss = F.mse_loss(rgb_rendered, target_rgb)
    sigma_loss = torch.mean(sigma ** 2) * 0.01  # 正则化 sigma
    smooth_loss = torch.mean(torch.autograd.grad(sigma.sum(), points, create_graph=True)[0] ** 2) * 0.001  # 平滑性正则化
    loss = recon_loss + sigma_loss + smooth_loss
    loss.backward()
    optimizer.step()

    # 每隔一定周期生成新视角
    if epoch % 50 == 0 and epoch > 0:
        new_origins, new_directions, rgb_new = generate_new_views(model, rays_origin, rays_direction)
        rays_origin = torch.cat([rays_origin, new_origins], dim=0)
        rays_direction = torch.cat([rays_direction, new_directions], dim=0)
        target_rgb = torch.cat([target_rgb, rgb_new], dim=0)
