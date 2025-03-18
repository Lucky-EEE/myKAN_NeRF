import torch
import torch.nn as nn
import torch.nn.functional as F
import kan

class NeRF(nn.Module):
    def __init__(self, num_coarse_samples=64, num_fine_samples=128, near=0.1, far=100.0):
        super(NeRF, self).__init__()
        # 增加网络容量以更好地处理稀疏输入
        width = [63, 128, 128, 64, 4]  # 输入维度63（3D坐标 + 位置编码），输出4（RGB + sigma）
        self.transformer = kan.KAN(width=width)
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.near = near
        self.far = far
        
        # 初始化正则化超参数
        self.lambda_sigma = 0.01  # sigma正则化权重
        self.lambda_smooth = 0.001  # 平滑性正则化权重

    def positional_encoding(self, xyz, viewdirs=None):
        """改进的位置编码，降低频率范围以减少对稀疏输入的敏感性"""
        xyz_enc = xyz
        for i in range(4):  # 减少到4个频率
            xyz_enc = torch.cat([xyz_enc, 
                               torch.sin(2.0 ** i * torch.pi * xyz),
                               torch.cos(2.0 ** i * torch.pi * xyz)], dim=-1)
        if viewdirs is not None:
            viewdirs_enc = viewdirs
            for i in range(2):  # 减少到2个频率
                viewdirs_enc = torch.cat([viewdirs_enc,
                                        torch.sin(2.0 ** i * torch.pi * viewdirs),
                                        torch.cos(2.0 ** i * torch.pi * viewdirs)], dim=-1)
            return xyz_enc, viewdirs_enc
        return xyz_enc, None

    def render(self, xyz, viewdirs=None):
        xyz_enc, viewdirs_enc = self.positional_encoding(xyz, viewdirs)
        features = self.transformer(xyz_enc)
        rgb = torch.sigmoid(features[:, :3])
        sigma = F.softplus(features[:, 3])  # 使用softplus替代ReLU，产生更平滑的密度场
        return rgb, sigma

    def forward(self, rays_origin, rays_direction, viewdirs=None):
        """改进的前向传播，包含分层采样策略"""
        # 粗采样
        z_vals_coarse, points_coarse = self._stratified_sample(rays_origin, rays_direction, self.num_coarse_samples)
        rgb_coarse, sigma_coarse = self.render(points_coarse, viewdirs)
        weights_coarse = self._compute_weights(sigma_coarse, z_vals_coarse)
        rgb_rendered_coarse = self.volume_rendering(rgb_coarse, sigma_coarse, z_vals_coarse, rays_direction)
        
        # 细采样（基于重要性采样）
        z_vals_fine = self._importance_sample(z_vals_coarse, weights_coarse, self.num_fine_samples)
        points_fine = rays_origin[:, None, :] + rays_direction[:, None, :] * z_vals_fine[:, :, None]
        rgb_fine, sigma_fine = self.render(points_fine, viewdirs)
        rgb_rendered_fine = self.volume_rendering(rgb_fine, sigma_fine, z_vals_fine, rays_direction)
        
        return {
            'rgb_coarse': rgb_rendered_coarse,
            'rgb_fine': rgb_rendered_fine,
            'sigma_coarse': sigma_coarse,
            'sigma_fine': sigma_fine,
            'points_coarse': points_coarse,
            'points_fine': points_fine,
            'weights_coarse': weights_coarse
        }

    def _stratified_sample(self, origins, directions, num_samples):
        """改进的分层采样，加入随机扰动"""
        # 确保输入张量具有正确的维度
        if origins.dim() == 1:
            origins = origins.unsqueeze(0)  # 添加批次维度
        if directions.dim() == 1:
            directions = directions.unsqueeze(0)  # 添加批次维度
        
        # 如果输入是2D张量，添加采样维度
        if origins.dim() == 2:
            origins = origins.unsqueeze(1)  # [B, 1, 3]
        if directions.dim() == 2:
            directions = directions.unsqueeze(1)  # [B, 1, 3]

        # 生成采样点
        bins = torch.linspace(self.near, self.far, num_samples + 1).to(origins.device)
        z_vals = bins[:-1] + (bins[1:] - bins[:-1]) * torch.rand_like(bins[:-1])
        z_vals = z_vals.expand(origins.shape[0], num_samples)  # [B, N]
        
        # 计算采样点的3D坐标
        points = origins + directions * z_vals.unsqueeze(-1)  # [B, N, 3]
        
        return z_vals, points

    def _compute_weights(self, sigma, z_vals):
        """计算体素权重"""
        delta_z = z_vals[..., 1:] - z_vals[..., :-1]
        alpha = 1.0 - torch.exp(-sigma * delta_z)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        return weights

    def _importance_sample(self, z_vals_coarse, weights, num_samples):
        """改进的重要性采样"""
        z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
        weights = weights + 1e-5  # 防止权重为0
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # 均匀采样
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(z_vals_coarse.device)
        
        # 反演采样方法
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(inds, 0, cdf.shape[-1] - 1)
        inds_g = torch.stack([below, above], dim=-1)
        
        cdf_g = torch.gather(cdf, -1, inds_g)
        z_vals_g = torch.gather(torch.cat([z_vals_mid, z_vals_mid[..., -1:]], dim=-1), -1, inds_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        z_vals_fine = z_vals_g[..., 0] + t * (z_vals_g[..., 1] - z_vals_g[..., 0])
        
        return z_vals_fine

    def volume_rendering(self, rgb, sigma, z_vals, directions):
        """体积渲染"""
        delta_z = z_vals[..., 1:] - z_vals[..., :-1]
        alpha = 1.0 - torch.exp(-sigma * delta_z)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        rgb_rendered = torch.sum(weights[..., None] * rgb, dim=-2)
        return rgb_rendered

    def compute_loss(self, outputs, target_rgb):
        """计算损失函数，包含多个正则化项"""
        # 重建损失
        rgb_loss_coarse = F.mse_loss(outputs['rgb_coarse'], target_rgb)
        rgb_loss_fine = F.mse_loss(outputs['rgb_fine'], target_rgb)
        rgb_loss = rgb_loss_coarse + rgb_loss_fine
        
        # Sigma正则化（鼓励稀疏性）
        sigma_loss = (torch.mean(outputs['sigma_coarse'] ** 2) + 
                     torch.mean(outputs['sigma_fine'] ** 2)) * self.lambda_sigma
        
        # 平滑性正则化
        grad_coarse = torch.autograd.grad(
            outputs['sigma_coarse'].sum(), 
            outputs['points_coarse'],
            create_graph=True
        )[0]
        grad_fine = torch.autograd.grad(
            outputs['sigma_fine'].sum(), 
            outputs['points_fine'],
            create_graph=True
        )[0]
        smooth_loss = (torch.mean(grad_coarse ** 2) + 
                      torch.mean(grad_fine ** 2)) * self.lambda_smooth
        
        # 总损失
        total_loss = rgb_loss + sigma_loss + smooth_loss
        
        return {
            'total_loss': total_loss,
            'rgb_loss': rgb_loss,
            'sigma_loss': sigma_loss,
            'smooth_loss': smooth_loss
        }

def generate_new_views(model, rays_origin, rays_direction, num_new_views=5):
    """生成新的视角数据用于自监督学习"""
    with torch.no_grad():
        # 从已有光线方向随机扰动生成新视角
        perturb = torch.randn_like(rays_direction) * 0.1
        new_directions = F.normalize(rays_direction + perturb, dim=-1)
        
        # 小范围扰动原点
        new_origins = rays_origin + torch.randn_like(rays_origin) * 0.05
        
        # 使用当前模型生成新视角的RGB值
        outputs = model(new_origins, new_directions)
        rgb_new = outputs['rgb_fine']  # 使用精细网络的输出
        
    return new_origins, new_directions, rgb_new
