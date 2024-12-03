import torch
import torch.nn as nn
import torch.nn.functional as F
import kan


class NeRF(nn.Module):
    def __init__(self, num_coarse_samples=64, num_fine_samples=128, near=0.1, far=100.0, width=[15, 15, 15, 15, 15]):
        super(NeRF, self).__init__()

        # 使用 KAN 作为 NeRF 的主干网络
        self.transformer = kan.KAN(width=width)

        # 参数设置
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.near = near
        self.far = far

    def positional_encoding(self, xyz, viewdirs=None):
        """
        位置编码：对输入的三维坐标进行频率编码，以增强模型的表示能力。
        """
        # 对 xyz 进行正弦和余弦编码
        freq = 10
        xyz_enc = xyz
        for i in range(6):  # 使用不同频率的正弦和余弦编码
            xyz_enc = torch.cat([xyz_enc, torch.sin(2.0 ** i * torch.pi * xyz),
                                 torch.cos(2.0 ** i * torch.pi * xyz)], dim=-1)

        if viewdirs is not None:
            # 对视角方向进行编码
            viewdirs_enc = viewdirs
            for i in range(4):
                viewdirs_enc = torch.cat([viewdirs_enc, torch.sin(2.0 ** i * torch.pi * viewdirs),
                                          torch.cos(2.0 ** i * torch.pi * viewdirs)], dim=-1)
            return xyz_enc, viewdirs_enc
        else:
            return xyz_enc, None

    def render(self, xyz, viewdirs=None):
        """
        使用 KAN 模型进行渲染，生成每个点的颜色和密度。
        """
        # 对输入位置进行位置编码
        xyz_enc, viewdirs_enc = self.positional_encoding(xyz, viewdirs)

        # 使用 KAN 模型进行前向推理，得到颜色和密度
        features = self.transformer(xyz_enc)

        # 假设 transformer 输出前三个值是 RGB 颜色，最后一个值是密度
        rgb = torch.sigmoid(features[:, :3])  # RGB值
        sigma = F.relu(features[:, 3])  # 密度

        return rgb, sigma

    def forward(self, rays_origin, rays_direction, viewdirs=None):
        """
        对于每条射线，进行采样并计算颜色和密度，最终合成图像。
        """
        # 计算射线的采样点
        z_vals, points = self.sample_along_rays(rays_origin, rays_direction)

        # 渲染样本点
        rgb, sigma = self.render(points, viewdirs)

        # 体积渲染公式：通过颜色和密度进行光线积分
        return self.volume_rendering(rgb, sigma, z_vals, rays_direction)

    def sample_along_rays(self, origins, directions):
        """
        沿着射线进行采样，返回采样的深度值和采样点。
        """
        # 这里简单地进行均匀采样
        z_vals = torch.linspace(self.near, self.far, self.num_coarse_samples).to(origins.device)
        points = origins[:, None, :] + directions[:, None, :] * z_vals[:, :, None]
        return z_vals, points

    def volume_rendering(self, rgb, sigma, z_vals, directions):
        """
        体积渲染：根据每个采样点的颜色和密度值，沿射线积累颜色。
        """
        # 计算体积渲染权重
        delta_z = z_vals[..., 1:] - z_vals[..., :-1]
        alpha = 1.0 - torch.exp(-sigma * delta_z)
        weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]

        # 渲染颜色
        rgb_rendered = torch.sum(weights[..., None] * rgb, dim=-2)

        return rgb_rendered
