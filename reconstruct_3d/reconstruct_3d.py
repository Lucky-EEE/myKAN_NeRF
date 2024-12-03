import torch
import numpy as np
import kan
from torch.utils.data import DataLoader

# 设置设备为 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model = kan.KAN(width=[15, 15, 15, 15, 15]).to(device)
model.load_state_dict(torch.load(r'../model/model_state_dict.pth'))
model.eval()

# 生成射线的函数
def generate_rays(camera_pose, points_3d):
    """
    给定相机位姿和3D点云，生成射线
    """
    rays = []
    for point in points_3d:
        # 使用相机的位姿矩阵将三维点投影到相机的视野内
        ray_origin = camera_pose[:3, 3]  # 相机位置
        point_tensor = torch.tensor(point, dtype=torch.float32, device=device)
        ray_direction = point_tensor - ray_origin  # 射线方向
        rays.append((ray_origin, ray_direction))
    return rays

# 基于射线对场景进行重建
def reconstruct_3d_scene(model, camera_pose, points_3d, device):
    """
    使用训练好的NeRF模型进行三维重建
    """
    rays = generate_rays(camera_pose, points_3d)

    reconstructed_points = []
    for ray_origin, ray_direction in rays:
        # 使用KAN模型进行推断
        input_data = torch.cat([ray_origin.flatten(), ray_direction.flatten()]).unsqueeze(0).to(device)
        output = model(input_data)

        # 获取模型的预测结果（颜色和密度等）
        color = output[:, :3].cpu().detach().numpy()  # 假设前三个输出为RGB
        density = output[:, 3].cpu().detach().numpy()  # 假设第四个输出为密度

        reconstructed_points.append((ray_origin.cpu().numpy(), ray_direction.cpu().numpy(), color, density))

    return reconstructed_points

# 渲染和显示重建的场景
def render_scene(reconstructed_points):
    """
    根据重建的点，渲染最终图像
    """
    # 这里你可以实现一个简单的渲染函数，将3D点投影到2D图像上
    # 使用颜色和密度等信息合成最终图像
    pass

# 主函数
def main():
    # 假设你已经加载了相机位姿和点云数据
    camera_pose = np.eye(4)  # 相机位姿矩阵 (此处为示例，替换为实际值)
    points_3d = np.random.rand(100, 3)  # 100个3D点（示例数据，替换为实际点云）

    # 将相机位姿矩阵转换为tensor
    camera_pose_tensor = torch.tensor(camera_pose, dtype=torch.float32).to(device)

    # 使用训练后的模型进行三维重建
    reconstructed_points = reconstruct_3d_scene(model, camera_pose_tensor, points_3d, device)

    # 渲染并显示重建的场景
    render_scene(reconstructed_points)


if __name__ == "__main__":
    main()
