import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import sys
import os
from tqdm import tqdm  # 导入 tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.NeRF import NeRF, generate_new_views


# Load camera and 3D point data
def load_camera_data(cameras_path, points_3d_path):
    cameras = np.load(cameras_path)
    points_3d = np.loadtxt(points_3d_path)
    return cameras, points_3d


def generate_rays(camera_pose, img_width=800, img_height=600, focal_length=1000.0):
    """从相机姿态生成光线原点和方向"""
    # 相机内参
    cx = img_width / 2
    cy = img_height / 2
    fx = fy = focal_length
    
    # 生成像素网格
    i, j = np.meshgrid(
        np.arange(img_width, dtype=np.float32),
        np.arange(img_height, dtype=np.float32),
        indexing='xy'
    )
    
    # 计算归一化的方向向量
    directions = np.stack([
        (i - cx) / fx,
        -(j - cy) / fy,
        -np.ones_like(i)
    ], axis=-1)
    
    # 应用相机旋转
    rotation = camera_pose[:3, :3]
    directions = np.dot(directions, rotation.T)
    
    # 归一化方向向量
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    
    # 获取相机位置（光线原点）
    origins = np.broadcast_to(camera_pose[:3, 3], directions.shape)
    
    return origins, directions


# Custom dataset for NeRF
class NeRFDataset(Dataset):
    def __init__(self, cameras, points_3d, img_width=800, img_height=600, focal_length=1000.0):
        self.cameras = cameras
        self.points_3d = points_3d
        self.img_width = img_width
        self.img_height = img_height
        self.focal_length = focal_length

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        camera_pose = self.cameras[idx]
        
        # 生成光线
        origins, directions = generate_rays(
            camera_pose, 
            self.img_width, 
            self.img_height, 
            self.focal_length
        )
        
        # 采样一些光线进行训练
        num_rays = 1024  # 每批次采样的光线数
        select_inds = np.random.choice(
            origins.shape[0] * origins.shape[1], 
            size=[num_rays], 
            replace=False
        )
        
        origins = origins.reshape(-1, 3)[select_inds]
        directions = directions.reshape(-1, 3)[select_inds]
        
        # 使用最近的3D点作为目标RGB值（简化处理）
        target_rgb = np.zeros((num_rays, 3))
        for i, (o, d) in enumerate(zip(origins, directions)):
            # 找到最近的3D点
            distances = np.linalg.norm(self.points_3d - o, axis=1)
            nearest_idx = np.argmin(distances)
            target_rgb[i] = self.points_3d[nearest_idx]  # 假设points_3d包含RGB值
        
        return {
            'rays_origin': torch.FloatTensor(origins),
            'rays_direction': torch.FloatTensor(directions),
            'target_rgb': torch.FloatTensor(target_rgb)
        }


# Training function
def train_nerf(model, train_dataloader, optimizer, device, num_epochs=10, 
               generate_views_frequency=50, save_model_frequency=5):
    """改进的训练函数"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_rgb_loss = 0
        total_sigma_loss = 0
        total_smooth_loss = 0
        
        # 使用 tqdm 包装数据加载器
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            rays_origin = batch['rays_origin'].to(device)
            rays_direction = batch['rays_direction'].to(device)
            target_rgb = batch['target_rgb'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(rays_origin, rays_direction)
            
            # 计算损失
            loss_dict = model.compute_loss(outputs, target_rgb)
            loss = loss_dict['total_loss']
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            total_rgb_loss += loss_dict['rgb_loss'].item()
            total_sigma_loss += loss_dict['sigma_loss'].item()
            total_smooth_loss += loss_dict['smooth_loss'].item()
            
            # 打印训练信息
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_dataloader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"RGB Loss: {loss_dict['rgb_loss'].item():.4f} "
                      f"Sigma Loss: {loss_dict['sigma_loss'].item():.4f} "
                      f"Smooth Loss: {loss_dict['smooth_loss'].item():.4f}")
            
            # 生成新视角数据
            if epoch > 0 and batch_idx % generate_views_frequency == 0:
                new_origins, new_directions, rgb_new = generate_new_views(
                    model, rays_origin, rays_direction)
                
                # 将新数据添加到当前batch
                rays_origin = torch.cat([rays_origin, new_origins], dim=0)
                rays_direction = torch.cat([rays_direction, new_directions], dim=0)
                target_rgb = torch.cat([target_rgb, rgb_new], dim=0)
        
        # 计算平均损失
        avg_loss = total_loss / len(train_dataloader)
        avg_rgb_loss = total_rgb_loss / len(train_dataloader)
        avg_sigma_loss = total_sigma_loss / len(train_dataloader)
        avg_smooth_loss = total_smooth_loss / len(train_dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Average Loss: {avg_loss:.4f}, "
              f"RGB Loss: {avg_rgb_loss:.4f}, "
              f"Sigma Loss: {avg_sigma_loss:.4f}, "
              f"Smooth Loss: {avg_smooth_loss:.4f}")
        
        # 保存模型
        if (epoch + 1) % save_model_frequency == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'../model/nerf_checkpoint_epoch_{epoch+1}.pth')


# Main setup and execution
def main():
    # 设置路径
    cameras_path = r"../preprocess_result/cameras.npy"
    points_3d_path = r"../preprocess_result/points_3d.txt"

    # 加载数据
    cameras, points_3d = load_camera_data(cameras_path, points_3d_path)

    # 数据预处理
    scaler = MinMaxScaler()
    cameras = scaler.fit_transform(cameras.reshape(-1, cameras.shape[-1])).reshape(cameras.shape)
    points_3d = scaler.fit_transform(points_3d)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型
    model = NeRF(
        num_coarse_samples=64,
        num_fine_samples=128,
        near=0.1,
        far=100.0
    ).to(device)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # 创建数据集和数据加载器
    dataset = NeRFDataset(
        cameras=cameras,
        points_3d=points_3d,
        img_width=800,
        img_height=600,
        focal_length=1000.0
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 每个batch包含一个视角的采样光线
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 训练模型
    train_nerf(
        model=model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=50,
        generate_views_frequency=50,
        save_model_frequency=5
    )


if __name__ == "__main__":
    main()