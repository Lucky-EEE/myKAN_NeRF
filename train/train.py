import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import kan  # 导入 pykan

# 加载相机数据和 3D 点云数据
def load_camera_data(cameras_path, points_3d_path):
    cameras = np.load(cameras_path)  # 加载相机位姿数据
    points_3d = np.loadtxt(points_3d_path)  # 加载3D点云数据
    return cameras, points_3d

# 自定义数据集类
class NeRFDataset(Dataset):
    def __init__(self, cameras, points_3d):
        self.cameras = cameras  # 相机位姿
        self.points_3d = points_3d  # 3D 点云数据

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        camera_pose = self.cameras[idx]  # 获取第 idx 个相机的位姿
        points_3d = self.points_3d[idx]  # 获取对应的 3D 点云数据

        # 处理为模型需要的输入（这里简化，假设只是简单的拼接）
        input_data = np.concatenate([camera_pose.flatten(), points_3d.flatten()])
        input_data = torch.tensor(input_data, dtype=torch.float32)

        # 假设模型需要的目标是渲染的图像，这里用 input_data 作为输出目标（你可以根据需求更改目标）
        target_data = input_data.clone()  # 示例：目标数据与输入相同（根据任务调整）

        return input_data, target_data

# 训练函数
def train_nerf(model, dataloader, optimizer, criterion, scheduler, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_data, target_data) in enumerate(dataloader):
            input_data = input_data.float().to(device)
            target_data = target_data.float().to(device)

            optimizer.zero_grad()

            # 前向传播
            output = model(input_data)

            # 计算损失
            criterion = torch.nn.SmoothL1Loss()
            loss = criterion(output, target_data)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}")

        # 每个epoch结束后打印平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}")

        # 更新学习率
        scheduler.step()

# 设置训练参数
def main():
    # 文件路径
    cameras_path = r"../preprocess_result/cameras.npy"
    points_3d_path = r"../preprocess_result/points_3d.txt"

    # 加载数据
    cameras, points_3d = load_camera_data(cameras_path, points_3d_path)

    # 数据归一化
    scaler = MinMaxScaler()
    cameras = scaler.fit_transform(cameras.reshape(-1, cameras.shape[-1])).reshape(cameras.shape)
    points_3d = scaler.fit_transform(points_3d)

    # 定义设备：使用 GPU（如果有）或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型
    model = kan.KAN(width=[15, 15, 15, 15, 15]).to(device)  # 根据实际需求调整输入和输出通道数

    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)  # 设置初始学习率为 1e-5,同时添加L2正则化

    # 学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch衰减学习率

    # 创建数据集和数据加载器
    dataset = NeRFDataset(cameras=cameras, points_3d=points_3d)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 训练模型
    train_nerf(model, dataloader, optimizer, criterion, scheduler, device, epochs=10)

    # 保存模型的state_dict
    torch.save(model.state_dict(), '../model/model_state_dict.pth')

# 运行训练
if __name__ == "__main__":
    main()
