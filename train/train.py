import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import kan  # Ensure kan is installed


# Load camera and 3D point data
def load_camera_data(cameras_path, points_3d_path):
    cameras = np.load(cameras_path)
    points_3d = np.loadtxt(points_3d_path)
    return cameras, points_3d


# Custom dataset for NeRF
class NeRFDataset(Dataset):
    def __init__(self, cameras, points_3d):
        self.cameras = cameras
        self.points_3d = points_3d

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        camera_pose = self.cameras[idx]
        points_3d = self.points_3d[idx % len(self.points_3d)]  # Adjust if points_3d length differs
        input_data = np.concatenate([camera_pose.flatten(), points_3d.flatten()])
        input_data = torch.tensor(input_data, dtype=torch.float32)
        target_data = input_data.clone()  # Adjust based on actual task
        return input_data, target_data


# Training function
def train_nerf(model, dataloader, optimizer, criterion, scheduler, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (input_data, target_data) in enumerate(dataloader):
            input_data = input_data.to(device)
            target_data = target_data.to(device)

            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target_data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")
        scheduler.step()


# Main setup and execution
def main():
    cameras_path = r"../preprocess_result/cameras.npy"
    points_3d_path = r"../preprocess_result/points_3d.txt"

    cameras, points_3d = load_camera_data(cameras_path, points_3d_path)

    # Normalize data
    scaler = MinMaxScaler()
    cameras = scaler.fit_transform(cameras.reshape(-1, cameras.shape[-1])).reshape(cameras.shape)
    points_3d = scaler.fit_transform(points_3d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = kan.KAN(width=[15, 15, 15, 15, 15]).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    dataset = NeRFDataset(cameras=cameras, points_3d=points_3d)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    train_nerf(model, dataloader, optimizer, criterion, scheduler, device, epochs=10)

    torch.save(model.state_dict(), '../model/model_state_dict.pth')


if __name__ == "__main__":
    main()