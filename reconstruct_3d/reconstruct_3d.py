import torch
import numpy as np
import kan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = kan.KAN(width=[15, 15, 15, 15, 15]).to(device)
model.load_state_dict(torch.load(r'../model/model_state_dict.pth'))
model.eval()


# Generate rays based on camera parameters
def generate_rays(camera_pose, img_width, img_height, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(img_width), np.arange(img_height))
    u = u.flatten()
    v = v.flatten()
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    ray_dirs = (R.T @ dirs.T).T
    ray_origins = np.broadcast_to(t, ray_dirs.shape)
    return ray_origins, ray_dirs


# Reconstruct scene with batched inference
def reconstruct_3d_scene(model, camera_pose, img_width, img_height, K, device, batch_size=1000):
    ray_origins, ray_dirs = generate_rays(camera_pose, img_width, img_height, K)
    ray_origins = torch.tensor(ray_origins, dtype=torch.float32, device=device)
    ray_dirs = torch.tensor(ray_dirs, dtype=torch.float32, device=device)

    num_rays = ray_origins.shape[0]
    reconstructed_points = []

    for i in range(0, num_rays, batch_size):
        batch_origins = ray_origins[i:i + batch_size]
        batch_dirs = ray_dirs[i:i + batch_size]
        input_data = torch.cat([batch_origins, batch_dirs], dim=-1)
        with torch.no_grad():
            output = model(input_data)
        reconstructed_points.append(output.cpu().numpy())

    return np.concatenate(reconstructed_points, axis=0)


# Render the reconstructed scene (placeholder)
def render_scene(reconstructed_points):
    # Implement rendering logic as needed
    pass


# Main execution
def main():
    camera_pose = np.eye(4)  # Replace with actual camera pose
    img_width, img_height = 1920, 1080  # Replace with actual image dimensions
    K = np.array([[5851.35, 0, 960], [0, 5851.35, 540], [0, 0, 1]])  # Replace with actual intrinsics

    reconstructed_points = reconstruct_3d_scene(model, camera_pose, img_width, img_height, K, device)
    render_scene(reconstructed_points)


if __name__ == "__main__":
    main()