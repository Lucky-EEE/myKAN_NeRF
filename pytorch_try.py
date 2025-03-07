import os
# 如果知道正确的 DLL 路径，可以添加
# os.environ['PATH'] += os.pathsep + 'D:\\Library\\bin'  # 替换为实际路径
import numpy as np
import trimesh
import pyrender
import imageio
import open3d as o3d
from IPython.display import HTML
from base64 import b64encode
import matplotlib.pyplot as plt
import torch

# Load point cloud
points_3d = np.loadtxt('points_3d.txt')
print("Original point cloud shape:", points_3d.shape)

# Load NeRF model
model_path = 'model_state_dict.pth'  # 确保路径正确
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)  # 加载模型权重
model.eval()
# 继续后续代码...

# 假设你的 NeRF 模型接受 3D 点和视角方向作为输入
def query_nerf(points, model, device):
    points_tensor = torch.from_numpy(points).float().to(device)
    viewdirs = torch.zeros_like(points_tensor)  # 简化：无视角依赖
    with torch.no_grad():
        raw = model(points_tensor, viewdirs)  # 输出 [N, 4]，RGB + sigma
    return raw.cpu().numpy()

# Query NeRF for colors and density
points_flat = points_3d.reshape(-1, 3)
raw = query_nerf(points_flat, model, device)
rgb = raw[:, :3]  # RGB颜色
sigma = raw[:, 3]  # 密度

# Filter points with significant density
threshold = 10.0  # 调整阈值以适应透明瓶子
mask = sigma > threshold
points_filtered = points_3d[mask]
colors = rgb[mask]

# Generate mesh
hull = ConvexHull(points_filtered)
hull_vertices = points_filtered[hull.vertices]
hull_colors = colors[hull.vertices]
vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
hull_faces = np.array([[vertex_map[idx] for idx in face] for face in hull.simplices])
mesh = trimesh.Trimesh(vertices=hull_vertices, faces=hull_faces, vertex_colors=hull_colors)
print("Mesh created with", len(mesh.vertices), "vertices and", len(mesh.faces), "faces")

# Create scene
scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])
pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
scene.add(pyrender_mesh)

# Load camera poses
cameras = np.load('cameras.npy')
camera_poses_4x4 = np.array([np.eye(4) if pose.shape != (4, 4) else pose for pose in cameras])
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = camera_poses_4x4[0]
camera_pose[:3, 3] = [0, 0, 35]
nc = pyrender.Node(camera=camera, matrix=camera_pose)
scene.add_node(nc)

# Add light
light_pose = np.eye(4)
light_pose[:3, 3] = [0, 0, 35]
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=100.0)
nl = pyrender.Node(light=light, matrix=light_pose)
scene.add_node(nl)

# Render video
r = pyrender.OffscreenRenderer(640, 480)
imgs = []
for i, camera_pose in enumerate(camera_poses_4x4):
    scene.set_pose(nc, pose=camera_pose)
    color, _ = r.render(scene)
    imgs.append(color)
    print(f"Rendered frame {i+1}/{len(camera_poses_4x4)}")

# Preview first frame
plt.imshow(imgs[0])
plt.title("First Frame with NeRF Colors")
plt.show()

# Save video
output_file = 'output_video_nerf.mp4'
writer = imageio.get_writer(output_file, format='FFMPEG', mode='I', fps=30, codec='libx264')
for img in imgs:
    writer.append_data(img)
writer.close()
print(f"Video saved as {output_file}")

# Embed video
with open(output_file, 'rb') as f:
    mp4 = f.read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
display_html = HTML("""
<video width=400 controls autoplay loop>
    <source src="%s" type="video/mp4">
</video>
""" % data_url)
display_html