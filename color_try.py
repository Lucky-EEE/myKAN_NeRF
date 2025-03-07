import os
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import pyrender
import imageio
import open3d as o3d
from IPython.display import HTML
from base64 import b64encode
import matplotlib.pyplot as plt


# Function to remove outliers
def remove_outliers_radius(file_path, nb_points=4, radius=0.5):
    print(f"Loading point cloud from {file_path}")
    points_3d = np.loadtxt(file_path)
    print("Original point cloud shape:", points_3d.shape)
    print("Sample points:", points_3d[:5])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    _, ind = point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_cloud = point_cloud.select_by_index(ind)

    print("Points after outlier removal:", len(inlier_cloud.points))
    o3d.visualization.draw_geometries([inlier_cloud])

    cleaned_points = np.asarray(inlier_cloud.points)
    np.savetxt("inliers_points_3d.txt", cleaned_points)
    return cleaned_points


# Function to convert 3x4 to 4x4 pose
def convert_to_4x4(pose_3x4):
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :] = pose_3x4
    return pose_4x4


# Main workflow
points_3d = remove_outliers_radius('points_3d.txt', nb_points=4, radius=0.5)
print("Cleaned point cloud shape:", points_3d.shape)

cameras = np.load('cameras.npy')
print("Original camera poses shape:", cameras.shape)
camera_poses_4x4 = np.array([convert_to_4x4(pose) for pose in cameras])
print("Converted camera poses shape:", camera_poses_4x4.shape)

# Generate convex hull
hull = ConvexHull(points_3d)
hull_vertices = points_3d[hull.vertices]
vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
hull_faces = np.array([[vertex_map[idx] for idx in face] for face in hull.simplices])
print("Hull faces shape:", hull_faces.shape, "Max index in faces:", hull_faces.max())

# Create mesh
mesh = trimesh.Trimesh(vertices=hull_vertices, faces=hull_faces)
print("Mesh created with", len(mesh.vertices), "vertices and", len(mesh.faces), "faces")

# Create scene with colored material
scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0])  # Black background
material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 0.0, 0.0, 1.0],  # Red color
    metallicFactor=0.0,  # Non-metallic
    roughnessFactor=0.5  # Moderately rough for better light reflection
)
pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
scene.add(pyrender_mesh)

# Set up camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.eye(4)
camera_pose[:3, 3] = [0, 0, 30]  # Move camera closer to z=30
nc = pyrender.Node(camera=camera, matrix=camera_pose)
scene.add_node(nc)

# Set up light
light_pose = np.eye(4)
light_pose[:3, 3] = [0, 0, 30]  # Light at camera position
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1000.0)  # Much brighter light
nl = pyrender.Node(light=light, matrix=light_pose)
scene.add_node(nl)

# Render video
r = pyrender.OffscreenRenderer(640, 480)
imgs = []
for i, camera_pose in enumerate(camera_poses_4x4):
    scene.set_pose(nc, pose=camera_pose)
    color, _ = r.render(scene)
    imgs.append(color)
    print(f"Rendered frame {i + 1}/{len(camera_poses_4x4)}")

# Preview first frame
plt.imshow(imgs[0])
plt.title("First Frame Preview")
plt.show()

# Save video
output_file = 'output_video_colored.mp4'
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