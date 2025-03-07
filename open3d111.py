import open3d as o3d
import numpy as np

def remove_outliers_radius(file_path, nb_points=16, radius=0.15):
    """
    使用半径离群点移除方法移除异常点。

    Args:
        file_path (str): 包含 3D 点坐标的文本文件路径。
        nb_points (int): 在给定半径内，点被认为是内点的最小邻居数。
        radius (float): 用于搜索邻居的半径。
    """
    inlier_cloud = None  # 初始化 inlier_cloud 为 None
    try:
        points_3d = np.loadtxt(file_path)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_3d)

        # 半径离群点移除
        cl, ind = point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)

        # 提取保留的点云
        inlier_cloud = point_cloud.select_by_index(ind)

        # 可视化结果
        o3d.visualization.draw_geometries([inlier_cloud])

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        return
    except ValueError:
        print(f"错误：文件 {file_path} 格式不正确。请确保每行包含三个数字（x, y, z），以空格分隔。")
        return

    if inlier_cloud is not None:  # 检查 inlier_cloud 是否被成功赋值
        np.savetxt("inliers_points_3d.txt", np.asarray(inlier_cloud.points))
    else:
        print("处理点云时出错，未保存文件。")

# 示例用法
file_path = "points_3d.txt"
remove_outliers_radius(file_path)