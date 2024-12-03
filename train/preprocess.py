import cv2
import numpy as np
import os


# 1. 加载抽帧图像
def load_images(frame_folder):
    image_paths = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')]
    images = [cv2.imread(img_path) for img_path in sorted(image_paths)]
    return images


# 2. 提取特征点和描述子
def extract_features(images):
    sift = cv2.SIFT_create()
    keypoints_list, descriptors_list = [], []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list


# 3. 特征匹配
def match_features(descriptors_list):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_list = []

    for i in range(len(descriptors_list) - 1):
        matches = bf.match(descriptors_list[i], descriptors_list[i + 1])
        matches = sorted(matches, key=lambda x: x.distance)
        matches_list.append(matches)

    return matches_list


# 4. 位姿估计
def estimate_camera_pose(K, matches, keypoints1, keypoints2):
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(points1, points2, K, cv2.RANSAC, 0.999, 1.0)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    return R, t


# 5. 稀疏重建
def triangulate_points(K, R1, t1, R2, t2, keypoints1, keypoints2, matches):
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches]).T
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches]).T

    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T


# 6. 保存结果
def save_model(cameras, points_3d, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # 保存 cameras 为 .npy 格式
    np.save(os.path.join(output_folder, "cameras.npy"), cameras)

    # 保存 3D 点云为 .txt 格式
    np.savetxt(os.path.join(output_folder, "points_3d.txt"), points_3d)


# 主程序
def main():
    frame_folder = r"..\frame"  # 替换为你存放抽帧图像的路径
    output_folder = r"..\preprocess_result"  # 替换为你希望保存结果的路径
    K = np.array([[5851.35, 0, 960], [0, 5851.35, 540], [0, 0, 1]])  # 假设的内参矩阵，根据实际相机调整

    images = load_images(frame_folder)
    keypoints_list, descriptors_list = extract_features(images)
    matches_list = match_features(descriptors_list)

    cameras = []
    points_3d_all = []

    # 初始化位姿
    R_prev, t_prev = np.eye(3), np.zeros((3, 1))

    for i in range(len(matches_list)):
        matches = matches_list[i]
        keypoints1, keypoints2 = keypoints_list[i], keypoints_list[i + 1]
        R, t = estimate_camera_pose(K, matches, keypoints1, keypoints2)

        # 稀疏重建
        points_3d = triangulate_points(K, R_prev, t_prev, R, t, keypoints1, keypoints2, matches)
        points_3d_all.extend(points_3d)

        # 更新相机位姿
        cameras.append(np.hstack((R_prev, t_prev)))
        R_prev, t_prev = R, t

    # 保存结果
    save_model(np.array(cameras), np.array(points_3d_all), output_folder)
    print(f"结果已保存到 {output_folder}")

#wzy
if __name__ == "__main__":
    main()
