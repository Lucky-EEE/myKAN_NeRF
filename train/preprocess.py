import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor


# Load images in parallel
def load_images(frame_folder):
    image_paths = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')]
    image_paths = sorted(image_paths)

    def load_image(path):
        try:
            return cv2.imread(path)
        except Exception as e:
            print(f"Warning: Could not load image {path}, error: {e}")
            return None

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_paths))

    # Filter out failed loads
    images = [img for img in images if img is not None]
    return images


# Extract keypoints and descriptors
def extract_features(images):
    sift = cv2.SIFT_create()
    keypoints_list, descriptors_list = [], []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list


# Feature matching with FLANN
def match_features(descriptors_list):
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_list = []

    for i in range(len(descriptors_list) - 1):
        matches = flann.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2)
        # Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        matches_list.append(good_matches)

    return matches_list


# Estimate camera pose
def estimate_camera_pose(K, matches, keypoints1, keypoints2):
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    return R, t


# Triangulate 3D points
def triangulate_points(K, R1, t1, R2, t2, keypoints1, keypoints2, matches):
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches]).T
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches]).T

    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T


# Save results
def save_model(cameras, points_3d, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "cameras.npy"), cameras)
    np.savetxt(os.path.join(output_folder, "points_3d.txt"), points_3d)


# Main execution
def main():
    frame_folder = r"../frame"  # Replace with your actual path
    output_folder = r"../preprocess_result"  # Replace with your actual path
    K = np.array([[5851.35, 0, 960], [0, 5851.35, 540], [0, 0, 1]])  # Replace with actual intrinsics

    images = load_images(frame_folder)
    if not images:
        print("Error: No images loaded")
        return

    keypoints_list, descriptors_list = extract_features(images)
    matches_list = match_features(descriptors_list)

    cameras = []
    points_3d_all = []

    # Initialize pose
    R_prev, t_prev = np.eye(3), np.zeros((3, 1))

    for i in range(len(matches_list)):
        matches = matches_list[i]
        if len(matches) < 8:  # Minimum matches required
            print(f"Warning: Insufficient matches between images {i} and {i + 1}")
            continue

        keypoints1, keypoints2 = keypoints_list[i], keypoints_list[i + 1]
        R, t = estimate_camera_pose(K, matches, keypoints1, keypoints2)

        # Sparse reconstruction
        points_3d = triangulate_points(K, R_prev, t_prev, R, t, keypoints1, keypoints2, matches)
        points_3d_all.extend(points_3d)

        # Update camera pose
        cameras.append(np.hstack((R_prev, t_prev)))
        R_prev, t_prev = R, t

    if cameras and points_3d_all:
        save_model(np.array(cameras), np.array(points_3d_all), output_folder)
        print(f"Results saved to {output_folder}")
    else:
        print("Error: No camera poses or 3D points generated")


if __name__ == "__main__":
    main()