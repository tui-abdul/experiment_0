import numpy as np
import open3d as o3d
import cv2

# -----------------------------
# Camera intrinsics
# -----------------------------
fx = 1639.26747
fy = 1639.23635
cx = 1295.36149
cy = 1042.72444

img_width = 2600
img_height = 2128

K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))  # assuming undistorted
#dist_coeffs = np.array(
#    [-0.219495, 0.060523, 0.000420, 0.000547, 0.000000],
#    dtype=np.float64
#).reshape(1, 5)
# -----------------------------
# Camera extrinsics (camera -> world)
# Column-major → already converted properly
# -----------------------------
T_wc = np.array([
    [ 0.55655557, -0.83053924, -0.02122401,  0.1155929 ],
    [-0.07919148, -0.02760281, -0.99647719,  0.06345451],
    [ 0.82702757,  0.55627569, -0.08113413, -0.02520101],
    [ 0.0,         0.0,         0.0,          1.0       ]
], dtype=np.float64)

# World -> camera
#T_wc = np.linalg.inv(T_cw)

R_wc = T_wc[:3, :3]
t_wc = T_wc[:3, 3]

# Convert rotation matrix to Rodrigues
rvec, _ = cv2.Rodrigues(R_wc)
tvec = t_wc.reshape(3, 1)

# -----------------------------
# Load point cloud
# -----------------------------
pcd = o3d.io.read_point_cloud("134.pcd")
points_world = np.asarray(pcd.points, dtype=np.float64)

# -----------------------------
# Project points using OpenCV
# -----------------------------
image_points, _ = cv2.projectPoints(
    points_world,
    rvec,
    tvec,
    K,
    dist_coeffs
)

image_points = image_points.reshape(-1, 2)

u = image_points[:, 0]
v = image_points[:, 1]

# -----------------------------
# Filter points
# -----------------------------
# Compute camera-space Z for front-facing test
points_cam = (R_wc @ points_world.T).T + t_wc
Z = points_cam[:, 2]

valid = (
    (Z > 0) &
    (u >= 0) & (u < img_width) &
    (v >= 0) & (v < img_height)
)

sliced_points = points_world[valid]

# -----------------------------
# Save sliced point cloud
# -----------------------------
sliced_pcd = o3d.geometry.PointCloud()
sliced_pcd.points = o3d.utility.Vector3dVector(sliced_points)

o3d.io.write_point_cloud("pointcloud_134_slice.pcd", sliced_pcd)

print(f"Original points: {len(points_world)}")
print(f"Visible points: {len(sliced_points)}")


# -----------------------------
# Visualize sliced point cloud
# -----------------------------
print("Visualizing sliced point cloud...")
pcd.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries(
    [pcd,sliced_pcd],
    window_name="Original (gray) vs Visible (red)",
    width=1280,
    height=800
)