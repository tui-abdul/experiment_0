import numpy as np
import open3d as o3d
import cv2

# -----------------------------
# Camera intrinsics
# -----------------------------
fx = 1635.74345
fy = 1635.63061
cx = 1265.0683
cy = 1056.23575

img_width = 2600
img_height = 2128

K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))  # assuming undistorted

# -----------------------------
# Camera extrinsics (camera -> world)
# -----------------------------
T_wc = np.array([
    [ 0.49057328, -0.87136263, 0.00806423,  0.1416327 ],
    [-0.02230826, -0.02180968, -0.99951322,  0.03757607],
    [ 0.87111434,  0.49015458, -0.03013782, -0.02256819],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float64)

rotation_matrix_pole_b = np.array([
    [9.22743745e-01, -2.37220580e-04, -3.85413966e-01],
    [-2.37220580e-04, 9.99999272e-01, -1.18343976e-03],
    [3.85413966e-01, 1.18343976e-03, 9.22743017e-01]
])

R_wc = T_wc[:3, :3]
t_wc = T_wc[:3, 3]

rvec, _ = cv2.Rodrigues(R_wc)
tvec = t_wc.reshape(3, 1)

# -----------------------------
# Load point cloud from .npy
# -----------------------------
points_data = np.load("803.npy")

if points_data.shape[1] >= 3:
    points_world = points_data[:, :3].astype(np.float64)
else:
    raise ValueError("NPY file must contain at least 3 columns (XYZ).")

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_world)

# Optional: load colors if available (Nx6 format)
if points_data.shape[1] >= 6:
    colors = points_data[:, 3:6]
    
    # Normalize if needed (if values are 0–255)
    if colors.max() > 1.0:
        colors = colors / 255.0
    
    pcd.colors = o3d.utility.Vector3dVector(colors)

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
points_cam = (R_wc @ points_world.T).T + t_wc
Z = points_cam[:, 2]

valid = (
    (Z > 0) &
    (u >= 0) & (u < img_width) &
    (v >= 0) & (v < img_height)
)

sliced_points = points_world[valid]

inv_rot = rotation_matrix_pole_b.T
sliced_points = sliced_points @ inv_rot.T

POINT_CLOUD_RANGE = [-40, -40, -5, 40, 40, 1]

# 1. Create a mask for each dimension
mask = (
    (points_world[:, 0] >= POINT_CLOUD_RANGE[0]) & (points_world[:, 0] <= POINT_CLOUD_RANGE[3]) & # X
    (points_world[:, 1] >= POINT_CLOUD_RANGE[1]) & (points_world[:, 1] <= POINT_CLOUD_RANGE[4]) & # Y
    (points_world[:, 2] >= POINT_CLOUD_RANGE[2]) & (points_world[:, 2] <= POINT_CLOUD_RANGE[5])   # Z
)

# 2. Filter the points using the mask
points_world = points_world[mask]

pcd_points_world = o3d.geometry.PointCloud()
pcd_points_world.points = o3d.utility.Vector3dVector(points_world)

# -----------------------------
# Save sliced point cloud
# -----------------------------
sliced_pcd = o3d.geometry.PointCloud()
sliced_pcd.points = o3d.utility.Vector3dVector(sliced_points)

o3d.io.write_point_cloud("pointcloud_007_slice.pcd", sliced_pcd)

print(f"Original points: {len(points_world)}")
print(f"Visible points: {len(sliced_points)}")

# -----------------------------
# Visualize
# -----------------------------
sliced_pcd.paint_uniform_color([1, 0, 0])  # red

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=2.0,
    origin=[0, 0, 0]
)

o3d.visualization.draw_geometries(
    [pcd_points_world, axes],
    window_name="Visible Points",
    width=1280,
    height=800
)