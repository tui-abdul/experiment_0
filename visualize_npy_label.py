import numpy as np
import open3d as o3d
import sys
import os
import math

# Define the valid point cloud range: [x_min, y_min, z_min, x_max, y_max, z_max]
POINT_CLOUD_RANGE = [-40, -40, -5, 60, 40, 1]


def visualize_pointcloud_with_labels(npy_path, txt_path):
    """
    Visualize a LiDAR .npy point cloud with 3D bounding boxes from .txt labels.
    Expected .txt format:
        center_x center_y center_z width height length yaw type
    """

    if not os.path.exists(npy_path):
        print(f"❌ Point cloud file not found: {npy_path}")
        return
    if not os.path.exists(txt_path):
        print(f"⚠️ Label file not found: {txt_path}")
        return

    # Load point cloud
    points = np.load(npy_path)
    if points.shape[1] == 3:
        points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
    elif points.shape[1] != 4:
        raise ValueError(f"Invalid .npy shape {points.shape}, expected Nx3 or Nx4")

    # Apply range filter
    x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    points = points[mask]
    print(f"🧭 Filtered {np.sum(~mask)} points outside range {POINT_CLOUD_RANGE}")

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Colorize by intensity
    intensities = points[:, 3]
    intensities = (intensities - intensities.min()) / (intensities.ptp() + 1e-8)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(intensities[:, None], 3, axis=1))

    geometries = [pcd]

    # Add coordinate axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geometries.append(axes)

    # Load bounding boxes
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    colors = {
        "car": [1, 0, 0],
        "cyclist": [0, 1, 0],
        "pedestrian": [0, 0, 1],
        "truck": [1, 1, 0],
        "van": [1, 0, 1],
    }

    for line in lines:
        parts = line.split()
        if len(parts) < 8:
            continue

        cx, cy, cz, w, h, l, yaw, obj_type = parts
        cx, cy, cz, w, h, l, yaw = map(float, [cx, cy, cz, w, h, l, yaw])

        # Create bounding box mesh
        bbox = create_3d_bbox(center=(cx, cy, cz), size=(w, h, l), yaw=yaw)
        color = colors.get(obj_type.lower(), [0, 1, 1])  # default cyan
        bbox.paint_uniform_color(color)
        geometries.append(bbox)

    # Visualize all
    print(f"✅ Loaded {len(points)} filtered points and {len(lines)} labels")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud with Range and Labels",
        width=1280,
        height=960,
    )


def create_3d_bbox(center, size, yaw):
    """
    Create a 3D bounding box mesh centered at `center` with given size and yaw.
    """
    w, h, l = size
    cx, cy, cz = center

    # Define 8 corners of the bounding box in local coordinates
    x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    y_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners = np.vstack([x_corners, y_corners, z_corners])

    # Rotation matrix (around Z-axis)
    R = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])
    rotated = R @ corners

    # Translate
    rotated[0, :] += cx
    rotated[1, :] += cy
    rotated[2, :] += cz

    # Create Open3D line set for the box edges
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # top
        [4, 5], [5, 6], [6, 7], [7, 4],  # bottom
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(rotated.T)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_pointcloud_with_labels.py <path_to_npy> <path_to_txt>")
        sys.exit(1)

    npy_path = sys.argv[1]
    txt_path = sys.argv[2]
    visualize_pointcloud_with_labels(npy_path, txt_path)
