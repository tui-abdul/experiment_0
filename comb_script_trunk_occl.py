import os
import json
from pathlib import Path
import shutil
import json
from typing import Dict, List

import numpy as np
import cv2
import open3d as o3d

from conversion_script import transform_bbox

# =========================
# CONFIGURATION
# =========================

SOURCE_ROOT = Path("/mnt/sda/Abdul_Haq/intersection_dataset/xtreme1_complete_dataset/correction/xtreme1_validation")
DEST_ROOT = Path("/mnt/sda/Abdul_Haq/experiment_0/data_extraction_attributes")

# Optional: store lidar paths used during processing
LIDAR_PATH_LOG = DEST_ROOT / "lidar_paths.txt"


TYPE_MAP = {
    "Car": "car",
    "Bicycle": "cyclist",
    "Pedestrian": "pedestrian"
}

DIFFICULTY = "Easy"  # Placeholder, can be determined based on truncated/occluded values if needed

# Rotation matrices
rotation_matrix_pole_a = np.array([
    [0.95148668, 0.00724281, -0.30760468],
    [0.00724281, 0.99891868, 0.04592391],
    [0.30760468, -0.04592391, 0.95040536]
])

rotation_matrix_pole_b = np.array([
    [9.22743745e-01, -2.37220580e-04, -3.85413966e-01],
    [-2.37220580e-04, 9.99999272e-01, -1.18343976e-03],
    [3.85413966e-01, 1.18343976e-03, 9.22743017e-01]
])


# =========================
# UTILITY FUNCTIONS
# =========================
def pcd_to_npy_and_rotation(pcd_points, output_path, rotation_matrix):
    """
    Convert Open3D PointCloud to NPY after applying inverse rotation.

    Args:
        pcd_points (o3d.geometry.PointCloud): Loaded point cloud
        output_path (str): Output path ending with .pcd (will be converted to .npy)
        rotation_matrix (np.ndarray): 3x3 rotation matrix
    """
    # Replace .pcd with .npy
    output_path = os.path.splitext(output_path)[0] + ".npy"

    points = np.asarray(pcd_points.points, dtype=np.float32)
    if points.size == 0:
        return False

    inv_rot = rotation_matrix.T
    points[:, :3] = points[:, :3] @ inv_rot.T

    #np.save(output_path, points[:, :3])

        # 🔹 Update Open3D point cloud
    pcd_rotated = o3d.geometry.PointCloud()
    pcd_rotated.points = o3d.utility.Vector3dVector(points[:, :3])

    # Optional: keep colors if present
    if pcd_points.has_colors():
        pcd_rotated.colors = pcd_points.colors

    # 🔹 Visualize
    #if visualize:
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=2, origin=[0, 0, 0]
    )   
    """"
    o3d.visualization.draw_geometries(
            [pcd_rotated, axis],
            window_name="Rotated Point Cloud",
            width=1024,
            height=768
        )
    """
    return points[:, :3], output_path


# =========================
# UTILITY FUNCTIONS
# =========================

def is_center_in_pointcloud_range(npy_path, center3D, margin=0.0):
    """
    center3D format:
    {
        'x': float,
        'y': float,
        'z': float
    }
    """
    if not os.path.exists(npy_path):
        return False

    points = np.load(npy_path)
    if points.size == 0:
        return False

    xyz = points[:, :3]

    center = np.array(
        [center3D["x"], center3D["y"], center3D["z"]],
        dtype=np.float32
    )

    min_xyz = xyz.min(axis=0) - margin
    max_xyz = xyz.max(axis=0) + margin

    return bool(np.all(center >= min_xyz) and np.all(center <= max_xyz))

def get_difficulty(truncated, occluded):
    try:
        truncated = float(truncated)
    except (TypeError, ValueError):
        return None
    #print("truncated",truncated)
    #print("occluded",occluded)
    if occluded == 0.0 and truncated <= 0.15:
        return "Easy"
    elif occluded == 1.0 and 0.15 < truncated <= 0.30:
        return "Medium"
    elif occluded == 2.0 and truncated > 0.30:
        return "Hard"
    return None


def xtreme1_to_kitti_all_angles(extracted_boxes, rotation_matrix,output_path):
    lines = []
    output_path = os.path.splitext(output_path)[0] + ".npy"
    for box in extracted_boxes:
        obj_type = TYPE_MAP.get(box['className'], None)
        if obj_type is None:
            continue 
        if box['center3D']['x'] == 0 and box['center3D']['y'] == 0 and box['center3D']['z'] == 0:    
            continue
        #if get_difficulty(box['truncated'], box['occluded']) is not DIFFICULTY:
        #    #print(f"Skipping track {box['trackId']} due to difficulty mismatch.",get_difficulty(box['truncated'], box['occluded']) )
        #    continue
        pos = box['center3D']
        rot = box['rotation3D']
        scale = box['size3D']
        yaw, center = transform_bbox(rotation_matrix, rot, pos)
        line = f"{box['truncated']} {box['occluded']}  {center[0]} {center[1]} {center[2]} {scale['x']} {scale['y']} {scale['z']} {yaw} {obj_type}"
        lines.append(line)
    return lines



# =========================
# JSON EXTRACTION FUNCTIONS INTRINSICS & EXTRINSICS
# =========================

def extract_K_Twc_from_json(json_path, camera_index=0):
    """
    Read camera intrinsics and extrinsics from JSON and return K and T_wc.

    Parameters
    ----------
    json_path : str
        Path to JSON file
    camera_index : int
        Index of camera in JSON array (default: 0)

    Returns
    -------
    K : (3,3) np.ndarray
        Camera intrinsic matrix
    T_wc : (4,4) np.ndarray
        Camera-to-world transform
    img_width : int
        Image width
    img_height : int
        Image height
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    cam = data[camera_index]

    # -----------------------------
    # Intrinsics
    # -----------------------------
    fx = cam["camera_internal"]["fx"]
    fy = cam["camera_internal"]["fy"]
    cx = cam["camera_internal"]["cx"]
    cy = cam["camera_internal"]["cy"]

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    # -----------------------------
    # Extrinsics (camera -> world)
    # -----------------------------
    ext = np.array(cam["camera_external"], dtype=np.float64)

    if cam.get("rowMajor", True):
        T_wc = ext.reshape(4, 4)
    else:
        # column-major → transpose
        T_wc = ext.reshape(4, 4).T

    img_width = cam["width"]
    img_height = cam["height"]

    return K, T_wc, img_width, img_height

# =========================
# POINT CLOUD SLICING FUNCTION
# =========================

def slice_pointcloud_by_camera(
    pcd_path,
    output_path,
    K,
    T_wc,
    img_width,
    img_height,
    dist_coeffs=None
):
    """
    Projects a world-frame point cloud into a camera and keeps only
    points that are visible in the image.

    Parameters
    ----------
    pcd_path : str
        Path to input .pcd file
    output_path : str
        Path to save sliced .pcd file
    K : (3,3) np.ndarray
        Camera intrinsic matrix
    T_wc : (4,4) np.ndarray
        Camera-to-world transform
    img_width : int
        Image width in pixels
    img_height : int
        Image height in pixels
    dist_coeffs : np.ndarray or None
        Distortion coefficients (default: zero distortion)
    """

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # -----------------------------
    # Extrinsics
    # -----------------------------
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3].reshape(3, 1)

    rvec, _ = cv2.Rodrigues(R_wc)

    # -----------------------------
    # Load point cloud
    # -----------------------------
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points_world = np.asarray(pcd.points, dtype=np.float64)
    #print(f"Loaded point cloud with {len(points_world)} points from {pcd_path}")
    #print("Camera intrinsics K:\n", rvec, t_wc,K,dist_coeffs)
    # -----------------------------
    # Project points
    # -----------------------------
    image_points, _ = cv2.projectPoints(
        points_world,
        rvec,
        t_wc,
        K,
        dist_coeffs
    )
    image_points = image_points.reshape(-1, 2)

    u = image_points[:, 0]
    v = image_points[:, 1]

    # -----------------------------
    # Camera-space Z (front-facing)
    # -----------------------------
    points_cam = (R_wc @ points_world.T).T + t_wc.ravel()
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
    #o3d.io.write_point_cloud(str(output_path), sliced_pcd)

    #print(f"Original points: {len(points_world)}")
    #print(f"Visible points: {len(sliced_points)}")

    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    """
    o3d.visualization.draw_geometries(
    [pcd,sliced_pcd],
    window_name="Original (gray) vs Visible (red)",
    width=1280,
    height=800
    )
    """
    return sliced_pcd

# =========================
# JSON EXTRACTION FUNCTIONS
# ========================= 

def extract_truncated_occluded(class_values: List[Dict]) -> Dict[str, float]:
    truncated = None
    occluded = None

    for item in class_values:
        if item.get("alias") == "truncated":
            truncated = float(item.get("value", 0))
        elif item.get("alias") == "occluded":
            occluded = float(item.get("value", 0))

    return truncated, occluded


def has_valid_2d_points(contour: Dict) -> bool:
    points = contour.get("points", [])
    if not points or len(points) < 4:
        return False

    # Check at least one point is non-zero
    for p in points:
        if p.get("x", 0) != 0 or p.get("y", 0) != 0:
            return True

    return False


def extract_3d_boxes(json_path: str) -> List[Dict]:
    with open(json_path, "r") as f:
        data = json.load(f)

    objects = data.get("objects", [])

    objects_2d = {}
    objects_3d = {}

    # Separate 2D and 3D objects by trackId
    for obj in objects:
        track_id = obj.get("trackId")
        obj_type = obj.get("type")

        if obj_type == "2D_RECT":
            if has_valid_2d_points(obj.get("contour", {})):
                objects_2d[track_id] = obj

        elif obj_type == "3D_BOX":
            objects_3d[track_id] = obj

    extracted_data = []

    # Match 2D and 3D using trackId
    for track_id in objects_2d.keys() & objects_3d.keys():
        obj_2d = objects_2d[track_id]
        obj_3d = objects_3d[track_id]

        class_name = obj_3d.get("className")

        truncated, occluded = extract_truncated_occluded(
            obj_3d.get("classValues", [])
        )

        contour_3d = obj_3d.get("contour", {})

        size3d = contour_3d.get("size3D", {})
        center3d = contour_3d.get("center3D", {})
        rotation3d = contour_3d.get("rotation3D", {})

        extracted_data.append({
            "trackId": track_id,
            "className": class_name,
            "truncated": truncated,
            "occluded": occluded,

            "size3D": {
                "x": size3d.get("x"),
                "y": size3d.get("y"),
                "z": size3d.get("z")
            },
            "center3D": {
                "x": center3d.get("x"),
                "y": center3d.get("y"),
                "z": center3d.get("z")
            },
            "rotation3D": {
                "x": rotation3d.get("x"),
                "y": rotation3d.get("y"),
                "z": rotation3d.get("z")
            }
        })

    return extracted_data


# =========================
# UTILITIES
# =========================

def ensure_dir(path: Path):
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def convert_json_to_txt(json_path: Path, txt_path: Path, lidar_pcd_path: Path, dest_lidar_dir: Path, int_ext_path: Path):
    """
    Placeholder for JSON -> TXT conversion logic.
    You will specify extraction rules in the next step.
    """
    #print(json_path)
    #print(int_ext_path)
    #print("lidar pc path",lidar_pcd_path)
    with open(json_path, "r") as f:
        data = json.load(f)
    extracted_boxes = extract_3d_boxes(json_path)  # Example of how to call the extraction function
    """
    for box in extracted_boxes:
        print(f"Track ID: {box['trackId']}")
        print(f"Class Name: {box['className']}")
        print(f"Truncated: {box['truncated']}")
        print(f"Occluded: {box['occluded']}")
        print(f"Size 3D: {box['size3D']}")
        print(f"Center 3D: {box['center3D']}")
        print(f"Rotation 3D: {box['rotation3D']}")
        print("-" * 30) 
    """
    K, T_wc, img_width, img_height = extract_K_Twc_from_json(int_ext_path)
    """
    print("Extracted intrinsics and extrinsics:")
    print("K:\n", K)
    print("T_wc:\n", T_wc)
    print(f"Image size: {img_width}x{img_height}")
    """
    #print("dest_lidar_path",dest_lidar_dir / f"{lidar_pcd_path.stem}.pcd")
    #print("lidar_pcd_path",lidar_pcd_path)
    sliced_points = slice_pointcloud_by_camera(
    pcd_path=lidar_pcd_path,
    output_path=dest_lidar_dir / f"{lidar_pcd_path.stem}.pcd",
    K=K,
    T_wc=T_wc,
    img_width=img_width,
    img_height=img_height
    )
    output_path_lidar = dest_lidar_dir / f"{lidar_pcd_path.stem}.pcd"
    lidar_folder = output_path_lidar.parent.name

    
    if lidar_folder == "lidar_point_cloud_0":
        rotation_matrix = rotation_matrix_pole_a
    elif lidar_folder == "lidar_point_cloud_1":
        rotation_matrix = rotation_matrix_pole_b
    
    points,output_path_fin = pcd_to_npy_and_rotation(sliced_points, output_path_lidar, rotation_matrix)

    kitti_lines = xtreme1_to_kitti_all_angles(extracted_boxes, rotation_matrix,output_path_lidar)
    #print("Generated KITTI-format lines:", kitti_lines)
    if not kitti_lines:
        print("txt path",txt_path)
        return
    np.save(output_path_fin, points[:, :3])
    with open(txt_path, "w") as f:
        for line in kitti_lines:
            f.write(line + "\n")
    """
    # ---- PLACEHOLDER CONTENT ----
    with open(txt_path, "w") as f:
        f.write(f"# Source JSON: {json_path.name}\n")
        f.write(f"# Corresponding LiDAR: {lidar_pcd_path}\n")
        f.write("# JSON-to-TXT conversion logic will go here\n")
    """
    #exit()

# =========================
# MAIN PROCESSING LOGIC
# =========================

def process_dataset():
    ensure_dir(DEST_ROOT)

    with open(LIDAR_PATH_LOG, "w") as lidar_log:

        # weather folders: cloudy, sunny, light_rain, etc.
        for weather_dir in SOURCE_ROOT.iterdir():
            if not weather_dir.is_dir():
                continue
              
            # sequence_0, sequence_1, ...
            for sequence_dir in weather_dir.iterdir():
                if not sequence_dir.is_dir():
                    continue
                    
                # scene_x_weather_sequence_x
                for scene_dir in sequence_dir.iterdir():
                    if not scene_dir.is_dir():
                        continue
                    # extract second index from scene name
                    scene_idx = int(scene_dir.name.split('_')[1])
                    camera_dir = scene_dir / f"camera_image_{scene_idx}"
                    # map scene index to camera/lidar index
                    if scene_idx in (0, 1):
                        lidar_idx  = 0
                    elif scene_idx in (2, 3):
                        lidar_idx  = 1
                    else:
                        # optional safety check
                        raise ValueError(f"Unexpected scene index: {scene_idx}")

                    
                    lidar_dir = scene_dir / f"lidar_point_cloud_{lidar_idx}"
                    results_dir = scene_dir / "result"
                    camera_config_dir = scene_dir / "camera_config"

                    if not results_dir.exists():
                        continue
                    
                    # Destination paths (mirror structure)
                    relative_scene_path = scene_dir.relative_to(SOURCE_ROOT)
                    dest_scene_root = DEST_ROOT / relative_scene_path
                    dest_camera_dir = dest_scene_root / camera_dir.name
                    dest_lidar_dir = dest_scene_root / lidar_dir.name
                    dest_int_ext_dir = dest_scene_root / camera_config_dir.name
                    dest_results_dir = dest_scene_root / "result"

                    ensure_dir(dest_camera_dir)
                    ensure_dir(dest_lidar_dir)
                    ensure_dir(dest_results_dir)

                    # Copy camera & lidar folders completely (structure preserved)
                    #if camera_dir.exists():
                    #    shutil.copytree(camera_dir, dest_camera_dir, dirs_exist_ok=True)

                    #if lidar_dir.exists():
                    #    shutil.copytree(lidar_dir, dest_lidar_dir, dirs_exist_ok=True)

                    # Process JSON files
                    for json_file in sorted(results_dir.glob("*.json")):
                        base_name = json_file.stem
                        lidar_pcd = lidar_dir / f"{base_name}.pcd"
                        int_ext_path = camera_config_dir / f"{base_name}.json"

                        dest_txt = dest_results_dir / f"{base_name}.txt"

                        # Log lidar path while processing JSON
                        if lidar_pcd.exists():
                            lidar_log.write(str(lidar_pcd.resolve()) + "\n")

                        convert_json_to_txt(
                            json_path=json_file,
                            txt_path=dest_txt,
                            lidar_pcd_path=lidar_pcd,
                            dest_lidar_dir=dest_lidar_dir,
                            int_ext_path=int_ext_path 
                        )

    print("Dataset processing complete.")
    print(f"LiDAR paths logged in: {LIDAR_PATH_LOG}")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    process_dataset()
