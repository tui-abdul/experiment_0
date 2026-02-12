import json
from typing import Dict, List


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

if __name__ == "__main__":
    json_path = "250_xtr.json"
    extracted_boxes = extract_3d_boxes(json_path)

    for box in extracted_boxes:
        print(f"Track ID: {box['trackId']}")
        print(f"Class Name: {box['className']}")
        print(f"Truncated: {box['truncated']}")
        print(f"Occluded: {box['occluded']}")
        print(f"Size 3D: {box['size3D']}")
        print(f"Center 3D: {box['center3D']}")
        print(f"Rotation 3D: {box['rotation3D']}")
        print("-" * 30)