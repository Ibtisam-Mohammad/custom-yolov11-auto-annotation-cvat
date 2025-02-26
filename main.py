import io
import base64
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from skimage.measure import approximate_polygon, find_contours
# from shared import to_cvat_mask  # if needed in your environment

MASK_THRESHOLD = 0.5

def init_context(context):
    context.logger.info("Init context... 0%")
    
    # --- Load the YOLO segmentation model ---
    # Replace "best_seg.pt" with your actual YOLO segmentation weights
    yolo_model = YOLO("best_seg.pt")
    context.user_data.yolo = yolo_model

    context.logger.info("Init context... 100%")

def to_cvat_mask(mask, box):
    """
    Convert a binary mask into the CVAT mask format:
      [mask data..., xtl, ytl, xbr, ybr]
    mask: a 2D np.uint8 array, 0 or 255
    box: (xtl, ytl, xbr, ybr) bounding box for the mask
    """
    (xtl, ytl, xbr, ybr) = box
    # Slice out the portion of 'mask' corresponding to the bounding box, flatten it, then
    # append the bounding box corners.
    sub_mask = mask[ytl:ybr + 1, xtl:xbr + 1]
    flattened = sub_mask.flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened

def handler(context, event):
    context.logger.info("Running YOLO segmentation inference")
    data = event.body

    # Optional confidence threshold (default: 0.1)
    conf_threshold = float(data.get("threshold", 0.1))
    
    # Decode the image (assumed to be base64 encoded)
    image_bytes = io.BytesIO(base64.b64decode(data["image"]))
    image = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return context.Response(
            body=json.dumps({"error": "Invalid image"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    height, width = image.shape[:2]

    # --- Run YOLO segmentation ---
    yolo_results = context.user_data.yolo(image, retina_masks=True)
    result = yolo_results[0]

    # Each 'mask' in result.masks.data is a [H,W] float array with 0..1 or boolean
    # We also have result.boxes.conf for confidence, result.boxes.cls for class, etc.
    # Ensure these are on CPU
    seg_masks = result.masks.data.cpu().numpy()  # shape: (N, H, W)
    confs = result.boxes.conf.cpu().numpy().tolist()
    clss = result.boxes.cls.cpu().numpy().tolist()
    class_names = result.names

    detections = []

    # Loop over each predicted object
    for idx in range(seg_masks.shape[0]):
        conf = confs[idx]
        if conf < conf_threshold:
            continue

        cls_index = int(clss[idx])
        label = class_names[cls_index]

        # Retrieve the mask (0..1 float). Convert to 0/255 for contour finding.
        mask_float = seg_masks[idx]  # shape (H, W)
        mask_uint8 = (mask_float * 255).astype(np.uint8)

        # Find polygon from the mask using scikit-image find_contours
        # find_contours => list of arrays [ (row,col), (row,col), ... ]
        contours = find_contours(mask_float, MASK_THRESHOLD)
        if not contours:
            continue
        # Just pick the largest contour
        contour = max(contours, key=lambda c: len(c))
        # The coordinates are in (row, col), we want (col, row)
        contour = np.fliplr(contour)

        approx = approximate_polygon(contour, tolerance=2.5)
        if len(approx) < 3:
            continue

        # We also build a bounding box for the mask for CVAT
        xs = contour[:, 0]
        ys = contour[:, 1]
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        # Round and clamp bounding box to integer coords
        x_min = int(max(0, min(x_min, width - 1)))
        x_max = int(max(0, min(x_max, width - 1)))
        y_min = int(max(0, min(y_min, height - 1)))
        y_max = int(max(0, min(y_max, height - 1)))

        # Build CVAT mask from the bounding box region
        cvat_mask = to_cvat_mask(mask_uint8, (x_min, y_min, x_max, y_max))

        # Build the detection
        detections.append({
            "confidence": str(float(conf)),
            "label": label,
            "points": approx.ravel().tolist(),  # flattened polygon
            "mask": cvat_mask,
            "type": "mask"
        })

    context.logger.info(f"Detections: {detections}")

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
