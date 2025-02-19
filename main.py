import io
import base64
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from skimage.measure import approximate_polygon, find_contours

# Import helper to convert a binary mask into CVAT mask format.
# (Make sure that shared.to_cvat_mask is available in your environment.)
# from shared import to_cvat_mask

# A threshold used for converting a binary mask into contours.
MASK_THRESHOLD = 0.5

def init_context(context):
    context.logger.info("Init context...  0%")
    
    # --- Load the YOLO model for oriented bounding boxes ---
    yolo_model = YOLO("best.pt")
    context.user_data.yolo = yolo_model

    # --- Load the SAM model ---
    sam_checkpoint = "/opt/nuclio/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model.to(device=device)
    sam_predictor = SamPredictor(sam_model)
    context.user_data.sam_predictor = sam_predictor

    context.logger.info("Init context...100%")

def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened

def handler(context, event):
    context.logger.info("Running YOLO+SAM segmentation refinement")
    data = event.body

    # Optional confidence threshold for YOLO detections (default: 0.1)
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

    # Convert the image to RGB for SAM
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Run YOLO to get oriented bounding boxes (OBBs) ---
    yolo_results = context.user_data.yolo(image)
    result = yolo_results[0]
    # Move result to CPU if needed.
    result.obb.cpu()

    # These attributes are expected from your YOLO model:
    # - result.obb.xyxy: The predicted polygon coordinates.
    # - result.obb.conf: The detection confidence.
    # - result.obb.cls: The predicted class indices.
    # - result.names: A mapping of class indices to class names.
    polygons = result.obb.xyxy.cpu().numpy().tolist()
    confs = result.obb.conf.cpu().numpy().tolist()
    clss = result.obb.cls.cpu().numpy().tolist()
    class_names = result.names

    # Prepare the SAM predictor by setting the image.
    sam_predictor = context.user_data.sam_predictor
    sam_predictor.set_image(image_rgb)

    height, width = image.shape[:2]
    detections = []

    # Process each YOLO detection.
    for polygon, conf, cls in zip(polygons, confs, clss):
        if conf < conf_threshold:
            continue

        label = class_names[int(cls)]

        # --- Convert the (possibly rotated) polygon into an axis-aligned box ---
        if len(polygon) > 4:
            xs = polygon[0::2]
            ys = polygon[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            x1, y1, x2, y2 = polygon

        # Ensure the bounding box is within the image dimensions.
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # SAM requires the box prompt in the format [x1, y1, x2, y2] with shape (1,4)
        box = np.array([[x1, y1, x2, y2]])

        # --- Use SAM to generate segmentation masks ---
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        best_index = np.argmax(scores)
        best_mask = masks[best_index]  # binary mask (True/False)

        # Convert the mask to uint8 format (0 and 255) for further processing.
        mask_uint8 = (best_mask.astype(np.uint8)) * 255

        # --- Extract a polygon from the segmentation mask ---
        # Use find_contours (from skimage) to get mask contours.
        contours = find_contours(best_mask, MASK_THRESHOLD)
        if not contours:
            continue

        # Select the largest contour (by length)
        contour = max(contours, key=lambda c: len(c))
        # find_contours returns coordinates in (row, col) order; flip them to (x, y)
        contour = np.fliplr(contour)
        # Simplify the contour to reduce the number of points.
        approx = approximate_polygon(contour, tolerance=2.5)
        if len(approx) < 3:
            continue

        # --- Convert the binary mask to CVAT's mask format ---
        # to_cvat_mask expects a bounding box and a binary mask.
        cvat_mask = to_cvat_mask((int(x1), int(y1), int(x2), int(y2)), mask_uint8)

        detections.append({
            "confidence": str(float(conf)),
            "label": label,
            "points": approx.ravel().tolist(),
            "mask": cvat_mask,
            "type": "mask",
        })

    context.logger.info(f"Detections: {detections}")

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )