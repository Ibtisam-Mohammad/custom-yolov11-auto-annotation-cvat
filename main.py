import io
import base64
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from skimage.measure import approximate_polygon, find_contours
# from shared import to_cvat_mask  # if needed in your environment

MASK_THRESHOLD = 0.5

def init_context(context):
    context.logger.info("Init context... 0%")
    
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
    """
    Convert a binary mask into the CVAT mask format:
      [mask data..., xtl, ytl, xbr, ybr]
    """
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened

def iou(boxA, boxB):
    """
    Compute IoU for axis-aligned bounding boxes:
      box = (x1, y1, x2, y2)
    """
    (xA1, yA1, xA2, yA2) = boxA
    (xB1, yB1, xB2, yB2) = boxB

    # Intersection
    interX1 = max(xA1, xB1)
    interY1 = max(yA1, yB1)
    interX2 = min(xA2, xB2)
    interY2 = min(yA2, yB2)
    interW = max(0, interX2 - interX1 + 1)
    interH = max(0, interY2 - interY1 + 1)
    interArea = interW * interH

    # Areas
    areaA = (xA2 - xA1 + 1) * (yA2 - yA1 + 1)
    areaB = (xB2 - xB1 + 1) * (yB2 - yB1 + 1)

    # IoU
    return interArea / float(areaA + areaB - interArea + 1e-8)

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
    # Move OBB to CPU if needed.
    result.obb.cpu()

    polygons = result.obb.xyxy.cpu().numpy().tolist()
    confs = result.obb.conf.cpu().numpy().tolist()
    clss = result.obb.cls.cpu().numpy().tolist()
    class_names = result.names

    # Prepare the SAM predictor by setting the image.
    sam_predictor = context.user_data.sam_predictor
    sam_predictor.set_image(image_rgb)

    height, width = image.shape[:2]
    all_detections = []

    # Process each YOLO detection and store them in a list first.
    for polygon, conf, cls in zip(polygons, confs, clss):
        if conf < conf_threshold:
            continue

        label = class_names[int(cls)]

        # --- Convert YOLO OBB polygon to axis-aligned bounding box ---
        if len(polygon) > 4:
            # e.g., polygon is [x1,y1,x2,y2, x3,y3, ...].
            xs = polygon[0::2]
            ys = polygon[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            x1, y1, x2, y2 = polygon

        # Ensure the bounding box is within image dimensions
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # --- Inflate bounding box by 10% in each dimension ---
        w_box = x2 - x1
        h_box = y2 - y1
        # expand 5% on each side => total 10% bigger
        x1_infl = x1 - 0.05 * w_box
        x2_infl = x2 + 0.05 * w_box
        y1_infl = y1 - 0.05 * h_box
        y2_infl = y2 + 0.05 * h_box

        # clamp again
        x1_infl = max(0, min(x1_infl, width - 1))
        x2_infl = max(0, min(x2_infl, width - 1))
        y1_infl = max(0, min(y1_infl, height - 1))
        y2_infl = max(0, min(y2_infl, height - 1))

        all_detections.append({
            "conf": float(conf),
            "label": label,
            "box": (int(x1_infl), int(y1_infl), int(x2_infl), int(y2_infl))
        })

    # ------------------------------------------------------------------
    # Filter out double detections via a bounding box IoU check (NMS-ish).
    # We keep the highest-conf box in case of overlap > iou_threshold.
    # ------------------------------------------------------------------
    iou_threshold = 0.5
    # Sort all_detections by confidence descending
    all_detections.sort(key=lambda d: d["conf"], reverse=True)

    final_detections = []
    for det in all_detections:
        boxA = det["box"]
        # check IOU with final_detections
        should_keep = True
        for kept in final_detections:
            boxB = kept["box"]
            if iou(boxA, boxB) > iou_threshold:
                # If we overlap more than iou_threshold,
                # skip the new one (since we sorted by conf desc).
                should_keep = False
                break
        if should_keep:
            final_detections.append(det)

    # ------------------------------------------------------------------
    # Now run SAM for each final detection box
    # ------------------------------------------------------------------
    detections = []
    for det in final_detections:
        (x1_infl, y1_infl, x2_infl, y2_infl) = det["box"]
        conf = det["conf"]
        label = det["label"]

        # SAM requires box: [x1, y1, x2, y2] shape (1,4)
        box_np = np.array([[x1_infl, y1_infl, x2_infl, y2_infl]])
        masks, scores, _ = sam_predictor.predict(box=box_np, multimask_output=True)
        best_index = np.argmax(scores)
        best_mask = masks[best_index]

        # Convert mask to uint8 [0,255]
        mask_uint8 = (best_mask.astype(np.uint8)) * 255

        # find_contours => [ [ (row,col), ... ], [ ... ] ]
        contours = find_contours(best_mask, MASK_THRESHOLD)
        if not contours:
            continue
        contour = max(contours, key=lambda c: len(c))
        # swap row <-> col => (col, row)
        contour = np.fliplr(contour)
        approx = approximate_polygon(contour, tolerance=2.5)
        if len(approx) < 3:
            continue

        cvat_mask = to_cvat_mask((x1_infl, y1_infl, x2_infl, y2_infl), mask_uint8)
        detections.append({
            "confidence": str(conf),
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
