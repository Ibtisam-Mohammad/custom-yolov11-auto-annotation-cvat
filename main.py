import io
import base64
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def init_context(context):
    context.logger.info("Init context... 0%")
    
    # --- Load the YOLO model ---
    yolo_model = YOLO("yolo11x.pt")
    context.user_data.yolo = yolo_model

    context.logger.info("Init context...100%")

def iou(boxA, boxB):
    """
    Compute IoU for axis-aligned bounding boxes:
      box = (x1, y1, x2, y2)
    """
    (xA1, yA1, xA2, yA2) = boxA
    (xB1, yB1, xB2, yB2) = boxB

    interX1 = max(xA1, xB1)
    interY1 = max(yA1, yB1)
    interX2 = min(xA2, xB2)
    interY2 = min(yA2, yB2)
    interW = max(0, interX2 - interX1 + 1)
    interH = max(0, interY2 - interY1 + 1)
    interArea = interW * interH

    areaA = (xA2 - xA1 + 1) * (yA2 - yA1 + 1)
    areaB = (xB2 - xB1 + 1) * (yB2 - yB1 + 1)

    return interArea / float(areaA + areaB - interArea + 1e-8)

def handler(context, event):
    context.logger.info("Running YOLO object detection")
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

    # --- Run YOLO to get detections ---
    yolo_results = context.user_data.yolo(image)
    result = yolo_results[0]

    # Use oriented bounding boxes if available; otherwise, fallback to axis-aligned boxes
    if hasattr(result, 'obb') and result.obb is not None:
        result.obb.cpu()
        polygons = result.obb.xyxy.cpu().numpy().tolist()
        confs = result.obb.conf.cpu().numpy().tolist()
        clss = result.obb.cls.cpu().numpy().tolist()
    elif hasattr(result, 'boxes') and result.boxes is not None:
        result.boxes.cpu()
        polygons = result.boxes.xyxy.cpu().numpy().tolist()
        confs = result.boxes.conf.cpu().numpy().tolist()
        clss = result.boxes.cls.cpu().numpy().tolist()
    else:
        context.logger.error("No detection attributes found in YOLO result.")
        return context.Response(
            body=json.dumps({"error": "No detections found"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    class_names = result.names
    height, width = image.shape[:2]
    all_detections = []

    for polygon, conf, cls in zip(polygons, confs, clss):
        if conf < conf_threshold:
            continue

        label = class_names[int(cls)]
        # Convert polygon or oriented box to an axis-aligned box
        if len(polygon) > 4:
            xs = polygon[0::2]
            ys = polygon[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            x1, y1, x2, y2 = polygon

        # Clamp bounding box coordinates to the image dimensions.
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Optionally inflate the bounding box by 10%.
        w_box = x2 - x1
        h_box = y2 - y1
        x1_infl = x1 - 0.05 * w_box
        y1_infl = y1 - 0.05 * h_box
        x2_infl = x2 + 0.05 * w_box
        y2_infl = y2 + 0.05 * h_box

        x1_infl = max(0, min(x1_infl, width - 1))
        y1_infl = max(0, min(y1_infl, height - 1))
        x2_infl = max(0, min(x2_infl, width - 1))
        y2_infl = max(0, min(y2_infl, height - 1))

        # Create a detection in CVAT format with a "points" key.
        detection = {
            "confidence": str(conf),
            "label": label,
            "type": "rectangle",
            "points": [int(x1_infl), int(y1_infl), int(x2_infl), int(y2_infl)]
        }
        all_detections.append(detection)

    # ------------------------------------------------------------------
    # Filter out duplicate detections via a simple IoU check (NMS-like)
    # ------------------------------------------------------------------
    iou_threshold = 0.5
    all_detections.sort(key=lambda d: float(d["confidence"]), reverse=True)
    final_detections = []
    for det in all_detections:
        boxA = (det["points"][0], det["points"][1], det["points"][2], det["points"][3])
        keep = True
        for kept in final_detections:
            boxB = (kept["points"][0], kept["points"][1], kept["points"][2], kept["points"][3])
            if iou(boxA, boxB) > iou_threshold:
                keep = False
                break
        if keep:
            final_detections.append(det)

    context.logger.info(f"Detections: {final_detections}")

    return context.Response(
        body=json.dumps(final_detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
