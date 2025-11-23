import cv2
import json
import numpy as np
import os
from ultralytics import YOLO
import torch
import easyocr
import random
import string
FRAME_STEP = 5
CONF_THRESH = 0.15
UP_SCALE = 1.0
YOLO_IMAGE_SIZE = 960

OUTPUT_JSON = "output/metadata.json"
os.makedirs("output", exist_ok=True)

torch.backends.cudnn.benchmark = True

print("\n🚀 Loading YOLOv8m vehicle/pedestrian detector…")
det_model = YOLO("yolov8m.pt")

print("🔍 Loading YOLOv11 license plate detector…")
lp_model = YOLO("models/license-plate-finetune-v1x.pt")

print("🔤 Loading EasyOCR… (GPU Enabled)")
ocr_reader = easyocr.Reader(['en'], gpu=True)

plate_text_cache = {}
color_cache = {}

def get_color(crop):
    if crop is None or crop.size == 0:
        return "unknown"

    small = cv2.resize(crop, (40, 40))
    avg = np.mean(small, axis=(0, 1))
    b, g, r = avg

    if r > g and r > b: return "red"
    if g > r and g > b: return "green"
    if b > r and b > g: return "blue"

    brightness = (r + g + b) / 3
    if brightness > 180: return "white"
    if brightness < 60: return "black"

    return "other"

def safe_crop(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎥 Loaded video | FPS={fps} | Frames={total}")

    metadata = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every FRAME_STEP
        if frame_id % FRAME_STEP != 0:
            frame_id += 1
            continue

        timestamp = frame_id / fps
        frame_up = cv2.resize(frame, None, fx=UP_SCALE, fy=UP_SCALE)

        
        results = det_model.track(
            frame_up,
            conf=CONF_THRESH,
            imgsz=YOLO_IMAGE_SIZE,
            persist=True,
            tracker="bytetrack.yaml",
            device=0
        )

        if results[0].boxes.id is None:
            frame_id += 1
            continue

        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

       
        lp_res = lp_model(frame_up, imgsz=320, conf=0.25)[0]
        lp_boxes = lp_res.boxes.xyxy.cpu().numpy() if lp_res.boxes else []
        lp_scores = lp_res.boxes.conf.cpu().numpy() if lp_res.boxes else []

       
        for i, track_id in enumerate(ids):

            label = det_model.names[clss[i]].lower()
            if label not in ["car", "truck", "bus", "motorcycle"]:
                continue

            x1, y1, x2, y2 = map(int, boxes[i])
            vehicle_crop = frame_up[y1:y2, x1:x2]

            # COLOR
            if track_id not in color_cache:
                color_cache[track_id] = get_color(vehicle_crop)

            vehicle_conf = float(confs[i])
            license_text = ""

           
            for j, lp_box in enumerate(lp_boxes):
                px1, py1, px2, py2 = map(int, lp_box)

                
                if px1 >= x1 and py1 >= y1 and px2 <= x2 and py2 <= y2:

                    
                    if track_id in plate_text_cache:
                        license_text = plate_text_cache[track_id]
                        break

                    
                    plate_crop = safe_crop(frame_up, lp_box)
                    if plate_crop is None or plate_crop.size == 0:
                        continue

                    
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape

                    if h < 40:   
                        scale = max(2, int(40 / h))
                        gray = cv2.resize(gray, (w*scale, h*scale))

                    
                    _, thresh = cv2.threshold(gray, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    
                    ocr_out = ocr_reader.readtext(thresh)
                    best_text = ""
                    best_score = 0

                    for det in ocr_out:
                        _, text, score = det
                        text = text.upper().replace(" ", "")
                        if len(text) >= 3 and score > best_score:
                            best_text = text
                            best_score = score

                    if best_text != "":
                        license_text = best_text
                        plate_text_cache[track_id] = best_text
                        break

            
            metadata.append({
                "track_id": int(track_id),
                "frame_id": int(frame_id),
                "timestamp": float(timestamp),
                "label": label,
                "confidence": vehicle_conf,
                "bbox": [x1, y1, x2, y2],
                "color": color_cache.get(track_id, "n/a"),
                "license_plate": license_text
            })

        frame_id += 1

    cap.release()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✔ Saved metadata to", OUTPUT_JSON)
    print("✔ Total detections:", len(metadata))



if __name__ == "__main__":
    process_video("2103099-uhd_3840_2160_30fps.mp4")
