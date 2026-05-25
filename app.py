import base64
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")

LINHA_X = 0.5
track_history = {}
entry_count = 0
exit_count = 0
ids_dentro = set()
frame_num = 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    global entry_count, exit_count, ids_dentro, track_history, frame_num

    data = request.get_json()
    img_bytes = base64.b64decode(data["image"].split(",")[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    linha_x = int(w * LINHA_X)

    frame_num += 1

    results = model.track(frame, classes=[0], persist=True, verbose=False, imgsz=320)

    detections = []
    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            track_id = int(boxes.id[i].item())
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            conf = float(boxes.conf[i].item())
            cx = (x1 + x2) // 2

            if track_id in track_history:
                prev_x = track_history[track_id]
                if prev_x < linha_x and cx >= linha_x:
                    entry_count += 1
                    ids_dentro.add(track_id)
                elif prev_x >= linha_x and cx < linha_x:
                    exit_count += 1
                    ids_dentro.discard(track_id)

            track_history[track_id] = cx
            detections.append({
                "id": track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": round(conf, 2),
            })

    # Clean stale tracks
    active = {d["id"] for d in detections}.union(ids_dentro)
    track_history = {k: v for k, v in track_history.items() if k in active}

    return jsonify({
        "count": len(ids_dentro),
        "boxes": detections,
        "entradas": entry_count,
        "saidas": exit_count,
        "linha_x": linha_x,
        "frame_w": w,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
