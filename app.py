import base64
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    img_bytes = base64.b64decode(data["image"].split(",")[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame, classes=[0])

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": round(conf, 2)})

    return jsonify({"count": len(detections), "boxes": detections})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
