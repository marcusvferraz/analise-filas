import base64
import os
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from ultralytics import YOLO

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "secret!")
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8n.pt")


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("frame")
def handle_frame(data):
    img_bytes = base64.b64decode(data.split(",")[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(frame, classes=[0])

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": round(conf, 2)})

    socketio.emit("detections", {
        "count": len(detections),
        "boxes": detections,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
