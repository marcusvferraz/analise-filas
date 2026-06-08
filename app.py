import base64
import os
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import database as db

app = Flask(__name__)
model = YOLO("yolov8n.pt")

ENTRADA_X = 0.80
SAIDA_X = 0.20
track_history = {}
entry_count = 0
exit_count = 0
ids_dentro = set()
frame_num = 0
save_counter = 0

# Real wait time tracking
pessoa_entrada = {}
soma_tempos_reais = 0.0
total_saidas_tracked = 0

try:
    db.criar_tabela()
    db_disponivel = True
    print("[DB] MySQL conectado - analise_filas pronto")
except Exception as e:
    db_disponivel = False
    print(f"[DB] MySQL indisponivel: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    global entry_count, exit_count, ids_dentro, track_history, frame_num, save_counter
    global pessoa_entrada, soma_tempos_reais, total_saidas_tracked

    data = request.get_json()
    img_bytes = base64.b64decode(data["image"].split(",")[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    entrada_x = int(w * ENTRADA_X)
    saida_x = int(w * SAIDA_X)

    frame_num += 1

    results = model.track(frame, classes=[0], persist=True, verbose=False, imgsz=480)

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
                estava_dentro = saida_x <= prev_x <= entrada_x
                esta_dentro = saida_x <= cx <= entrada_x

                if not estava_dentro and esta_dentro:
                    entry_count += 1
                    ids_dentro.add(track_id)
                    pessoa_entrada[track_id] = time.time()
                elif estava_dentro and not esta_dentro:
                    exit_count += 1
                    ids_dentro.discard(track_id)
                    if track_id in pessoa_entrada:
                        duracao = (time.time() - pessoa_entrada.pop(track_id)) / 60
                        soma_tempos_reais += duracao
                        total_saidas_tracked += 1

            track_history[track_id] = cx
            detections.append({
                "id": track_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": round(conf, 2),
            })

    # Clean stale tracks
    active = {d["id"] for d in detections}.union(ids_dentro)
    track_history = {k: v for k, v in track_history.items() if k in active}

    # Real average wait time (minutes)
    tempo_real_medio = round(soma_tempos_reais / total_saidas_tracked, 1) if total_saidas_tracked > 0 else None

    # Save metrics every 10 detect calls
    num_pessoas = len(ids_dentro)
    save_counter += 1
    if db_disponivel and save_counter >= 10:
        save_counter = 0
        try:
            db.salvar_metrica(num_pessoas, num_pessoas * 2, entry_count, exit_count, tempo_real_medio)
        except Exception:
            pass

    return jsonify({
        "count": num_pessoas,
        "boxes": detections,
        "entradas": entry_count,
        "saidas": exit_count,
        "entrada_x": entrada_x,
        "saida_x": saida_x,
        "frame_w": w,
        "tempo_real_medio": tempo_real_medio,
    })


@app.route("/historico")
def historico():
    return render_template("historico.html")


@app.route("/api/historico")
def api_historico():
    if not db_disponivel:
        return jsonify({"erro": "Banco de dados indisponivel"}), 503
    try:
        dados = db.listar_historico(500)
        return jsonify(dados)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/api/estatisticas")
def api_estatisticas():
    if not db_disponivel:
        return jsonify({"erro": "Banco de dados indisponivel"}), 503
    try:
        dados = db.estatisticas()
        return jsonify(dados)
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
