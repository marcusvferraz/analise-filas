import cv2
from ultralytics import YOLO
import argparse
from datetime import datetime

TEMPO_POR_PESSOA = 2
LIMIAR_PEQUENA = 3
LIMIAR_MEDIA = 6

LINHA_Y = 0.6
track_history = {}
entry_count = 0
exit_count = 0
ids_dentro = set()


def get_status(qtd):
    if qtd == 0:
        return "SEM FILA", (0, 255, 0)
    if qtd < LIMIAR_PEQUENA:
        return "FILA PEQUENA", (0, 255, 0)
    if qtd < LIMIAR_MEDIA:
        return "FILA MÉDIA", (0, 255, 255)
    return "FILA LOTADA", (0, 0, 255)


def draw_overlay(frame, num_pessoas, tempo_est):
    h, w = frame.shape[:2]
    agora = datetime.now().strftime("%H:%M:%S")
    status, cor_status = get_status(num_pessoas)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "ANALISE DE FILAS", (20, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    cv2.putText(frame, agora, (w - 120, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 180), 1)

    cv2.putText(frame, f"Pessoas: {num_pessoas}  |  Tempo estimado: {tempo_est}min",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.putText(frame, f"Status: {status}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 2)

    bar_x, bar_y, bar_w, bar_h = 20, 95, 200, 8
    razao = min(num_pessoas / LIMIAR_MEDIA, 1.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (int(bar_x + bar_w * razao), bar_y + bar_h), cor_status, -1)

    return frame


model = YOLO("yolov8n.pt")


def main():
    global entry_count, exit_count, ids_dentro, track_history

    parser = argparse.ArgumentParser(description="Análise de Filas - Detecção de Pessoas")
    parser.add_argument("--source", type=str, default="0",
                        help="Caminho do vídeo ou '0' para webcam")
    args = parser.parse_args()

    source = int(args.source) if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Erro ao abrir a fonte de vídeo")
        return

    skip = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        linha_y = int(h * LINHA_Y)
        display = cv2.flip(frame, 1)
        linha_y_flip = int(h * (1 - LINHA_Y))

        frame_count += 1
        if frame_count % skip == 0:
            results = model.track(frame, classes=[0], persist=True, verbose=False)

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    track_id = int(boxes.id[i].item())
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    conf = boxes.conf[i].item()
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if track_id in track_history:
                        prev_y = track_history[track_id]
                        if prev_y < linha_y and cy >= linha_y:
                            entry_count += 1
                            ids_dentro.add(track_id)
                        elif prev_y >= linha_y and cy < linha_y:
                            exit_count += 1
                            ids_dentro.discard(track_id)

                    track_history[track_id] = cy

        # Clean old tracks
        track_history = {k: v for k, v in track_history.items()
                         if k in ids_dentro or
                         any(boxes.id is not None and int(b.id) == k
                             for r in [results] if r
                             for boxes in [r.boxes] if boxes.id is not None
                             for b in boxes)}

        num_pessoas = len(ids_dentro)
        tempo_est = num_pessoas * TEMPO_POR_PESSOA

        # Draw virtual line
        cv2.line(display, (0, linha_y_flip), (w, linha_y_flip), (255, 255, 0), 2)
        cv2.putText(display, "LINDA VIRTUAL", (10, linha_y_flip - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Entry/exit counters
        cv2.putText(display, f"Entradas: {entry_count}  Saidas: {exit_count}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw tracked boxes
        if results and results[0].boxes.id is not None:
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                track_id = int(box.id[i].item())
                x1, y1, x2, y2 = map(int, box.xyxy[i])
                conf = box.conf[i].item()
                color = (0, 255, 0) if track_id in ids_dentro else (100, 100, 100)
                cv2.rectangle(display, (w - x1, y1), (w - x2, y2), color, 2)
                cv2.putText(display, f"#{track_id}", (w - x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        display = draw_overlay(display, num_pessoas, tempo_est)

        cv2.imshow("Analise de Filas - Pressione 'q' para sair", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
