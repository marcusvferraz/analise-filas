import cv2
from ultralytics import YOLO
import argparse
from datetime import datetime

TEMPO_POR_PESSOA = 2
LIMIAR_PEQUENA = 3
LIMIAR_MEDIA = 6

LINHA_X = 0.5
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
    cv2.rectangle(overlay, (0, 0), (w, 75), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(frame, f"Pessoas: {num_pessoas}  |  Tempo: {tempo_est}min  |  {status}",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_status, 2)

    cv2.putText(frame, agora, (w - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

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
    results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        linha_x = int(w * LINHA_X)
        display = cv2.flip(frame, 1)

        frame_count += 1
        if frame_count % skip == 0:
            results = model.track(frame, classes=[0], persist=True, verbose=False, imgsz=320)

            if results and results[0].boxes.id is not None:
                for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id):
                    track_id = int(tid.item())
                    x1, y1, x2, y2 = map(int, box.tolist())
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

        # Rebuild ids_dentro from currently visible people on the inside
        ids_dentro = set()
        if results and results[0].boxes.id is not None:
            for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id):
                track_id = int(tid.item())
                x1, y1, x2, y2 = map(int, box.tolist())
                cx = (x1 + x2) // 2
                if cx >= linha_x:
                    ids_dentro.add(track_id)

        num_pessoas = len(ids_dentro)
        tempo_est = num_pessoas * TEMPO_POR_PESSOA

        # Draw vertical virtual line (mirrored)
        linha_x_flip = w - linha_x
        cv2.line(display, (linha_x_flip, 0), (linha_x_flip, h), (255, 255, 0), 2)
        cv2.putText(display, "LINHA VIRTUAL", (linha_x_flip + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Entry/exit counters
        cv2.putText(display, f"E: {entry_count}  S: {exit_count}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw tracked boxes
        if results and results[0].boxes.id is not None:
            for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id):
                track_id = int(tid.item())
                x1, y1, x2, y2 = map(int, box.tolist())
                color = (0, 255, 0) if track_id in ids_dentro else (100, 100, 100)
                cv2.rectangle(display, (w - x1, y1), (w - x2, y2), color, 2)

        display = draw_overlay(display, num_pessoas, tempo_est)

        cv2.imshow("Analise de Filas - Pressione 'q' para sair", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
