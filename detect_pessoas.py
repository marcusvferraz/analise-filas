import cv2
from ultralytics import YOLO
import argparse

TEMPO_POR_PESSOA = 2  # minutos estimados por pessoa na fila
LIMIAR_PEQUENA = 3
LIMIAR_MEDIA = 6


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
    status, cor_status = get_status(num_pessoas)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, "ANALISE DE FILAS", (20, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

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
    last_boxes = []
    num_pessoas = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        frame_count += 1
        if frame_count % skip == 0:
            results = model(frame, classes=[0])

            last_boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    last_boxes.append((x1, y1, x2, y2, conf))
            num_pessoas = len(last_boxes)

        display = cv2.flip(frame, 1)

        for x1, y1, x2, y2, conf in last_boxes:
            cv2.rectangle(display, (w - x1, y1), (w - x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{conf:.2f}", (w - x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tempo_est = num_pessoas * TEMPO_POR_PESSOA
        display = draw_overlay(display, num_pessoas, tempo_est)

        cv2.imshow("Analise de Filas - Pressione 'q' para sair", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
