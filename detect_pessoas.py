import cv2
from ultralytics import YOLO
import argparse

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        results = model(frame, classes=[0])  # classe 0 = pessoa no COCO

        num_pessoas = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cv2.rectangle(display, (w - x1, y1), (w - x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{conf:.2f}", (w - x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                num_pessoas += 1

        cv2.putText(display, f"Pessoas: {num_pessoas}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Analise de Filas - Pressione 'q' para sair", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
