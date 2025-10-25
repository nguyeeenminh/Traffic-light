from ultralytics import YOLO
import cv2
# Load mô hình YOLOv8 pre-trained
model = YOLO('car.pt')

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera!")
    exit()

CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng độ tin cậy

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán với YOLOv8
    results = model(frame)

    # Xử lý các bounding box
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue  # Bỏ qua nếu độ tin cậy thấp

            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 - Phát hiện đối tượng", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
