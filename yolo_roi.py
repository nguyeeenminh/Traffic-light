from ultralytics import YOLO
import cv2
import pickle
import numpy as np

# Load mô hình
model = YOLO('car.pt')

# Load 4 điểm vùng nhận diện
with open("roi_points.pkl", "rb") as f:
    roi_points = pickle.load(f)
roi_polygon = np.array(roi_points)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    found_vehicle = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            if conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Kiểm tra xem tâm của box có nằm trong vùng nhận diện không
            if cv2.pointPolygonTest(roi_polygon, (center_x, center_y), False) >= 0:
                found_vehicle = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Vẽ khung nhận diện
    cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    # Hiển thị thông báo
    if found_vehicle:
        cv2.putText(frame, "BAT DEN XANH", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "BAT DEN VANG", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Kiểm tra vùng nhận diện", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
