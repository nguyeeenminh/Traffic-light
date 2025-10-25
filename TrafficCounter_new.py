import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# ================== Video Input ==================
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Check webcam
if not cap.isOpened():
    print("❌ Cannot open webcam!")
    exit()

# ================== Load YOLO model ==================
model = YOLO("car.pt")

# ================== Classes ==================
classNames = ["car"]

# ================== Tracking setup ==================
tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.3)

# ================== Define REGION (x1, y1, x2, y2) ==================
# chỉnh vùng ở giữa màn hình theo ý m, ví dụ như khung 600x300 giữa frame
region = [340, 210, 940, 510]  # [x1, y1, x2, y2]
totalCount = []
prev_count = -1

# ================== Main Loop ==================
while True:
    success, img = cap.read()
    if not success:
        print("⚠️ Frame not captured properly.")
        break

    results = model(img, stream=True)
    detections = np.empty((0, 5))

    # ===== Detection Phase =====
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Debug detection output
            print(f"Detected: {currentClass}, conf={conf}")

            # Only count vehicle-type classes
            if currentClass in ["car"] and conf > 0.2:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # ===== Tracking Phase =====
    resultsTracker = tracker.update(detections)

    # Draw region rectangle + label
    cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (0, 0, 255), 3)
    cv2.putText(img, "REGION", (region[0] + 10, region[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ===== Tracking and Counting =====
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = x2 - x1, y2 - y1

        # Draw bounding boxes + IDs
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'vid: {int(id)}', (max(0, x1), max(35, y1)),
                           scale=1.2, thickness=2, colorT=(0, 0, 0),
                           colorR=(240, 255, 255), offset=5)

        # Vehicle center point
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 3, (255, 255, 0), cv2.FILLED)

        # Check if center is inside the region
        if region[0] < cx < region[2] and region[1] < cy < region[3]:
            if id not in totalCount:
                totalCount.append(id)
                cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (0, 255, 0), 4)

    # ===== Display Count =====
    vehicle_count = len(totalCount)
    cvzone.putTextRect(img, f'Traffic Count: {vehicle_count}', (25, 50),
                       scale=3, thickness=2, colorT=(75, 0, 130),
                       colorR=(230, 230, 250), font=cv2.FONT_HERSHEY_PLAIN,
                       offset=10, border=2, colorB=(0, 0, 0))

    # ===== Save count only when changed =====
    if vehicle_count != prev_count:
        with open("../vehicle_count.txt", "w") as f:
            f.write(f"Traffic Count is {vehicle_count}")
        prev_count = vehicle_count

    # ===== Show output =====
    cv2.imshow("Traffic Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
