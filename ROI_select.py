import cv2
import numpy as np
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Đã chọn điểm: ({x}, {y})")

cap = cv2.VideoCapture(0)
cv2.namedWindow("Chọn 4 điểm")
cv2.setMouseCallback("Chọn 4 điểm", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Vẽ các điểm đã chọn
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    # Vẽ khung nếu đủ 4 điểm
    if len(points) == 4:
        cv2.polylines(frame, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)

    cv2.imshow("Chọn 4 điểm", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Lưu 4 điểm để dùng cho phần sau
import os, pickle

save_path = os.path.join(os.path.dirname(__file__), "roi_points.pkl")
with open(save_path, "wb") as f:
    pickle.dump(points, f)
print("ROI saved to:", save_path)

