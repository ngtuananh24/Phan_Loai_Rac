from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load mô hình
model = YOLO(r"D:\K57-KMT\Semester_2_2024-2025\ThiGiacMay\BTL\Waste-Classification-using-YOLOv8-main\streamlit-detection-tracking - app\weights\yoloooo.pt")

# Load ảnh
img_path = "rac_1.jpg"  # Đổi thành đường dẫn ảnh của bạn
img = cv2.imread(img_path)

# Chạy mô hình dự đoán
results = model(img)

# Vẽ kết quả lên ảnh
for r in results:
    img_plot = r.plot()  # Vẽ bounding box lên ảnh

# Hiển thị ảnh
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Load ảnh
img_path = "out.png"  # Đổi thành đường dẫn ảnh của bạn
img = cv2.imread(img_path)

# Chạy mô hình dự đoán
results = model(img)

# Vẽ kết quả lên ảnh
for r in results:
    img_plot = r.plot()  # Vẽ bounding box lên ảnh

# Hiển thị ảnh
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
