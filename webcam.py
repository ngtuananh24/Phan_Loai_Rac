import tkinter as tk
from tkinter import filedialog, Toplevel, Scale
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image, ImageTk

# Load mô hình YOLO
model = YOLO(r"D:\K57-KMT\Semester_2_2024-2025\ThiGiacMay\BTL\Waste-Classification-using-YOLOv8-main\best.pt")

# Biến toàn cục
cap = None
running = False
brightness_value = 50  # Giá trị độ sáng mặc định (0-100)

def adjust_brightness(img, brightness):
    """ Điều chỉnh độ sáng ảnh bằng Gamma Correction """
    gamma = 1.5 - (brightness / 100)  # Gamma càng cao thì ảnh càng sáng
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def start_webcam():
    global cap, running
    running = True

    # Tạo cửa sổ webcam
    webcam_window = Toplevel(root)
    webcam_window.title("📷 Webcam Nhận Diện")
    webcam_window.geometry("700x700")
    webcam_window.configure(bg="white")

    # Nút dừng webcam
    def stop_webcam():
        global running
        running = False
        cap.release()
        webcam_window.destroy()

    btn_stop = tk.Button(webcam_window, text="🛑 Dừng webcam", command=stop_webcam, font=("Arial", 14), bg="red", fg="white")
    btn_stop.pack(pady=10)

    # Thanh điều chỉnh độ sáng
    def update_brightness(value):
        global brightness_value
        brightness_value = int(value)

    scale_brightness = Scale(webcam_window, from_=0, to=100, orient="horizontal", label="Độ sáng", command=update_brightness)
    scale_brightness.set(brightness_value)
    scale_brightness.pack()

    # Label hiển thị video
    label_video = tk.Label(webcam_window, bg="white")
    label_video.pack()

    cap = cv2.VideoCapture(0)

    def update_frame():
        global running
        if not running:
            return

        ret, frame = cap.read()
        if not ret:
            stop_webcam()
            return

        # Giảm sáng ảnh
        frame = adjust_brightness(frame, brightness_value)

        # Resize ảnh
        img_resized = cv2.resize(frame, (640, 640))

        # Chạy YOLO dự đoán
        results = model(img_resized)
        for r in results:
            img_plot = r.plot()

        # Chuyển đổi ảnh từ BGR sang RGB
        img_rgb = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Hiển thị ảnh
        label_video.img_tk = img_tk
        label_video.configure(image=img_tk)

        # Cập nhật lại sau 200ms
        webcam_window.after(200, update_frame)

    update_frame()

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (640, 640))

    # Chạy YOLO
    results = model(img_resized)
    for r in results:
        img_plot = r.plot()

    # Hiển thị kết quả
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Ảnh đã nhận diện")
    plt.show()

# Tạo giao diện chính
root = tk.Tk()
root.title("♻ Nhận diện rác thải bằng YOLOv8")
root.geometry("400x350")
root.configure(bg="#f0f0f0")

# Tiêu đề
title_label = tk.Label(root, text="♻ Nhận diện rác thải thông minh", font=("Arial", 16, "bold"), fg="blue", bg="#f0f0f0")
title_label.pack(pady=10)

# Nút chọn ảnh
btn_select = tk.Button(root, text="🖼 Chọn ảnh", command=select_image, font=("Arial", 14), bg="#28a745", fg="white", padx=20, pady=10)
btn_select.pack(pady=10)

# Nút mở webcam
btn_webcam = tk.Button(root, text="📷 Mở webcam", command=start_webcam, font=("Arial", 14), bg="#007BFF", fg="white", padx=20, pady=10)
btn_webcam.pack(pady=10)

# Chạy giao diện
root.mainloop()
