import tkinter as tk
from tkinter import filedialog, Toplevel
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load mô hình YOLO
model = YOLO(r"D:\K57-KMT\Semester_2_2024-2025\ThiGiacMay\BTL\Waste-Classification-using-YOLOv8-main\best.pt")

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # Đọc ảnh
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (640, 640))

    # Chạy YOLO dự đoán
    results = model(img_resized)
    for r in results:
        img_plot = r.plot()

    # Chuyển ảnh sang định dạng Tkinter
    img_rgb = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Tạo cửa sổ mới để hiển thị ảnh
    result_window = Toplevel(root)
    result_window.title("Kết quả nhận diện")
    result_window.geometry("700x700")
    result_window.configure(bg="white")

    # Tiêu đề
    label_title = tk.Label(result_window, text="🔍 Kết quả nhận diện", font=("Arial", 18, "bold"), fg="blue", bg="white")
    label_title.pack(pady=10)

    # Hiển thị ảnh nhận diện
    label_img = tk.Label(result_window, image=img_tk, bg="white")
    label_img.image = img_tk
    label_img.pack(pady=10)

# Tạo cửa sổ chính
root = tk.Tk()
root.title("♻ Nhận diện rác thải")
root.geometry("400x300")
root.configure(bg="#f0f0f0")

# Tiêu đề
title_label = tk.Label(root, text="♻ Nhận diện rác thải thông minh", font=("Arial", 18, "bold"), fg="#007BFF", bg="#f0f0f0")
title_label.pack(pady=10)

# Nút chọn ảnh
btn_select = tk.Button(root, text="📷 Chọn ảnh", command=select_image, font=("Arial", 16), bg="#28a745", fg="white", padx=20, pady=10)
btn_select.pack(pady=20)

# Chạy giao diện
root.mainloop()
