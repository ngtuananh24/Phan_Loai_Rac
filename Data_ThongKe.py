import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Danh sách tên class theo ID
class_names = ['Pin', 'Lon', 'Bia_Cac_Tong', 'Hop dung do uong', 'Chai thuy tinh',
               'Giay', 'Tui_Nhua', 'Chai_nhua', 'Nhap_chai_nhua', 'nap_bat_lon']

# Đường dẫn thư mục chứa file nhãn
label_dir = r"D:\K57-KMT\Semester_2_2024-2025\ThiGiacMay\BTL\Waste-Classification-using-YOLOv8-main\data2\train\labels"

# Dictionary lưu số lượng mẫu của mỗi class
class_count = defaultdict(int)

# Duyệt qua tất cả file txt trong thư mục
for file_name in os.listdir(label_dir):
    if file_name.endswith(".txt"):  # Chỉ xử lý file nhãn
        with open(os.path.join(label_dir, file_name), "r") as f:
            for line in f.readlines():
                class_id = int(line.split()[0])  # Lấy ID class
                class_count[class_id] += 1  # Tăng số lượng mẫu của class đó

# Hiển thị thống kê
print("📊 Thống kê số lượng mẫu trong từng class:")
for class_id, count in sorted(class_count.items()):
    class_name = class_names[class_id]
    print(f"{class_name}: {count} mẫu")

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.bar([class_names[i] for i in class_count.keys()], class_count.values(), color='skyblue')

plt.xlabel("Class")
plt.ylabel("Số lượng mẫu")
plt.title("Thống kê số lượng mẫu trong từng class")
plt.xticks(rotation=45)  # Xoay tên class để dễ nhìn
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Hiển thị số lượng trên từng cột
for i, v in enumerate(class_count.values()):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=10)

plt.show()
