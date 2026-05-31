import os
import glob
import random

# 1. Khai báo đường dẫn gốc (Dùng r"..." để Windows không bị lỗi ký tự đặc biệt)
val_dir = r"C:\Users\DUONG\Desktop\Paper_Q3_YOLO\VisDrone\VisDrone\images\val"
# Mình suy luận thư mục train của bạn sẽ nằm cùng cấp:
train_dir = r"C:\Users\DUONG\Desktop\Paper_Q3_YOLO\VisDrone\VisDrone\images\train"

# 2. Quét toàn bộ ảnh
train_imgs = glob.glob(os.path.join(train_dir, "*.jpg"))
val_imgs = glob.glob(os.path.join(val_dir, "*.jpg"))

# Gộp chung
all_imgs = train_imgs + val_imgs
print(f"Tổng số ảnh thu thập được: {len(all_imgs)} ảnh (Chuẩn là 7018)")

# 3. Trộn ngẫu nhiên
random.seed(42)  # Giữ nguyên seed để kết quả luôn cố định
random.shuffle(all_imgs)

# 4. Chia lại tỉ lệ
new_train = all_imgs[:6471]
new_val = all_imgs[6471:]

print(f"Tập Train mới: {len(new_train)} ảnh")
print(f"Tập Val mới:   {len(new_val)} ảnh")

# ========================================================
# 5. BƯỚC QUAN TRỌNG NHẤT CHO WINDOWS: Đổi "\" thành "/"
# ========================================================
new_train = [img.replace('\\', '/') for img in new_train]
new_val = [img.replace('\\', '/') for img in new_val]

# 6. Chọn nơi lưu 2 file .txt (Lưu ngay ngoài thư mục VisDrone cho gọn)
output_dir = r"C:\Users\DUONG\Desktop\Paper_Q3_YOLO\VisDrone\VisDrone"

with open(os.path.join(output_dir, "train_resplit.txt"), "w") as f:
    f.write("\n".join(new_train))

with open(os.path.join(output_dir, "val_resplit.txt"), "w") as f:
    f.write("\n".join(new_val))

print(f"Đã tạo xong file tại:\n - {output_dir}\\train_resplit.txt\n - {output_dir}\\val_resplit.txt")