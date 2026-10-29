import os
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN & TỪ KHÓA ---
SOURCE_DIR = r"D:\Ảnh\Ảnh đã tải"        # Đường dẫn chứa 20,000 tấm ảnh của bạn
OUTPUT_DIR = r"D:\Code\tools\fillter_image\result"     # Thư mục lưu ảnh lọc được
THRESHOLD = 0.30                     # Tăng ngưỡng lên để khắt khe và chính xác hơn

# Nhãn mục tiêu (Mong muốn giữ lại)
POSITIVE_LABELS = [
    "a photo of a woman wearing a two-piece bikini",
    "a woman in a sexy bikini on the beach or swimming pool",
]

# Nhãn phủ định (Những loại trang phục khác cần loại bỏ nếu điểm cao hơn)
NEGATIVE_LABELS = [
    "a woman wearing a normal t-shirt and jeans or casual clothes",
    "a woman wearing sports gym clothes or activewear or sports bra",
    "a woman wearing a long dress",
    "a close up portrait of a face"
]

ALL_LABELS = POSITIVE_LABELS + NEGATIVE_LABELS

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- KHỞI TẠO MÔ HÌNH CLIP ---
MODEL_NAME = "openai/clip-vit-base-patch32"
print(f"Đang tải mô hình AI chuẩn từ Hugging Face ({MODEL_NAME})...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy trên thiết bị: {device.upper()}")

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

def is_bikini(image_path):
    """Kiểm tra ảnh có chứa bikini chính xác thông qua đối chiếu Positive/Negative."""
    try:
        image = Image.open(image_path).convert("RGB")
        
        inputs = processor(
            text=ALL_LABELS, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            
        # Lấy điểm cao nhất của nhóm tích cực và nhóm tiêu cực
        max_positive_score = max([probs[i].item() for i in range(len(POSITIVE_LABELS))])
        max_negative_score = max([probs[i].item() for i in range(len(POSITIVE_LABELS), len(ALL_LABELS))])
        
        # Điều kiện khắt khe: Điểm tích cực phải vượt ngưỡng VÀ lớn hơn điểm tiêu cực
        if max_positive_score >= THRESHOLD and max_positive_score > max_negative_score:
            return True, max_positive_score
            
    except Exception as e:
        return False, 0.0
    return False, 0.0

def main():
    image_files = [
        f for f in os.listdir(SOURCE_DIR) 
        if f.lower().endswith(VALID_EXTENSIONS)
    ]
    
    print(f"Tìm thấy {len(image_files)} ảnh tại '{SOURCE_DIR}'. Bắt đầu quét...")

    copied_count = 0
    
    for file_name in tqdm(image_files, desc="Processing"):
        file_path = os.path.join(SOURCE_DIR, file_name)
        
        match, score = is_bikini(file_path)
        if match:
            dest_path = os.path.join(OUTPUT_DIR, file_name)
            shutil.copy2(file_path, dest_path)
            copied_count += 1

    print(f"\nHoàn tất! Đã lọc và sao chép thành công {copied_count} tấm ảnh vào '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()