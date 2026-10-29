import os
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# --- CẤU HÌNH ĐƯỜNG DẪN & TỪ KHÓA ---
SOURCE_DIR = r"D:\Ảnh\Ảnh đã tải"        # Thư mục chứa ảnh gốc
OUTPUT_DIR = r"D:\Code\tools\fillter_image\result"     # Thư mục lưu ảnh bikini lọc được
HISTORY_FILE = r"D:\Code\tools\fillter_image\processed_history.txt" # File ghi nhớ lịch sử đã quét
THRESHOLD = 0.30                     # Ngưỡng khớp
MAX_WORKERS = 4                      # Số luồng chạy song song (có thể tăng lên 6 hoặc 8 nếu máy khỏe)

# Nhãn mục tiêu (Mong muốn giữ lại)
POSITIVE_LABELS = [
    "a photo of a woman wearing a two-piece bikini",
    "a woman in a sexy bikini on the beach or swimming pool",
]

# Nhãn phủ định (Loại bỏ các trang phục khác)
NEGATIVE_LABELS = [
    "a woman wearing a normal t-shirt and jeans or casual clothes",
    "a woman wearing sports gym clothes or activewear or sports bra",
    "a woman wearing a long dress",
    "a close up portrait of a face"
]

ALL_LABELS = POSITIVE_LABELS + NEGATIVE_LABELS

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- KHỞI TẠO MÔ HÌNH CLIP ---
MODEL_NAME = "openai/clip-vit-base-patch32"
print(f"Đang tải mô hình AI chuẩn từ Hugging Face ({MODEL_NAME})...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy trên thiết bị: {device.upper()}")

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

# Khóa an toàn khi ghi file lịch sử từ nhiều luồng cùng lúc
history_lock = threading.Lock()

def load_history():
    """Đọc danh sách các file đã từng được quét từ file lịch sử."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_history(file_name):
    """Ghi thêm tên file vừa quét xong vào file lịch sử (Thread-safe)."""
    with history_lock:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(file_name + "\n")

def process_single_image(file_name):
    """Hàm xử lý cho từng ảnh độc lập (chạy trong luồng riêng)."""
    file_path = os.path.join(SOURCE_DIR, file_name)
    try:
        image = Image.open(file_path).convert("RGB")
        inputs = processor(
            text=ALL_LABELS, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            
        max_positive_score = max([probs[i].item() for i in range(len(POSITIVE_LABELS))])
        max_negative_score = max([probs[i].item() for i in range(len(POSITIVE_LABELS), len(ALL_LABELS))])
        
        is_match = False
        if max_positive_score >= THRESHOLD and max_positive_score > max_negative_score:
            dest_path = os.path.join(OUTPUT_DIR, file_name)
            shutil.copy2(file_path, dest_path)
            is_match = True
            
        # Lưu vào lịch sử sau khi xử lý xong
        save_history(file_name)
        return is_match
    except Exception as e:
        # Nếu lỗi file ảnh, vẫn ghi vào lịch sử để không bị lặp lại ở lần sau
        save_history(file_name)
        return False

def main():
    all_files = [
        f for f in os.listdir(SOURCE_DIR) 
        if f.lower().endswith(VALID_EXTENSIONS)
    ]
    
    processed_files = load_history()
    image_files = [f for f in all_files if f not in processed_files]
    
    print(f"Tổng số ảnh trong root: {len(all_files)}")
    print(f"Số ảnh đã được quét trước đó: {len(processed_files)}")
    print(f"Số ảnh mới cần quét đợt này: {len(image_files)}")

    if len(image_files) == 0:
        print("\nKhông có ảnh mới nào cần xử lý!")
        return

    copied_count = 0
    
    # Sử dụng ThreadPoolExecutor để chạy đa luồng song song
    print(f"Bắt đầu quét với {MAX_WORKERS} luồng song song...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tất cả các tác vụ vào pool
        futures = {executor.submit(process_single_image, file_name): file_name for file_name in image_files}
        
        # Hiển thị thanh tiến trình hoàn thành
        with tqdm(total=len(image_files), desc="Multithreading Processing") as pbar:
            for future in as_completed(futures):
                try:
                    is_match = future.result()
                    if is_match:
                        copied_count += 1
                except Exception as exc:
                    pass
                finally:
                    pbar.update(1)

    print(f"\nHoàn tất! Đã quét xong, lọc thêm được {copied_count} tấm ảnh bikini mới vào '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()