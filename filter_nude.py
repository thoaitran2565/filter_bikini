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
OUTPUT_DIR = r"D:\Code\tools\fillter_image\result_nude" # Thư mục lưu ảnh khỏa thân lọc được
HISTORY_FILE = r"D:\Code\tools\fillter_image\processed_nude_history.txt" # File ghi nhớ lịch sử quét nude
THRESHOLD = 0.45                     # TĂNG MẠNH THRESHOLD lên để lọc cực kỳ khắt khe
MAX_WORKERS = 4                      # Số luồng chạy song song

# Nhãn mục tiêu (Mô tả chi tiết và khắt khe hơn)
POSITIVE_LABELS = [
    "a photo of an explicit nude woman showing full body parts without clothes",
    "an adult erotic photo of a completely naked woman",
    "explicit full frontal nudity of a woman"
]

# Nhãn phủ định (Bổ sung thêm ảnh chân dung, ảnh da thịt góc gần, đồ bơi, đồ lót)
NEGATIVE_LABELS = [
    "a close up selfie or portrait of a face",
    "a woman wearing a bikini or lingerie or underwear",
    "a woman wearing casual clothes or dress",
    "a photo focusing only on skin or shoulders with clothes on"
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
        # Điều kiện: Điểm tích cực vượt THRESHOLD và cao hơn điểm tiêu cực (đồ bơi, đồ lót, quần áo thường)
        if max_positive_score >= THRESHOLD and max_positive_score > max_negative_score:
            dest_path = os.path.join(OUTPUT_DIR, file_name)
            shutil.copy2(file_path, dest_path)
            is_match = True
            
        save_history(file_name)
        return is_match
    except Exception as e:
        save_history(file_name)
        return False

def main():
    all_files = [
        f for f in os.listdir(SOURCE_DIR) 
        if f.lower().endswith(VALID_EXTENSIONS)
    ]
    
    processed_files = load_history()
    image_files = [f for f in all_files if f not in processed_files]
    
    print(f"TỔNG QUÁT NUDE - Số ảnh trong root: {len(all_files)}")
    print(f"Số ảnh đã được quét trước đó: {len(processed_files)}")
    print(f"Số ảnh mới cần quét đợt này: {len(image_files)}")

    if len(image_files) == 0:
        print("\nKhông có ảnh mới nào cần xử lý!")
        return

    copied_count = 0
    
    print(f"Bắt đầu quét với {MAX_WORKERS} luồng song song...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_image, file_name): file_name for file_name in image_files}
        
        with tqdm(total=len(image_files), desc="Processing Nude Images") as pbar:
            for future in as_completed(futures):
                try:
                    is_match = future.result()
                    if is_match:
                        copied_count += 1
                except Exception:
                    pass
                finally:
                    pbar.update(1)

    print(f"\nHoàn tất! Đã quét xong, lọc được {copied_count} tấm ảnh khỏa thân vào '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()