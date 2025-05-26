import os
import json
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Cấu hình ===
INPUT_JSONL = "short_text_clean_rnnt/metadata.jsonl"     # File nguồn bạn đưa vào
AUDIO_PREFIX = "/kaggle/input/short-text-stt-small-clean-ver1/short_text_clean_rnnt/audio"
OUTPUT_TRAIN = "train_metadata.jsonl"
OUTPUT_TEST = "test_metadata.jsonl"
SPLIT_RATIO = 0.95
NUM_WORKERS = 8  # hoặc 8 tùy máy
# === Hàm xử lý từng dòng ===
def process_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        item = json.loads(line)
        filename = item.get("audio_filepath")
        if not filename:
            return None
        # Gắn prefix đường dẫn
        item["audio_filepath"] = os.path.join(AUDIO_PREFIX, filename).replace("\\", "/")
        return item
    except Exception as e:
        print(f"[ERROR] Lỗi dòng: {e} → {line[:80]}")
        return None

# === Đọc và xử lý song song ===
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    lines = f.readlines()

results = []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(process_line, line) for line in lines]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Đang xử lý"):
        result = future.result()
        if result:
            results.append(result)

# === Chia train/test ===
random.shuffle(results)
split_idx = int(len(results) * SPLIT_RATIO)
train_data = results[:split_idx]
test_data = results[split_idx:]

# === Ghi ra file jsonl đúng định dạng
def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(OUTPUT_TRAIN, train_data)
write_jsonl(OUTPUT_TEST, test_data)

print(f"[✅] Train: {len(train_data)} → {OUTPUT_TRAIN}")
print(f"[✅] Test : {len(test_data)} → {OUTPUT_TEST}")