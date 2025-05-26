import os
import csv
import json
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Cấu hình ===
INPUT_CSV = "short_text_clean_rnnt/metadata.csv"
AUDIO_DIR = "short_text_clean_rnnt/audio"
OUTPUT_JSONL = "short_text_clean_rnnt/metadata.jsonl"
NUM_WORKERS = 8  # Tùy vào CPU

# === Hàm xử lý từng dòng ===
def process_row(row):
    filename = row["filename"]
    text = row["transcription"].strip()
    audio_path = os.path.join(AUDIO_DIR, filename)

    if not os.path.isfile(audio_path):
        print(f"[WARNING] File không tồn tại: {audio_path}")
        return None

    try:
        with sf.SoundFile(audio_path) as f:
            duration = len(f) / f.samplerate
        return {
            "audio_filepath": filename,
            "duration": round(duration, 2),
            "offset": 0,
            "text": text
        }
    except Exception as e:
        print(f"[ERROR] Không thể xử lý {filename}: {e}")
        return None

# === Đọc toàn bộ CSV ===
with open(INPUT_CSV, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    data = list(reader)

# === Xử lý song song ===
results = []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(process_row, row) for row in data]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Converting to JSONL"):
        result = future.result()
        if result:
            results.append(result)

# === Ghi ra JSONL ===
with open(OUTPUT_JSONL, mode="w", encoding="utf-8") as f_out:
    for item in results:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
