import torch

# Audio parameters (FIXED)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

# Training parameters - Optimized cho short sentences
BATCH_SIZE = 16  # Giảm batch size để stable training với ít data
NUM_WORKERS = 4
MAX_EPOCHS = 100  # Tăng epochs cho better convergence với ít data
ATTENTION_CONTEXT_SIZE = (40, 2)  # Giảm context cho short sentences

# Path
# PRETRAINED_ENCODER_WEIGHT = '/kaggle/input/encoder-rnnt-swhiper-small/small_encoder.pt'
# BG_NOISE_PATH = ["/path/to/AudioSet", "/path/to/musan", "/path/to/FSDnoisy18k"]
# TRAIN_MANIFEST = ["./data/short_sentences.jsonl"]  # Dataset có nhiều câu ngắn
# VAL_MANIFEST = ["./data/val_short_sentences.jsonl"]
PRETRAINED_ENCODER_WEIGHT = '/kaggle/input/encoder-rnnt-swhiper-small/small_encoder.pt'
BG_NOISE_PATH = ["/kaggle/working/datatest/noise/fsdnoisy18k"]
TRAIN_MANIFEST = ["/kaggle/input/jsonl-stt-short-text-rnnt/train_metadata.jsonl"]
VAL_MANIFEST = ["/kaggle/input/jsonl-stt-short-text-rnnt/test_metadata.jsonl"]
LOG_DIR = './checkpoints_enhanced'


# Optimizer parameters - Điều chỉnh cho stable training
TOTAL_STEPS = 2000000  # Giảm total steps
WARMUP_STEPS = 1000    # Giảm warmup
LR = 5e-5              # Giảm learning rate cho stable training
MIN_LR = 1e-6

# Tokenizer parameters
VOCAB_SIZE = 1024
TOKENIZER_MODEL_PATH = './weights/tokenizer_spe_bpe_v1024_pad/tokenizer.model'
RNNT_BLANK = 1024
PAD = 1

# Enhanced decoding parameters cho short sentences
MAX_SYMBOLS = 15       # Tăng từ 3 lên 15
MIN_DURATION = 0.3     # Giảm từ 0.9 xuống 0.3
MAX_DURATION = 10.0    # Giảm từ 15.1 xuống 10.0
MIN_TEXT_LEN = 1       # Giảm từ 3 xuống 1

# Whisper-small parameters
N_STATE = 768
N_HEAD = 12
N_LAYER = 12

# Short sentence specific parameters
SHORT_SENTENCE_THRESHOLD = 2.0  # seconds
SHORT_SENTENCE_BOOST = 3.0      # data augmentation multiplier
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
