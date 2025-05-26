import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from enhanced_constants import *
from utils.enhanced_dataset import EnhancedAudioDataset, enhanced_collate_fn
from models.enhanced_training import EnhancedStreamingRNNT
from utils.scheduler import WarmupLR

def inference_short_sentences():
    """Example inference cho short sentences"""

    # Load trained model
    model = EnhancedStreamingRNNT.load_from_checkpoint(
        checkpoint_path="./checkpoints_enhanced/enhanced-rnnt-best.ckpt",
        att_context_size=ATTENTION_CONTEXT_SIZE,
        vocab_size=VOCAB_SIZE,
        tokenizer_model_path=TOKENIZER_MODEL_PATH
    )
    model.eval()

    # Test short sentences
    test_audios = [
        "./test_audio/dung_roi.wav",     # "đúng rồi"
        "./test_audio/duoc.wav",         # "được"
        "./test_audio/khong.wav",        # "không"
        "./test_audio/co.wav",           # "có"
        "./test_audio/vang.wav",         # "vâng"
    ]

    for audio_path in test_audios:
        # Load và preprocess audio
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Convert to mel spectrogram
        dataset = EnhancedAudioDataset(
            manifest_files=[],  # Empty for inference
            tokenizer_model_path=TOKENIZER_MODEL_PATH
        )
        melspec = dataset.log_mel_spectrogram(
            torch.from_numpy(waveform), N_MELS, 0, 'cpu'
        )

        # Inference
        with torch.no_grad():
            mel_batch = melspec.unsqueeze(0)  # Add batch dimension
            mel_len = torch.tensor([melspec.shape[1]])

            # Enhanced decoding với length penalty
            predictions = model.enhanced_greedy_decoding(
                mel_batch, mel_len, max_symbols=10, use_length_penalty=True
            )

            print(f"🎵 {audio_path}: '{predictions[0]}'")


def analyze_short_sentence_performance():
    """Phân tích performance trên short sentences"""

    # Load model
    model = EnhancedStreamingRNNT.load_from_checkpoint(
        checkpoint_path="./checkpoints_enhanced/enhanced-rnnt-best.ckpt",
        att_context_size=ATTENTION_CONTEXT_SIZE,
        vocab_size=VOCAB_SIZE,
        tokenizer_model_path=TOKENIZER_MODEL_PATH
    )

    # Load test dataset
    test_dataset = EnhancedAudioDataset(
        manifest_files=["./data/test_short_sentences.jsonl"],
        tokenizer_model_path=TOKENIZER_MODEL_PATH,
        augment=False
    )

    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=enhanced_collate_fn)

    short_correct = 0
    short_total = 0
    long_correct = 0
    long_total = 0

    for batch in test_loader:
        x, x_len, y, y_len, durations = batch

        # Predict
        predictions = model.enhanced_greedy_decoding(x, x_len)

        # Ground truth
        y_cpu = y[0].cpu().numpy()[:y_len[0]]
        ground_truth = model.tokenizer.decode_ids(y_cpu.tolist())

        # Check accuracy
        is_correct = predictions[0].strip() == ground_truth.strip()

        if durations[0] <= SHORT_SENTENCE_THRESHOLD:
            short_total += 1
            if is_correct:
                short_correct += 1
        else:
            long_total += 1
            if is_correct:
                long_correct += 1

    print(f"📊 Short sentences accuracy: {short_correct}/{short_total} = {short_correct/short_total*100:.1f}%")
    print(f"📊 Long sentences accuracy: {long_correct}/{long_total} = {long_correct/long_total*100:.1f}%")

if __name__ == "__main__":
    # main()  # Training
    # inference_short_sentences()  # Inference
    analyze_short_sentence_performance()  # Analysis