import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from enhanced_constants import *
from utils.enhanced_dataset import EnhancedAudioDataset, enhanced_collate_fn
from models.enhanced_training import EnhancedStreamingRNNT
from utils.scheduler import WarmupLR

def main():
    # Enhanced datasets với short sentence optimization
    train_dataset = EnhancedAudioDataset(
        manifest_files=TRAIN_MANIFEST,
        bg_noise_path=BG_NOISE_PATH,
        shuffle=True,
        augment=True,
        tokenizer_model_path=TOKENIZER_MODEL_PATH,
        max_duration=MAX_DURATION,
        min_duration=MIN_DURATION,
        min_text_len=MIN_TEXT_LEN,
        short_sentence_boost=SHORT_SENTENCE_BOOST,
        short_threshold=SHORT_SENTENCE_THRESHOLD
    )
    print("Hành thành 1")

    val_dataset = EnhancedAudioDataset(
        manifest_files=VAL_MANIFEST,
        shuffle=False,
        tokenizer_model_path=TOKENIZER_MODEL_PATH,
        max_duration=MAX_DURATION,
        min_duration=MIN_DURATION,
        min_text_len=MIN_TEXT_LEN,
        short_sentence_boost=1.0,  # No boost cho validation
        short_threshold=SHORT_SENTENCE_THRESHOLD
    )
    print("Hành thành 2")

    # Data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        collate_fn=enhanced_collate_fn,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        collate_fn=enhanced_collate_fn,
        pin_memory=True
    )

    # Enhanced model
    model = EnhancedStreamingRNNT(
        att_context_size=ATTENTION_CONTEXT_SIZE,
        vocab_size=VOCAB_SIZE,
        tokenizer_model_path=TOKENIZER_MODEL_PATH
    )

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_short_wer',  # Monitor short sentence WER specifically
        dirpath=LOG_DIR,
        filename='enhanced-rnnt-{epoch:02d}-{val_short_wer:.3f}',
        save_top_k=3,
        mode='min'
    )

    # Early stopping dựa trên short sentence performance
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_short_wer',
        patience=10,
        mode='min',
        min_delta=0.001
    )

    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Trainer với specific configuration cho short sentences
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed",
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=pl.loggers.TensorBoardLogger(LOG_DIR, name="enhanced_rnnt"),
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,  # Validate every epoch cho better monitoring
        gradient_clip_val=1.0,  # Gradient clipping cho stability
        accumulate_grad_batches=2,  # Gradient accumulation cho effective larger batch
        log_every_n_steps=50,
    )

    # Train model
    print("🚀 Starting enhanced training for short sentences...")
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    print(f"🎯 Short sentence threshold: {SHORT_SENTENCE_THRESHOLD}s")
    print(f"🔥 Short sentence boost factor: {SHORT_SENTENCE_BOOST}x")

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()  # Training
    # inference_short_sentences()  # Inference
    # analyze_short_sentence_performance()  # Analysis