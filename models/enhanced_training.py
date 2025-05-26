import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import warprnnt_numba
import sentencepiece as spm
from jiwer import wer
from loguru import logger
from torch import nn
from models.encoder import AudioEncoder
from models.enhanced_decoder import EnhancedDecoder
from models.enhanced_jointer import EnhancedJointer
from enhanced_constants import *
from utils.scheduler import WarmupLR

class FocalRNNTLoss(nn.Module):
    """Focal Loss variant cho RNNT để handle imbalanced short sentences"""
    def __init__(self, alpha=0.25, gamma=2.0, blank_id=1024):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.blank_id = blank_id
        self.rnnt_loss = warprnnt_numba.RNNTLossNumba(blank=blank_id, reduction="none")

    def forward(self, logits, targets, input_lengths, target_lengths):
        # Standard RNNT loss
        losses = self.rnnt_loss(logits, targets, input_lengths, target_lengths)

        # Apply focal loss weighting
        p = torch.exp(-losses)  # Convert loss to probability
        focal_weight = self.alpha * (1 - p) ** self.gamma
        focal_losses = focal_weight * losses

        return focal_losses.mean()

class EnhancedStreamingRNNT(pl.LightningModule):
    def __init__(self, att_context_size, vocab_size, tokenizer_model_path):
        super().__init__()

        # Load pretrained encoder
        encoder_state_dict = torch.load(PRETRAINED_ENCODER_WEIGHT,
                                      map_location="cuda" if torch.cuda.is_available() else "cpu",
                                      weights_only=True)
        encoder_state_dict['model_state_dict']['conv3.weight'] = encoder_state_dict['model_state_dict']['conv2.weight']
        encoder_state_dict['model_state_dict']['conv3.bias'] = encoder_state_dict['model_state_dict']['conv2.bias']

        self.encoder = AudioEncoder(
            n_mels=N_MELS,
            n_state=N_STATE,
            n_head=N_HEAD,
            n_layer=N_LAYER,
            att_context_size=att_context_size
        )
        self.encoder.load_state_dict(encoder_state_dict['model_state_dict'], strict=False)

        # 🚨 NOTE: Make sure these imports work in your environment
        # You might need to adjust the import paths
        self.decoder = EnhancedDecoder(vocab_size=vocab_size + 1, dropout=0.1)
        self.joint = EnhancedJointer(vocab_size=vocab_size + 1, dropout=0.1)

        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)

        # Multiple loss functions
        self.rnnt_loss = warprnnt_numba.RNNTLossNumba(blank=RNNT_BLANK, reduction="mean")
        self.focal_loss = FocalRNNTLoss(alpha=0.25, gamma=2.0, blank_id=RNNT_BLANK)

        # Optimizer with different learning rates cho different components
        self.optimizer = self._setup_optimizer()

        # Short sentence tracking
        self.short_sentence_threshold = 10  # tokens

    def _setup_optimizer(self):
        """Setup optimizer với different learning rates"""
        # Lower learning rate cho pretrained encoder
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())
        joint_params = list(self.joint.parameters())

        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': LR * 0.1},  # Lower LR cho pretrained
            {'params': decoder_params, 'lr': LR},
            {'params': joint_params, 'lr': LR}
        ], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

        return optimizer

    def length_aware_loss_weighting(self, loss, target_lengths):
        """Apply length-aware weighting cho short sentences"""
        # Short sentences get higher weight
        short_mask = (target_lengths <= self.short_sentence_threshold).float()
        long_mask = (target_lengths > self.short_sentence_threshold).float()

        # Weight: short sentences = 2.0, long sentences = 1.0
        weights = short_mask * 2.0 + long_mask * 1.0

        return loss * weights.mean()

    def enhanced_greedy_decoding(self, x, x_len, max_symbols=10, use_length_penalty=True):
        """Enhanced decoding với length penalty cho short sentences"""
        enc_out, _ = self.encoder(x, x_len)
        all_sentences = []

        for batch_idx in range(enc_out.shape[0]):
            hypothesis = [[None, None, 0.0]]  # [label, state, score]
            seq_enc_out = enc_out[batch_idx, :, :].unsqueeze(0)
            seq_ids = []

            for time_idx in range(seq_enc_out.shape[1]):
                current_seq_enc_out = seq_enc_out[:, time_idx, :].unsqueeze(1)

                not_blank = True
                symbols_added = 0

                while not_blank and symbols_added < max_symbols:
                    if hypothesis[-1][0] is None:
                        last_token = torch.tensor([[RNNT_BLANK]], dtype=torch.long, device=seq_enc_out.device)
                        last_seq_h_n = None
                    else:
                        last_token = hypothesis[-1][0]
                        last_seq_h_n = hypothesis[-1][1]

                    if last_seq_h_n is None:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token)
                    else:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token, last_seq_h_n)

                    logits = self.joint(current_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]

                    # Apply temperature scaling cho short sentences
                    temperature = 0.8 if len(seq_ids) < 5 else 1.0
                    logits = logits / temperature

                    probs = F.softmax(logits, dim=-1)
                    _, token_id = probs.max(0)
                    token_id = token_id.detach().item()

                    if token_id == RNNT_BLANK:
                        not_blank = False
                    else:
                        symbols_added += 1
                        score = probs[token_id].item()
                        hypothesis.append([
                            torch.tensor([[token_id]], dtype=torch.long, device=current_seq_enc_out.device),
                            current_seq_h_n,
                            hypothesis[-1][2] + np.log(score)
                        ])
                        seq_ids.append(token_id)

            # Length penalty cho short sentences
            if use_length_penalty and len(seq_ids) < 5:
                # Boost score cho short sentences
                final_score = hypothesis[-1][2] + len(seq_ids) * 0.1
            else:
                final_score = hypothesis[-1][2]

            all_sentences.append(self.tokenizer.decode(seq_ids))

        return all_sentences

    def training_step(self, batch, batch_idx):
        # 🔥 CHANGE 1: Handle 5 values instead of 4
        x, x_len, y, y_len, durations = batch  # ✅ Added durations

        # Enhanced decoding evaluation every 1000 steps
        if batch_idx != 0 and batch_idx % 1000 == 0:
            all_pred = self.enhanced_greedy_decoding(x, x_len, max_symbols=10)
            all_true = []
            for i, y_i in enumerate(y):
                y_i = y_i.cpu().numpy().astype(int).tolist()
                y_i = y_i[:y_len[i]]
                all_true.append(self.tokenizer.decode_ids(y_i))

            # Separate WER cho short và long sentences
            short_pred, short_true = [], []
            long_pred, long_true = [], []

            for i, (pred, true) in enumerate(zip(all_pred, all_true)):
                if y_len[i] <= self.short_sentence_threshold:
                    short_pred.append(pred)
                    short_true.append(true)
                else:
                    long_pred.append(pred)
                    long_true.append(true)

            if short_pred:
                short_wer = wer(short_true, short_pred)
                self.log("train_short_wer", short_wer, prog_bar=True, on_step=True)

            if long_pred:
                long_wer = wer(long_true, long_pred)
                self.log("train_long_wer", long_wer, prog_bar=False, on_step=True)

        # Forward pass
        enc_out, x_len = self.encoder(x, x_len)

        y_start = torch.cat([torch.full((y.shape[0], 1), RNNT_BLANK, dtype=torch.int).to(y.device), y], dim=1)
        dec_out, _ = self.decoder(y_start)
        logits = self.joint(enc_out, dec_out)

        # Curriculum learning: start with focal loss, gradually move to standard loss
        if self.current_epoch < 5:
            loss = self.focal_loss(logits.to(torch.float32), y.int(), x_len.int(), y_len.int())
        else:
            loss = self.rnnt_loss(logits.to(torch.float32), y.int(), x_len.int(), y_len.int())

        # Apply length-aware weighting
        loss = self.length_aware_loss_weighting(loss, y_len)

        # Regularization cho overfitting với small dataset
        if self.current_epoch > 10:
            # L2 regularization cho decoder và joint
            l2_reg = 0.001 * (
                sum(p.pow(2.0).sum() for p in self.decoder.parameters()) +
                sum(p.pow(2.0).sum() for p in self.joint.parameters())
            )
            loss = loss + l2_reg

        if batch_idx % 100 == 0:
            self.log("train_loss", loss.detach().item(), prog_bar=True, on_step=True)
            self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # 🔥 CHANGE 2: Handle 5 values instead of 4
        x, x_len, y, y_len, durations = batch  # ✅ Added durations

        all_pred = self.enhanced_greedy_decoding(x, x_len, max_symbols=15)
        all_true = []
        for i, y_i in enumerate(y):
            y_i = y_i.cpu().numpy().astype(int).tolist()
            y_i = y_i[:y_len[i]]
            all_true.append(self.tokenizer.decode_ids(y_i))

        # Detailed evaluation
        short_pred, short_true = [], []
        long_pred, long_true = [], []

        for i, (pred, true) in enumerate(zip(all_pred, all_true)):
            if y_len[i] <= self.short_sentence_threshold:
                short_pred.append(pred)
                short_true.append(true)
            else:
                long_pred.append(pred)
                long_true.append(true)

        # Calculate losses
        enc_out, x_len = self.encoder(x, x_len)
        y_start = torch.cat([torch.full((y.shape[0], 1), RNNT_BLANK, dtype=torch.int).to(y.device), y], dim=1)
        dec_out, _ = self.decoder(y_start)
        logits = self.joint(enc_out, dec_out)

        loss = self.rnnt_loss(logits.to(torch.float32), y.int(), x_len.int(), y_len.int())

        # Log metrics
        self.log("val_loss", loss.item(), prog_bar=True)

        if short_pred:
            short_wer = wer(short_true, short_pred)
            self.log("val_short_wer", short_wer, prog_bar=True, on_epoch=True)

        if long_pred:
            long_wer = wer(long_true, long_pred)
            self.log("val_long_wer", long_wer, prog_bar=False, on_epoch=True)

        overall_wer = wer(all_true, all_pred)
        self.log("val_wer", overall_wer, prog_bar=True, on_epoch=True)

        if batch_idx % 500 == 0:
            logger.info("=== SHORT SENTENCES ===")
            for pred, true in list(zip(short_pred, short_true))[:3]:
                logger.info(f"Pred: {pred}")
                logger.info(f"True: {true}")
                logger.info("---")

        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [{"scheduler": WarmupLR(self.optimizer, WARMUP_STEPS, TOTAL_STEPS, MIN_LR), "interval": "step"}],
        )