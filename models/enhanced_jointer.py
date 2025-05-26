from torch import nn
import torch
import torch.nn.functional as F

class EnhancedJointer(nn.Module):
    def __init__(self, encoder_dim=768, decoder_dim=768, vocab_size=1024, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Projection layers để match dimensions
        self.encoder_projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.decoder_projection = nn.Sequential(
            nn.Linear(decoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-layer fusion network cho better short sentence handling
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention mechanism cho better encoder-decoder fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Output layer với better initialization
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )

        # Short sentence boost weights (learnable)
        self.short_sentence_weight = nn.Parameter(torch.ones(vocab_size))

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization cho all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, enc, dec):
        # enc: (B, T, encoder_dim)
        # dec: (B, U, decoder_dim)

        batch_size, T, _ = enc.shape
        _, U, _ = dec.shape

        # Project encoder và decoder features
        enc_proj = self.encoder_projection(enc)  # (B, T, hidden_dim)
        dec_proj = self.decoder_projection(dec)  # (B, U, hidden_dim)

        # Expand dimensions for broadcasting
        enc_expanded = enc_proj.unsqueeze(2).expand(-1, -1, U, -1)  # (B, T, U, hidden_dim)
        dec_expanded = dec_proj.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, U, hidden_dim)

        # Element-wise addition (original RNNT approach)
        combined_add = enc_expanded + dec_expanded  # (B, T, U, hidden_dim)

        # Apply fusion layers
        fused = self.fusion_layers(combined_add)  # (B, T, U, hidden_dim//2)

        # Cross-attention mechanism cho better fusion
        # Reshape for attention: (B*T*U, 1, hidden_dim)
        enc_reshaped = enc_expanded.reshape(-1, 1, self.hidden_dim)
        dec_reshaped = dec_expanded.reshape(-1, 1, self.hidden_dim)

        # Apply attention
        attended, _ = self.attention(
            query=enc_reshaped,
            key=dec_reshaped,
            value=dec_reshaped
        )  # (B*T*U, 1, hidden_dim)

        # Reshape back
        attended = attended.reshape(batch_size, T, U, self.hidden_dim)

        # Concatenate fused features với attended features
        final_features = torch.cat([fused, attended], dim=-1)  # (B, T, U, hidden_dim//2 + hidden_dim)

        # Final output layer
        logits = self.output_layer(final_features)  # (B, T, U, vocab_size)

        # Apply short sentence boost (learnable weights)
        logits = logits * self.short_sentence_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return logits