from torch import nn
import torch
import torch.nn.functional as F
import math

class EnhancedDecoder(nn.Module):
    def __init__(self, vocab_size=1024, embed_dim=768, hidden_dim=768, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Enhanced Embedding with better initialization for short words
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_ln = nn.LayerNorm(embed_dim)

        # LSTM thay vì GRU cho better long-term memory
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Projection layer để stable training
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output_ln = nn.LayerNorm(hidden_dim)

        # Context gate cho short sentence adaptation
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # Initialize weights tốt hơn cho short sentences
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization cho embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        if hasattr(self.embedding, 'padding_idx') and self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)

        # Initialize LSTM weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias = 1 để avoid vanishing gradient
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)

    def init_hidden(self, batch_size, device):
        """Improved hidden state initialization cho short sentences"""
        # Sử dụng learned initialization thay vì random
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

        # Small random initialization cho stability
        h_0 = h_0 + torch.randn_like(h_0) * 0.01
        c_0 = c_0 + torch.randn_like(c_0) * 0.01

        return (h_0, c_0)

    def forward(self, y, hidden_state=None):
        batch_size = y.size(0)

        # Enhanced embedding with scaling và normalization
        y = self.embedding(y) * self.embed_scale  # (B, U, Embed_dim)
        y = self.embed_ln(y)
        y = self.embed_dropout(y)

        # LSTM forward
        if hidden_state is not None:
            y, hidden_state = self.rnn(y, hidden_state)
        else:
            hidden_state = self.init_hidden(batch_size, y.device)
            y, hidden_state = self.rnn(y, hidden_state)  # (B, U, Hidden_dim)

        # Output projection với residual connection
        output = self.output_projection(y)
        output = self.output_dropout(output)
        output = self.output_ln(output)

        # Context gating cho short sentence adaptation
        gate = self.context_gate(output)
        output = output * gate + y * (1 - gate)  # Adaptive mixing

        return output, hidden_state