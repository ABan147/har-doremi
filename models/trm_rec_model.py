#!/usr/bin/env python3
"""
Transformer-based Reconstruction Model (TRMRec)

A transformer-based masked autoencoder for time series reconstruction
with domain adaptation capabilities.
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TRMRec(nn.Module):
    """Transformer-based Reconstruction Model for time series data."""
    
    def __init__(self, n_channels, n_steps, kernel_size=16, stride=4, num_domains=3):
        super(TRMRec, self).__init__()
        self.hidden_dim = 128

        # Input embedding layer
        self.patch_embedding = nn.Conv1d(
            n_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.hidden_dim, max_seq_len=n_steps)

        # Domain embedding
        self.domain_embedding = nn.Embedding(num_domains, self.hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=16,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        # Decoder with SwiGLU and RMSNorm
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 256 * 2),
            SwiGLU(256 * 2, 256),
            RMSNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128 * 2),
            SwiGLU(128 * 2, 128),
            RMSNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, n_channels)
        )

    def forward(self, x_in, domain_ids, mask):
        """
        Forward pass of the TRMRec model.
        
        Args:
            x_in: Input tensor, shape (batch_size, seq_len, n_channels), already masked
            domain_ids: Domain IDs, shape (batch_size,)
            mask: Mask tensor, shape (batch_size, seq_len, n_channels), 
                  0 indicates masked positions, 1 indicates unmasked
        
        Returns:
            reconstruction: Reconstruction result, shape (batch_size, seq_len, n_channels),
                           masked parts are reconstructed, unmasked parts keep original input values
        """
        device = next(self.parameters()).device
        x_in = x_in.to(device)
        domain_ids = domain_ids.to(device)
        mask = mask.to(device)

        # Input reshape [batch, seq_len, channels] -> [batch, channels, seq_len]
        x = x_in.permute(0, 2, 1)

        # Patch embedding [batch, channels, seq_len] -> [batch, hidden_dim, seq_len]
        x = self.patch_embedding(x)

        # Dimension adjustment [batch, seq_len, hidden_dim]
        x = x.permute(0, 2, 1)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Add domain embedding [batch, 1, hidden_dim]
        domain_emb = self.domain_embedding(domain_ids).unsqueeze(1)
        x += domain_emb

        # Transformer encoding [batch, seq_len, hidden_dim]
        x = self.encoder(x)

        # Decoder reconstruction for entire sequence [batch, seq_len, n_channels]
        reconstruction_full = self.decoder(x)

        # Combine output:
        # - Masked parts (mask == 0) use reconstruction values
        # - Unmasked parts (mask == 1) use original input values
        reconstruction = reconstruction_full * (1 - mask) + x_in * mask

        return reconstruction


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, hidden_size, max_seq_len=120):
        super(PositionalEncoding, self).__init__()
        self.pos_enc = self.positional_encoding(hidden_size, max_seq_len)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1)].cuda(x.device)
        return x

    def positional_encoding(self, hidden_size, max_seq_len):
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pos_enc = torch.zeros(max_seq_len, hidden_size)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out * 2)  # Expand dimension
        self.w2 = nn.Linear(dim_out, dim_out)     # Output transformation

    def forward(self, x):
        x = self.w1(x)          # [batch, steps, dim_out*2]
        x1, x2 = x.chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)  # Use SiLU activation


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x