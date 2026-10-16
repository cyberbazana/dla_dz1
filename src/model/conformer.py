import math
import torch
from torch import nn


class ConvSubsampling(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, stride=2),
            nn.ReLU(),
        )
        new_freq = self.calc_new_length(self.calc_new_length(input_dim))
        self.proj = nn.Linear(d_model * new_freq, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def calc_new_length(length):
        return (length - 1) // 2 # для кернел 3
    
    def transform_input_lengths(self, input_lengths):
        return self.calc_new_length(self.calc_new_length(input_lengths))

    def forward(self, x, lengths):
        x = x.transpose(1, 2).unsqueeze(1)  # [B, 1, T, F] для конв канал
        x = self.conv(x)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f) # для проекции частоты
        x = self.dropout(self.proj(x))
        lengths = self.transform_input_lengths(lengths)
        return x, lengths


class SinusoidEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        inv_freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        t = x.size(1)
        positions = torch.arange(t - 1, -t, -1, device=x.device, dtype=x.dtype)
        sinusoid = positions[:, None] * self.inv_freq.to(dtype=x.dtype)[None, :]
        return torch.cat((sinusoid.sin(), sinusoid.cos()), dim=-1)


class MHSA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.u = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.v_small = nn.Parameter(torch.zeros(num_heads, self.head_dim))

    def forward(self, x, pos, padding_mask):
        b, t, d = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2) #B H T dh
        k = k.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        pos = self.pos_proj(pos)
        pos = pos.reshape(2 * t - 1, self.num_heads, self.head_dim) #2T-1 H dh
        
        # u -> 1 H 1 dh
        # k -> B H dh T
        content_score = torch.matmul(q + self.u[None, :, None, :], k.transpose(2, 3))
        # B H T T
        
        # H Dh 2T-1 pos need
        position_score_raw = torch.matmul(q + self.v_small[None, :, None, :], pos.permute(1, 0, 2).transpose(1, 2))
        
        # B H T 2T-1 res
        i = torch.arange(t, device=x.device)[:, None]
        j = torch.arange(t, device=x.device)[None, :]
        
        relative_index = (t - 1) - (i - j)
        relative_index = relative_index.expand(b, self.num_heads, t, t)
        position_score = position_score_raw.gather(-1, relative_index)
        # B H T T
        score = (content_score + position_score) * self.scale
        score = score.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        res = torch.softmax(score, dim=-1)
        res = self.dropout(res)
        x = torch.matmul(res, v)
        x = x.transpose(1, 2).reshape(b, t, d)
        x = self.out_proj(x)
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1),
            nn.GLU(dim=1),
            nn.ConstantPad1d(((kernel_size - 1) // 2, kernel_size // 2), 0.0),
            nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=False),
            nn.BatchNorm1d(d_model), # авось падинги несильно помешают
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask):
        x = self.norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, expansion=4, conv_kernel_size=32, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff1 = FeedForward(d_model, expansion, dropout)
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = MHSA(d_model, num_heads, dropout)
        self.mhsa_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForward(d_model, expansion, dropout)
        

    def forward(self, x, pos, padding_mask):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa_dropout(self.mhsa(self.mhsa_norm(x), pos, padding_mask))
        
        x = x + self.conv(x, padding_mask)
        x = self.norm(x + 0.5 * self.ff2(x))
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, d_model=144, num_layers=16, num_heads=4, expansion=4, conv_kernel_size=32, dropout=0.1):
        super().__init__()
        self.output_dim = d_model
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, expansion, conv_kernel_size,dropout)
            for i in range(num_layers)
        ])
        self.positional_encoding = SinusoidEncoding(d_model)
        self.conv_sub = ConvSubsampling(input_dim, d_model)
        

    def forward(self, spectrogram, spectrogram_length, **kwargs):
        x, lengths = self.conv_sub(spectrogram, spectrogram_length)
        pad_mask = (torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None])
        pos = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, pos, pad_mask)
        return x, lengths
    
    def transform_input_lengths(self, input_lengths):
        return self.conv_sub.transform_input_lengths(input_lengths)