import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder.position import DistanceBias

class DistanceBiasedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dist_bias = DistanceBias()
        
    def forward(self, x, dist_matrix):
        # x shape: (Batch, Seq_len, Embed_Dim)
        B, N, C = x.size()
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Tính QK^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Tính khoảng cách và thêm vào scores
        bias = self.dist_bias(dist_matrix) # Shape: (N, N)
        bias = bias.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, N, N) để broadcast
        scores = scores + bias
        
        # Softmax và nhân với V
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)

class BuildingGroupEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8):
        super().__init__()
        self.dist_attn = DistanceBiasedSelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, features, dist_matrix):
        # features: (1, N, 64) bao gồm Buildings, POIs, Random points
        # dist_matrix: (N, N)
        
        # Biased Transformer Block
        attn_out = self.dist_attn(features, dist_matrix)
        x = self.norm1(features + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Average Pooling để ra Vector Group
        group_embedding = x.mean(dim=1) # Shape: (1, 64)
        return group_embedding