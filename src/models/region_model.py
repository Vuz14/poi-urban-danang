import torch
import torch.nn as nn

class RegionEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, num_layers=2):
        super().__init__()
        # Cấp độ vùng dùng Vanilla Transformer (không cần bias khoảng cách nữa)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, group_embeddings):
        # group_embeddings: (Batch, Num_Groups, Embed_Dim)
        region_features = self.transformer(group_embeddings)
        
        # Average Pooling theo paper
        region_repr = region_features.mean(dim=1)
        return region_repr