import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.tau = temperature
        
    def forward(self, anchor_group, positive_group, negative_groups):
        """
        Contrastive Learning ở cấp độ Cụm công trình.
        """
        # Tính tương đồng Cosine
        sim_pos = F.cosine_similarity(anchor_group, positive_group).unsqueeze(-1) / self.tau
        sim_neg = F.cosine_similarity(anchor_group.unsqueeze(1), negative_groups.unsqueeze(0), dim=-1) / self.tau
        
        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor_group.device)
        
        return F.cross_entropy(logits, labels)

class AdaptiveTripletLoss(nn.Module):
    def __init__(self, lamba=50.0):
        super().__init__()
        self.lamba = lamba
        
    def forward(self, z_a, z_p, z_n, wasserstein_dist):
        """
        z_a: Region Anchor
        z_p: Region Positive (vùng lân cận / chồng lấn)
        z_n: Region Negative (vùng ngẫu nhiên xa)
        wasserstein_dist: W(u_a, u_n) tính bằng Optimal Transport giữa các Group Embeddings bên trong
        """
        # L1 Distance
        d_ap = torch.norm(z_a - z_p, p=1)
        d_an = torch.norm(z_a - z_n, p=1)
        
        # Tính Adaptive Margin = lambda * W
        margin = self.lamba * wasserstein_dist
        
        loss = F.relu(d_ap - d_an + margin)
        return loss