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
        # 1. BẢO VỆ CHIỀU (Nếu tensor bị rớt xuống 1D, tự động nâng lên 2D)
        if anchor_group.dim() == 1:
            anchor_group = anchor_group.unsqueeze(0)
        if positive_group.dim() == 1:
            positive_group = positive_group.unsqueeze(0)
        if negative_groups.dim() == 1:
            negative_groups = negative_groups.unsqueeze(0)

        # 2. TÍNH TƯƠNG ĐỒNG BẰNG CHIỀU CUỐI CÙNG (dim=-1) ĐỂ KHÔNG BAO GIỜ LỖI
        sim_pos = F.cosine_similarity(anchor_group, positive_group, dim=-1).unsqueeze(-1) / self.tau
        sim_neg = F.cosine_similarity(anchor_group.unsqueeze(1), negative_groups.unsqueeze(0), dim=-1) / self.tau
        
        # 3. TÍNH LOSS
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
        wasserstein_dist: W(u_a, u_n) tính bằng Optimal Transport
        """
        dist_pos = F.pairwise_distance(z_a, z_p, p=2)
        dist_neg = F.pairwise_distance(z_a, z_n, p=2)
        
        # Ký hiệu margin thích ứng với khoảng cách phân phối (Wasserstein)
        adaptive_margin = self.lamba * wasserstein_dist
        
        loss = torch.relu(dist_pos - dist_neg + adaptive_margin)
        return loss.mean()