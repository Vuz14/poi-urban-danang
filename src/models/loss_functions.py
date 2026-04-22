import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Hàm InfoNCE Loss cơ bản (Dùng cho Baseline V1, V2, V3 ban đầu)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        # Tính Cosine Similarity
        pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True)
        neg_sim = torch.matmul(anchor, negatives.transpose(1, 2)).squeeze(1)

        # Gộp lại: [pos_sim, neg_sim_1, neg_sim_2, ...]
        logits = torch.cat([pos_sim, neg_sim], dim=-1) / self.temperature

        # Label luôn là 0 (vì positive sample nằm ở index 0)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)
        return loss

# Loss nâng cấp với Margin để ép cụm t-SNE tách xa nhau hơn, giúp tăng khả năng phân biệt giữa các cụm
class MarginInfoNCELoss(nn.Module):
    """
    Hàm InfoNCE Loss có Margin - Khóa Nhiệt độ cố định để tránh AI "ăn gian"
    """
    def __init__(self, temperature=0.1, margin=0.05): 
        super().__init__()
        # Trả lại giá trị cố định, KHÔNG dùng nn.Parameter nữa
        self.temperature = temperature
        self.margin = margin

    def forward(self, anchor, positive, negatives):
        if anchor.dim() == 1: anchor = anchor.unsqueeze(0)
        if positive.dim() == 1: positive = positive.unsqueeze(0)
        if negatives.dim() == 1: negatives = negatives.unsqueeze(0)

        pos_sim = F.cosine_similarity(anchor, positive, dim=-1).unsqueeze(-1)
        neg_sim = F.cosine_similarity(anchor.unsqueeze(1), negatives.unsqueeze(0), dim=-1)
        
        pos_sim_margin = pos_sim - self.margin
        
        # Chỉ cần chia trực tiếp cho self.temperature
        logits = torch.cat([pos_sim_margin, neg_sim], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
        
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