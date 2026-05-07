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


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss theo category.
    Dung de giu semantic neighborhood sach: Cafe gan Cafe, Nha hang gan Nha hang.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if features is None or labels is None or features.shape[0] < 2:
            return torch.zeros((), device=features.device if features is not None else "cpu")

        labels = labels.view(-1, 1)
        valid = labels.squeeze(1) >= 0
        features = features[valid]
        labels = labels[valid]
        if features.shape[0] < 2:
            return torch.zeros((), device=features.device)

        features = F.normalize(features, p=2, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
        pos_mask = torch.eq(labels, labels.T) & ~self_mask
        if not pos_mask.any():
            return torch.zeros((), device=features.device)

        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.float().sum(dim=1).clamp_min(1.0)
        valid_anchors = pos_mask.any(dim=1)
        return -mean_log_prob_pos[valid_anchors].mean()


class SemanticAwareContrastiveLoss(nn.Module):
    """
    InfoNCE co margin + hard-negative weighting + SupCon theo danh muc.

    - negative_weights > 1 lam cac negative khac category bi phat nang hon.
    - SupCon tach ro cac danh muc trong batch, tranh spatial proximity keo Cafe gan Quan nhau.
    """
    def __init__(
        self,
        temperature=0.1,
        margin=0.05,
        supcon_weight=0.35,
        hard_negative_margin=0.08,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.supcon_weight = supcon_weight
        self.hard_negative_margin = hard_negative_margin
        self.supcon = SupConLoss(temperature=temperature)

    def forward(
        self,
        anchor,
        positive,
        negatives,
        batch_features=None,
        category_labels=None,
        negative_weights=None,
    ):
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
        if positive.dim() == 1:
            positive = positive.unsqueeze(0)
        if negatives.dim() == 1:
            negatives = negatives.unsqueeze(0)

        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)

        pos_sim = F.cosine_similarity(anchor, positive, dim=-1).unsqueeze(-1) - self.margin
        neg_sim = F.cosine_similarity(anchor.unsqueeze(1), negatives.unsqueeze(0), dim=-1)

        if negative_weights is not None:
            negative_weights = negative_weights.to(device=anchor.device, dtype=neg_sim.dtype).view(1, -1)
            neg_sim = neg_sim + self.hard_negative_margin * (negative_weights - 1.0).clamp_min(0.0)
            neg_logit_bias = torch.log(negative_weights.clamp_min(1e-6))
        else:
            neg_logit_bias = 0.0

        neg_logits = neg_sim / self.temperature + neg_logit_bias
        logits = torch.cat([pos_sim / self.temperature, neg_logits], dim=1)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
        nce_loss = F.cross_entropy(logits, targets)

        if batch_features is not None and category_labels is not None and self.supcon_weight > 0:
            supcon_loss = self.supcon(batch_features, category_labels)
            return nce_loss + self.supcon_weight * supcon_loss

        return nce_loss
    
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
