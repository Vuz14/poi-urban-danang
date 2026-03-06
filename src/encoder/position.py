import torch
import torch.nn as nn

class DistanceBias(nn.Module):
    def __init__(self):
        super().__init__()
        # Tham số có thể huấn luyện lambda
        self.lamda = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, dist_matrix):
        """
        dist_matrix: Ma trận (N, N) khoảng cách
        Công thức: D_tilde_ij = log((1 + max(D)^1.5) / (1 + D_ij^1.5))
        """
        max_d = torch.max(dist_matrix)
        
        # Thêm 1e-8 để tránh chia cho 0 hoặc log(0)
        numerator = 1.0 + torch.pow(max_d, 1.5)
        denominator = 1.0 + torch.pow(dist_matrix, 1.5)
        
        d_tilde = torch.log(numerator / denominator)
        return self.lamda * d_tilde