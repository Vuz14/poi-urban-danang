import torch
import torch.nn as nn
from torchvision import models

class BuildingGeometryEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        # Sử dụng ResNet-50
        self.resnet = models.resnet50(pretrained=True)
        # Sửa lớp Linear cuối: trừ 3 vì ta sẽ concat thêm 3 features (size, cos_alpha, sin_alpha)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim - 3)
        
    def forward(self, image_tensor, size_and_rotation):
        """
        image_tensor: shape (Batch, 3, 224, 224) - Ảnh đa giác tòa nhà
        size_and_rotation: shape (Batch, 3) - Chứa [size, cos(alpha), sin(alpha)]
        """
        # Trích xuất đặc trưng hình ảnh
        visual_features = self.resnet(image_tensor)
        
        # Nối vector hình ảnh với 3 giá trị scalar (kích thước, cos, sin)
        # Kết quả sẽ có số chiều = embed_dim
        final_embedding = torch.cat([visual_features, size_and_rotation], dim=-1)
        return final_embedding