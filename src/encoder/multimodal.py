import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import yaml

class MultimodalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        embed_dim = config['model']['embed_dim']
        clip_model_name = config['model']['clip_model']
        
        # Tải mô hình CLIP
        self.model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Lớp Linear để map vector 512 chiều của CLIP về 64 chiều (embed_dim)
        self.projection = nn.Linear(512, embed_dim)
        
    def forward(self, images=None, texts=None):
        """
        Xử lý đồng thời cả ảnh và Text (hoặc một trong hai)
        """
        # Tránh lỗi khi chạy trên GPU
        device = next(self.model.parameters()).device
        
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        # Chiến lược Fusion: Nếu có cả 2, lấy trung bình
        if images is not None and texts is not None:
            combined_feat = (outputs.image_embeds + outputs.text_embeds) / 2
        elif images is not None:
            combined_feat = outputs.image_embeds
        else:
            combined_feat = outputs.text_embeds
            
        # Chiếu về kích thước chung
        return self.projection(combined_feat)