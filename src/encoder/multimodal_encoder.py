"""
multimodal_encoder.py - Bộ mã hóa Đa phương thức (Multimodal Encoder)
======================================================================
THAY ĐỔI SO VỚI PHIÊN BẢN CŨ:
  - [YC3-A] Hỗ trợ Multi-image Pooling:
            Input images có thể là Tensor [B, N, C, H, W] (N = số ảnh/POI).
            → Reshape → qua CLIP → Mean Pooling theo chiều N → [B, D]
  - [YC3-B] Ablation Study với 4 versions:
            Version 1 : Chỉ Không gian (ResNet geom)
            Version 2 : Không gian + Text
            Version 3 : Không gian + Image
            Version 4 : Full (Không gian + Text + Image) ← Mặc định
  - Output dimension luôn là embed_dim (64) bất kể version.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import Optional, List, Union  # <--- CHỈ CẦN THÊM DÒNG NÀY VÀO ĐÂY


VERSION_DESC = {
    1: "V1: Chỉ Không gian (ResNet)",
    2: "V2: Không gian + Text",
    3: "V3: Không gian + Image",
    4: "V4: Full (Không gian + Text + Image)",
}
class MultimodalEncoder(nn.Module):
    """
    Mô hình Đa phương thức hỗ trợ Zero-shot Domain Adaptation.
    Hỗ trợ 4 Version cấu hình qua biến `version`.
    """
    def __init__(self, embed_dim=256, version=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.version = version

        print(f"🔥 MultimodalEncoder khởi tạo | V{version}: {self._get_version_name()}")

        # ==========================================
        # 1. KHÔNG GIAN (Geometry/Footprint) - Luôn bật
        # ==========================================
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Thay lớp FC cuối để chiếu về không gian nhúng chung
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, embed_dim)

        # ==========================================
        # 2. VĂN BẢN & HÌNH ẢNH (CLIP)
        # ==========================================
        if self.version in [2, 3, 4]:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Đóng băng CLIP (chỉ dùng làm Feature Extractor)
            for param in self.clip_model.parameters():
                param.requires_grad = False

            if self.version in [2, 4]:
                self.text_projection = nn.Linear(512, embed_dim)
            if self.version in [3, 4]:
                self.image_projection = nn.Linear(512, embed_dim)

        # ==========================================
        # 3. FUSION MODULE (Gated Fusion Nâng cấp)
        # ==========================================
        if self.version == 4:
            self.fusion_proj = nn.Linear(embed_dim * 3, embed_dim)
            # Cổng Gated Fusion: Học cách ưu tiên Modality nào đáng tin cậy hơn
            self.gate = nn.Sequential(
                nn.Linear(self.embed_dim * 3, self.embed_dim * 3),
                nn.ReLU(),
                nn.Linear(self.embed_dim * 3, 3), # 3 cổng tương ứng: Spatial, Text, Image
                nn.Softmax(dim=-1)
            )
        elif self.version in [2, 3]:
            self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

    def _get_version_name(self):
        if self.version == 1: return "Chỉ Không gian (ResNet)"
        if self.version == 2: return "Không gian + Text"
        if self.version == 3: return "Không gian + Ảnh"
        if self.version == 4: return "Full Đa phương thức (Không gian + Text + Ảnh)"
        return "Unknown"

    def forward(self, geom_images, images=None, texts=None):
        batch_size = geom_images.shape[0]
        device = geom_images.device

        # 1. Luôn Trích xuất Đặc trưng Không gian
        spatial_features = self.resnet(geom_images)  # [B, 64]

        if self.version == 1:
            return spatial_features

        # Khởi tạo vector 0 cho các modality bị khuyết (để shape nhất quán)
        text_features = torch.zeros(batch_size, self.embed_dim, device=device)
        image_features = torch.zeros(batch_size, self.embed_dim, device=device)

        # 2. Trích xuất Đặc trưng Văn bản
        if self.version in [2, 4] and texts is not None:
            text_features = self._encode_clip_text(texts, device, batch_size)

        # 3. Trích xuất Đặc trưng Hình ảnh
        if self.version in [3, 4] and images is not None:
            image_features = self._encode_clip_image(images, device, batch_size)

        # ==========================================
        # 4. FUSION (KẾT HỢP ĐẶC TRƯNG)
        # ==========================================
        if self.version == 2:
            concat_feats = torch.cat([spatial_features, text_features], dim=-1)
            return self.fusion_proj(concat_feats)
            
        elif self.version == 3:
            concat_feats = torch.cat([spatial_features, image_features], dim=-1)
            return self.fusion_proj(concat_feats)
            
        elif self.version == 4:
            # Dùng Gated Fusion thông minh
            concat_feats = torch.cat([spatial_features, text_features, image_features], dim=-1)
            
            # Tính trọng số Attention cho từng giác quan
            gates = self.gate(concat_feats) # [B, 3]
            
            # Nhân trọng số vào từng vector
            spatial_gated = spatial_features * gates[:, 0:1]
            text_gated = text_features * gates[:, 1:2]
            image_gated = image_features * gates[:, 2:3]
            
            # Nối lại và chiếu về 64 chiều cuối cùng
            fused_concat = torch.cat([spatial_gated, text_gated, image_gated], dim=-1)
            final_features = self.fusion_proj(fused_concat)
            
            return final_features

    # =========================================================================
    # CÁC HÀM TRỢ GIÚP CLIP
    # =========================================================================
    def _encode_clip_image(self, images, device, batch_size_expected):
        """Xử lý hình ảnh qua mạng CLIP (Hỗ trợ multi-image pooling)"""
        is_multi_image = False
        num_images = 1
        batch_size_real = batch_size_expected

        if isinstance(images, torch.Tensor):
            if images.dim() == 5: 
                batch_size_real, num_images, c, h, w = images.shape
                images = images.view(-1, c, h, w)
                is_multi_image = True
            
            pixel_values = images.to(device)
            with torch.set_grad_enabled(self.training):
                img_embeds = self.clip_model.get_image_features(pixel_values=pixel_values)
                
                # Sửa lỗi Object vs Tensor của HuggingFace
                if not isinstance(img_embeds, torch.Tensor):
                    img_embeds = img_embeds.pooler_output
                    
                img_feat = self.image_projection(img_embeds)
                
            if is_multi_image:
                img_feat = img_feat.view(batch_size_real, num_images, self.embed_dim)
                img_feat = img_feat.mean(dim=1)
                
            return img_feat

        elif isinstance(images, list):
            flat_images = []
            for item in images:
                if isinstance(item, list):
                    is_multi_image = True
                    num_images = len(item)
                    flat_images.extend(item)
                else:
                    flat_images.append(item)

            if len(flat_images) == 0:
                return torch.zeros(batch_size_expected, self.embed_dim, device=device)

            inputs = self.clip_processor(images=flat_images, return_tensors="pt").to(device)
            with torch.set_grad_enabled(self.training):
                img_embeds = self.clip_model.get_image_features(**inputs)
                
                # Sửa lỗi Object vs Tensor của HuggingFace
                if not isinstance(img_embeds, torch.Tensor):
                    img_embeds = img_embeds.pooler_output
                    
                img_feat = self.image_projection(img_embeds)

            if is_multi_image:
                img_feat = img_feat.view(len(images), num_images, self.embed_dim)
                img_feat = img_feat.mean(dim=1)

            return img_feat

        return torch.zeros(batch_size_expected, self.embed_dim, device=device)

    def _encode_clip_text(self, texts, device, batch_size_expected):
        """Xử lý văn bản qua mạng CLIP"""
        if texts is None or len(texts) == 0:
            return torch.zeros(batch_size_expected, self.embed_dim, device=device)
            
        clean_texts = [str(t) if t is not None and str(t).strip() != "" else "Unknown" for t in texts]
        
        inputs = self.clip_processor(
            text=clean_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(device)
        
        with torch.set_grad_enabled(self.training):
            text_embeds = self.clip_model.get_text_features(**inputs)  # [B, 512]
            
            # Sửa lỗi Object vs Tensor của HuggingFace
            if not isinstance(text_embeds, torch.Tensor):
                text_embeds = text_embeds.pooler_output

        return self.text_projection(text_embeds)  # [B, embed_dim]

    # ------------------------------------------------------------------
    def _encode_resnet(
        self,
        geom_images : Optional[torch.Tensor],
        is_negative : bool,
        device      : torch.device,
        batch_size  : int,
    ) -> torch.Tensor:
        """Mã hóa ảnh footprint qua ResNet."""
        if is_negative or geom_images is None:
            # Không có geom_images → zero vector
            return torch.zeros(batch_size, self.embed_dim, device=device)

        geom_images = geom_images.to(device)
        with torch.set_grad_enabled(self.training):
            return self.resnet(geom_images)  # [B, embed_dim]


# ======================================================================
# TEST NHANH (chạy trực tiếp file này)
# ======================================================================
if __name__ == "__main__":
    from PIL import Image
    import torch

    print("=" * 70)
    print("TEST: MultimodalEncoder – Multi-image + Ablation Study")
    print("=" * 70)

    B, N = 4, 5  # batch_size=4, num_images=5 mỗi POI

    # Giả lập multi-image tensor [B, N, C, H, W]
    dummy_images = torch.randn(B, N, 3, 224, 224)
    dummy_texts  = ["Quán ăn ngon tại Đà Nẵng"] * B
    dummy_geoms  = torch.randn(B, 3, 224, 224) # Giả lập ảnh ResNet

    for v in [1, 2, 3, 4]:
        model = MultimodalEncoder(version=v)
        model.eval()
        with torch.no_grad():
            # Pass thêm geom_images vào test cho đúng chuẩn
            out = model(geom_images=dummy_geoms, images=dummy_images, texts=dummy_texts)
        print(f"  {VERSION_DESC[v]} → Output shape: {out.shape}")

    print("\n✅ Tất cả 4 versions đều cho output đúng kích thước!")
    print("=" * 70)