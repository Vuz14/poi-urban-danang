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

import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import yaml
from torchvision import models
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPModel, CLIPProcessor


# Ánh xạ version → mô tả để in log
VERSION_DESC = {
    1: "V1: Chỉ Không gian (ResNet)",
    2: "V2: Không gian + Text",
    3: "V3: Không gian + Image",
    4: "V4: Full (Không gian + Text + Image)",
}


class MultimodalEncoder(nn.Module):
    """Bộ mã hóa Đa phương thức.

    Kiến trúc:
      - CLIP  : Mã hóa ảnh POI (hỗ trợ multi-image) + Văn bản Review
      - ResNet: Mã hóa ảnh hình học tòa nhà (footprint / geom_images)
      - Fusion: Linear Projection → ghép các embedding về cùng không gian embed_dim

    Ablation Study:
      version=1 → Chỉ ResNet (geom)
      version=2 → ResNet + Text CLIP
      version=3 → ResNet + Image CLIP
      version=4 → ResNet + Text + Image CLIP (Full, mặc định)
    """

    def __init__(self, config_path: str = "config.yaml", version: int = 4):
        """
        Tham số:
            config_path : Đường dẫn file YAML cấu hình (tùy chọn).
            version     : Phiên bản Ablation Study (1–4). Mặc định = 4 (Full).
        """
        super().__init__()

        # [YC3-B] Kiểm tra version hợp lệ
        assert version in (1, 2, 3, 4), "version phải thuộc {1, 2, 3, 4}"
        self.version = version
        print(f"🔥 MultimodalEncoder khởi tạo | {VERSION_DESC[version]}")

        # ---- Cấu hình mặc định -------------------------------------------
        default_config = {
            "model": {
                "embed_dim"   : 64,
                "clip_model"  : "openai/clip-vit-base-patch32",
                "resnet_model": "resnet50",
            }
        }

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            config = {**default_config, **(cfg or {})}
            config["model"] = {**default_config["model"], **config.get("model", {})}
        else:
            config = default_config

        embed_dim        = config["model"]["embed_dim"]
        clip_model_name  = config["model"]["clip_model"]
        resnet_model_name = config["model"].get("resnet_model", "resnet50")
        self.embed_dim   = embed_dim

        # ---- CLIP (Text + Image) ------------------------------------------
        self.clip_model  = CLIPModel.from_pretrained(clip_model_name)
        self.processor   = CLIPProcessor.from_pretrained(clip_model_name)
        clip_proj_dim    = self.clip_model.config.projection_dim  # thường = 512

        # Projection riêng cho từng modality (Text và Image)
        self.text_projection  = nn.Linear(clip_proj_dim, embed_dim)
        self.image_projection = nn.Linear(clip_proj_dim, embed_dim)

        # ---- ResNet (Geom / Footprint) ------------------------------------
        self.resnet     = getattr(models, resnet_model_name)(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.resnet.fc  = nn.Linear(self.resnet.fc.in_features, embed_dim)

        # ---- Fusion layers (luôn nhận 3 * embed_dim → embed_dim) ----------
        # [YC3-B] Các version thiếu modality sẽ dùng zero vector → đầu vào
        # luôn có đúng kích thước 3*embed_dim, tránh Dimension Mismatch.
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    # ======================================================================
    # FORWARD
    # ======================================================================
    def forward(self, geom_images=None, images=None, texts=None):
            """
            Forward Pass (Tối ưu cho từng Version)
            """
            # Sử dụng đúng hàm suy luận batch_size của bạn
            batch_size = self._infer_batch_size(images, texts, geom_images)
            device = next(self.parameters()).device
            
            # 1. Trích xuất Không gian (ResNet) - Luôn bật ở cả 4 Version
            # Gọi thẳng xuống hàm _encode_resnet gốc của bạn ở bên dưới
            spatial_features = self._encode_resnet(
                geom_images=geom_images, is_negative=False, device=device, batch_size=batch_size
            )

            # -----------------------------------------------------------
            # [TỐI ƯU V1]: Nếu là mode 1, chỉ dùng ResNet -> Trả về luôn!
            # Bỏ qua hoàn toàn logic CLIP và lớp Fusion phía sau.
            # -----------------------------------------------------------
            if self.version == 1:
                return spatial_features

            # 2. Trích xuất Text & Image (CLIP) cho các Version 2, 3, 4
            text_features = torch.zeros(batch_size, self.embed_dim, device=device)
            image_features = torch.zeros(batch_size, self.embed_dim, device=device)

            if self.version in [2, 4]:
                # Gọi đúng tên hàm _encode_clip_text
                text_features = self._encode_clip_text(
                    texts=texts, is_negative=False, device=device, batch_size=batch_size
                )

            if self.version in [3, 4]:
                # Gọi đúng tên hàm _encode_clip_image
                image_features = self._encode_clip_image(
                    images=images, is_negative=False, device=device, batch_size=batch_size
                )

            # 3. Fusion (Ghép nối) dành cho Mode 2, 3, 4
            # Spatial (64) + Text (64) + Image (64) = 192 -> Linear -> 64
            combined = torch.cat([spatial_features, text_features, image_features], dim=1)
            final_embedding = self.fusion(combined)

            return final_embedding

    # ======================================================================
    # HÀM PHỤ TRỢ NỘI BỘ
    # ======================================================================
    def _infer_batch_size(self, images, texts, geom_images) -> int:
        """Suy ra batch_size từ tham số đầu vào."""
        if isinstance(images, torch.Tensor):
            return images.shape[0]
        elif isinstance(images, list) and len(images) > 0:
            return len(images)
        elif isinstance(texts, list):
            return len(texts)
        elif isinstance(geom_images, torch.Tensor):
            return geom_images.shape[0]
        return 1

    # ------------------------------------------------------------------
    # [YC3-A] Encode ảnh qua CLIP với Multi-image Pooling
    # ------------------------------------------------------------------
    def _encode_clip_image(
        self,
        images    : Optional[Union[torch.Tensor, List]],
        is_negative: bool,
        device    : torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Mã hóa ảnh qua CLIP và áp dụng Mean Pooling nếu multi-image.

        Hỗ trợ 3 dạng đầu vào:
          (a) Tensor [B, C, H, W]       → xử lý bình thường (1 ảnh/POI)
          (b) Tensor [B, N, C, H, W]    → reshape → xử lý N*B ảnh → mean pool theo N
          (c) List[PIL.Image] độ dài B  → xử lý bình thường (1 ảnh/POI)
        """
        if is_negative or images is None:
            # Negative sample hoặc không có ảnh → zero vector
            return torch.zeros(batch_size, self.embed_dim, device=device)

        is_multi_image = False  # Flag đánh dấu đầu vào multi-image
        num_images     = 1

        # ---- Phát hiện và xử lý Tensor 5D [B, N, C, H, W] ---------------
        if isinstance(images, torch.Tensor) and images.dim() == 5:
            # [YC3-A] Multi-image: reshape [B, N, C, H, W] → [B*N, C, H, W]
            B, N, C, H, W  = images.shape
            is_multi_image = True
            num_images     = N
            batch_size_real = B
            images = images.view(B * N, C, H, W)  # Gộp batch và num_images lại

# ---- [ĐÃ TỐI ƯU SIÊU TỐC] Truyền thẳng Tensor vào mạng CLIP ----
        if isinstance(images, torch.Tensor):
            # Tensor đã được Dataloader xử lý (Crop 224, Normalize), chỉ cần đẩy lên GPU
            pixel_values = images.to(device)
            with torch.set_grad_enabled(self.training):
                img_embeds = self.clip_model.get_image_features(pixel_values=pixel_values) # [B*N, 512]
        else:
            # Fallback an toàn nếu đầu vào vô tình là List[PIL.Image]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.set_grad_enabled(self.training):
                img_embeds = self.clip_model.get_image_features(**inputs)

        # Projection về embed_dim
        img_feat = self.image_projection(img_embeds)  # [B*N, embed_dim]

        # ---- [YC3-A] Mean Pooling nếu multi-image ------------------------
        if is_multi_image:
            # Reshape về [B, N, embed_dim] rồi lấy trung bình theo N
            img_feat = img_feat.view(batch_size_real, num_images, self.embed_dim)
            img_feat = img_feat.mean(dim=1)  # [B, embed_dim]

        return img_feat  # [B, embed_dim]

    # ------------------------------------------------------------------
    def _encode_clip_text(
        self,
        texts     : Optional[List[str]],
        is_negative: bool,
        device    : torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Mã hóa văn bản qua CLIP Text Encoder."""
        # Chống lỗi text rỗng do Void truyền vào
        valid_texts = [t if t and str(t).strip() else "empty void space" for t in texts] if texts else None

        if is_negative or valid_texts is None or len(valid_texts) == 0:
            return torch.zeros(batch_size, self.embed_dim, device=device)

        inputs = self.processor(
            text=valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.set_grad_enabled(self.training):
            text_embeds = self.clip_model.get_text_features(**inputs)  # [B, 512]

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