import os

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import yaml
from torchvision import models
from torchvision.transforms.functional import to_pil_image
from typing import Optional, List, Union
# Thêm vào đầu file multimodal_encoder.py
print("🔥 Đã nạp thành công MultimodalEncoder từ file encoder!")

class MultimodalEncoder(nn.Module):
    """Bộ mã hóa Đa phương thức (Multimodal Encoder).

    Mục tiêu:
    - CLIP để mã hóa: Ảnh POI + Văn bản Review
    - ResNet để mã hóa: Ảnh hình học tòa nhà (footprint)
    - Fusion: Linear Projection để ghép các embedding về cùng một không gian.

    Yêu cầu:
    - Không tạo lỗi lệch chiều (Dimension Mismatch) khi Forward.
    - Hỗ trợ Dynamic Negative Sampling từ Urban Voids
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()

        # ----- Cấu hình mặc định (nếu không có file config) -----
        default_config = {
            "model": {
                "embed_dim": 64,
                "clip_model": "openai/clip-vit-base-patch32",
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

        embed_dim = config["model"]["embed_dim"]
        clip_model_name = config["model"]["clip_model"]
        resnet_model_name = config["model"].get("resnet_model", "resnet50")

        # ----- CLIP encoder (Ảnh POI + Text Review) -----
        self.model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # CLIP image/text embedding thường có kích thước 512
        self.projection = nn.Linear(self.model.config.projection_dim, embed_dim)

        # ----- ResNet encoder (ảnh footprint) -----
        self.resnet = getattr(models, resnet_model_name)(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)

        # ----- Fusion (ghép 2 embedding) -----
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        images: Optional[Union[torch.Tensor, List]] = None,
        texts: Optional[List[str]] = None,
        geom_images: Optional[torch.Tensor] = None,
        is_negative: bool = False,
    ):
        """Forward pass cho Multimodal Encoder - Hỗ trợ POI thực + Negative Samples.

        Args:
            images: Tensor hoặc List[PIL.Image] (ảnh POI) cho CLIP.
            texts: List[str] (review / mô tả) cho CLIP.
            geom_images: Tensor (ảnh footprint) cho ResNet.
            is_negative: bool - True nếu là Negative Sample (sẽ dùng Dummy Data).

        Returns:
            torch.Tensor với shape (batch, 64)
        """
        device = next(self.parameters()).device
        batch_size = self._infer_batch_size(images, texts, geom_images)

        # CLIP encoding
        clip_feat = self._encode_clip(images, texts, is_negative, device, batch_size)

        # ResNet encoding
        geom_feat = self._encode_resnet(geom_images, is_negative, device, batch_size)

        # Fusion
        return self._fuse_embeddings(clip_feat, geom_feat)

    def _infer_batch_size(
        self,
        images: Optional[Union[torch.Tensor, List]],
        texts: Optional[List[str]],
        geom_images: Optional[torch.Tensor],
    ) -> int:
        """Suy ra batch size từ input."""
        if isinstance(images, torch.Tensor):
            return images.shape[0]
        elif isinstance(images, list):
            return len(images)
        elif isinstance(texts, list):
            return len(texts)
        elif isinstance(geom_images, torch.Tensor):
            return geom_images.shape[0]
        else:
            return 1

    def _encode_clip(
        self,
        images: Optional[Union[torch.Tensor, List]],
        texts: Optional[List[str]],
        is_negative: bool,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Mã hóa ảnh + text bằng CLIP model."""
        
        # Xử lý Negative Sample: tự động gán Dummy Data
        if is_negative:
            images = torch.zeros(batch_size, 3, 224, 224, device=device)
            texts = [""] * batch_size

        # Nếu vẫn không có ảnh/text, tạo Dummy
        if images is None:
            images = torch.zeros(batch_size, 3, 224, 224, device=device)
        if texts is None or len(texts) == 0:
            texts = [""] * batch_size

        # Nếu input images là Tensor đã normalize (ResNet style), unnormalize trước khi đưa vào CLIP processor.
        if isinstance(images, torch.Tensor):
            # Chuyển về CPU để xử lý image_proc, tránh mất GPU
            images = images.detach().cpu()
            # Unnormalize từ mean/std ImageNet thành [0,1]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = images * std + mean
            images = images.clamp(0.0, 1.0)
            images = [to_pil_image(img) for img in images]

        # Processor
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # CLIP Forward
        with torch.set_grad_enabled(self.training):
            outputs = self.model(**inputs)

        # Merge embeddings
        if images is not None and texts is not None:
            clip_emb = (outputs.image_embeds + outputs.text_embeds) / 2.0
        elif images is not None:
            clip_emb = outputs.image_embeds
        else:
            clip_emb = outputs.text_embeds

        # Project
        clip_feat = self.projection(clip_emb)
        return clip_feat

    def _encode_resnet(
        self,
        geom_images: Optional[torch.Tensor],
        is_negative: bool,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Mã hóa ảnh footprint bằng ResNet."""
        
        # Xử lý Negative Sample
        if is_negative:
            geom_images = torch.zeros(batch_size, 3, 224, 224, device=device)

        # Nếu không có ảnh, tạo Dummy
        if geom_images is None:
            geom_images = torch.zeros(batch_size, 3, 224, 224, device=device)

        geom_images = geom_images.to(device)

        # ResNet Forward
        with torch.set_grad_enabled(self.training):
            geom_feat = self.resnet(geom_images)

        return geom_feat

    def _fuse_embeddings(
        self,
        clip_feat: Optional[torch.Tensor],
        geom_feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ghép nối 2 embeddings từ CLIP và ResNet."""
        
        if clip_feat is not None and geom_feat is not None:
            fused = torch.cat([clip_feat, geom_feat], dim=-1)
            return self.fusion(fused)
        elif clip_feat is not None:
            return clip_feat
        elif geom_feat is not None:
            return geom_feat
        else:
            raise ValueError("❌ Cần cung cấp ít nhất một trong: images/texts hoặc geom_images")

# =====================================================================
# PHẦN TEST (CHẠY KHI GỌI TRỰC TIẾP FILE NÀY)
# =====================================================================
if __name__ == "__main__":
    import os
    import sys
    
    # ÉP PYTHON TÌM TRONG THƯ MỤC 'src' ĐẦU TIÊN (ƯU TIÊN SỐ 1)
    sys.path.insert(0, r"D:\poi-urban-danang\src")

    # BÂY GIỜ NÓ SẼ TÌM ĐÚNG FILE dataset.py TRONG SRC
    from data.dataset import POIDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image

    print("=" * 80)
    print("TEST TÍCH HỢP: MULTIMODAL ENCODER + DATASET.PY (BỎ QUA TẢI ẢNH)")
    print("=" * 80)
    
    model = MultimodalEncoder()
    model.eval()

    # Transform cơ bản để chuyển ảnh thành Tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Khởi tạo DataLoader từ file CSV
    csv_path = r"D:\poi-urban-danang\dataset\processed\poi_processed_data.csv"
    
    if os.path.exists(csv_path):
        dataset_obj = POIDataset(csv_file=csv_path, image_transform=None)
        
        # 💡 MẸO: Bypass ảnh để test text cho nhanh
        dataset_obj._download_image = lambda url: Image.new('RGB', (224, 224), (0, 0, 0))
        
        
        def collate_fn(batch):
            return {
                'poi_id': [item['poi_id'] for item in batch],
                'district': [item['district'] for item in batch],
                'coords': torch.stack([item['coords'] for item in batch]),
                'text': [item['text'] for item in batch],
                'image': [item['image'] for item in batch],  # 👈 giữ list PIL
            }

        dataloader = DataLoader(dataset_obj, batch_size=4, shuffle=True, collate_fn=collate_fn) 
                # Rút thử 1 Batch (4 POI) ra để test
        batch = next(iter(dataloader))
        
        print("\n[TEST] Forward Pass với dữ liệu thực từ CSV:")
        print(f"🆔 POI IDs  : {batch['poi_id']}")
        print(f"📝 Texts    : {batch['text'][0][:60]}...")
        print(f"📍 Coords   : {batch['coords'].shape}")
        print(f"🖼️ Images   : {len(batch['image'])} images")
        # Chạy qua Multimodal Encoder
        with torch.no_grad():
            out = model(
                images=batch['image'], 
                texts=batch['text'], 
                is_negative=False
            )
            
        print(f"\n✅ Output model shape: {out.shape} (Dự kiến: torch.Size([4, 64]))")
        print("🎉 TUYỆT VỜI! Dataset.py và MultimodalEncoder.py đã kết nối thành công!")
        print("=" * 80)
    else:
        print(f"❌ Không tìm thấy file CSV tại: {csv_path}")