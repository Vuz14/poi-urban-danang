import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import sys

# Thêm đường dẫn gốc để import thư mục src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.encoder.multimodal_encoder import MultimodalEncoder
from src.data.dataset import POIDataset, custom_collate_fn

def extract_embeddings(model, dataloader, device):
    all_features = []
    all_labels = []
    model.eval() # BẮT BUỘC ĐỂ ZERO-SHOT KHÔNG BỊ HỎNG BATCHNORM
    
    with torch.no_grad():
        for batch in dataloader:
            poi_data = batch['poi']
            images = poi_data['image'].to(device) if isinstance(poi_data['image'], torch.Tensor) else poi_data['image']
            geom_images = poi_data['geom_image'].to(device)
            texts = poi_data['text']
            
            # Trích xuất 64-dim embeddings
            feats = model(geom_images=geom_images, images=images, texts=texts)
            import torch.nn.functional as F
            feats = F.normalize(feats, p=2, dim=1) # Chuẩn hóa L2
            
            all_features.append(feats.cpu().numpy())
            all_labels.extend(poi_data['category'])
            
    return np.concatenate(all_features, axis=0), np.array(all_labels)

if __name__ == "__main__":
    print("❄️ ĐÓNG BĂNG MÔ HÌNH VÀ TRÍCH XUẤT ĐẶC TRƯNG CHÉO (ZERO-SHOT)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load mô hình ĐÃ TRAIN TỪ GOOGLE MAPS (Giả sử Version 1)
    VERSION = 1
    model = MultimodalEncoder(version=VERSION).to(device)
    model_path = os.path.join(PROJECT_ROOT, f"models_saved/multimodal_best_v{VERSION}.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Đã tải weights thành công từ {model_path}")
    else:
        print(f"⚠️ Chưa có file {model_path}. Hãy chạy python main.py trước!")
        sys.exit()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Extract Google Maps Embeddings
    print("⏳ Đang trích xuất dữ liệu Google Maps...")
    gmap_ds = POIDataset(
        csv_file=os.path.join(PROJECT_ROOT, "dataset/processed/master_nodes_google_maps_clean.csv"),
        image_transform=transform,
        geom_image_dir=os.path.join(PROJECT_ROOT, "dataset/building_images_google_maps")
    )
    gmap_loader = DataLoader(gmap_ds, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    gmap_emb, gmap_labels = extract_embeddings(model, gmap_loader, device)

    # 3. Extract Foody Embeddings (ZERO-SHOT)
    print("⏳ Đang trích xuất dữ liệu Foody (Zero-shot)...")
    foody_ds = POIDataset(
        csv_file=os.path.join(PROJECT_ROOT, "dataset/processed/master_nodes_foody.csv"), # Thêm file Foody nếu đã có
        image_transform=transform,
        geom_image_dir=os.path.join(PROJECT_ROOT, "dataset/building_images_foody")
    )
    foody_loader = DataLoader(foody_ds, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    foody_emb, foody_labels = extract_embeddings(model, foody_loader, device)

    # 4. Lưu lại thành file .npy để các bước sau phân tích
    os.makedirs(os.path.join(PROJECT_ROOT, "reports/embeddings"), exist_ok=True)
    np.save(os.path.join(PROJECT_ROOT, "reports/embeddings/gmap_emb.npy"), gmap_emb)
    np.save(os.path.join(PROJECT_ROOT, "reports/embeddings/gmap_labels.npy"), gmap_labels)
    np.save(os.path.join(PROJECT_ROOT, "reports/embeddings/foody_emb.npy"), foody_emb)
    np.save(os.path.join(PROJECT_ROOT, "reports/embeddings/foody_labels.npy"), foody_labels)
    
    print("📁 Đã trích xuất và lưu Embeddings tại: reports/embeddings/")