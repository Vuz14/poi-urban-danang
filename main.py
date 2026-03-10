import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

# Import các file từ hệ thống của bạn
from src.data.dataset import POIDataset
from src.encoder.multimodal import MultimodalEncoder  # Căn chỉnh theo máy bạn
from utlis.geo_utils import haversine_matrix_torch # Căn chỉnh theo máy bạn
from src.models.building_group import BuildingGroupEncoder
from src.models.loss_functions import InfoNCELoss

# =========================================================================
# 1. HÀM HUẤN LUYỆN AI (TRAINING)
# =========================================================================
def train_urban_ai():
    print("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MÔ HÌNH (TRAINING)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Đang chạy trên: {device.upper()}")

    # 1. Chuẩn bị Dữ liệu
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = POIDataset(csv_file="dataset/processed/poi_processed_data.csv", image_transform=transform)
    
    # Giữ nguyên batch_size=32 để chia thành 4 nhóm
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # 2. Khởi tạo Models
    multimodal_encoder = MultimodalEncoder().to(device)
    group_encoder = BuildingGroupEncoder(embed_dim=64, num_heads=4).to(device)

    for param in multimodal_encoder.model.parameters():
        param.requires_grad = False
    for param in multimodal_encoder.projection.parameters():
        param.requires_grad = True

    # 3. Tối ưu hóa và Hàm Loss
    optimizer = optim.Adam([
        {'params': multimodal_encoder.projection.parameters(), 'lr': 1e-4},
        {'params': group_encoder.parameters(), 'lr': 1e-4}
    ])
    
    # BÍ KÍP 1: Tăng Temperature lên 0.5 để Softmax không bị bão hòa
    criterion = InfoNCELoss(temperature=0.5).to(device)

    # 4. BẮT ĐẦU VÒNG LẶP
    num_epochs = 10 
    loss_history = [] 
    
    multimodal_encoder.train()
    group_encoder.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad() 

            coords = batch['coords'].to(device)
            images = batch['image'].to(device)
            texts = batch['text']

            poi_features = multimodal_encoder(images=images, texts=texts)

            # BÍ KÍP 2: CHIA BATCH THÀNH 4 NHÓM (Mỗi nhóm 8 POI)
            group_size = 8
            
            # NHÓM 1: LÀM ANCHOR
            anchor_feats = poi_features[:group_size]
            anchor_coords = coords[:group_size]
            dist_matrix_anchor = haversine_matrix_torch(anchor_coords)
            anchor_seq = anchor_feats.unsqueeze(0)
            anchor_group = group_encoder(anchor_seq, dist_matrix_anchor).mean(dim=1) 

            # NHÓM 2: LÀM POSITIVE (BÍ KÍP 3: Tăng nhiễu lên 0.15 để làm khó AI)
            noise = torch.randn_like(anchor_feats) * 0.15
            positive_seq = (anchor_feats + noise).unsqueeze(0)
            positive_group = group_encoder(positive_seq, dist_matrix_anchor).mean(dim=1)

            # NHÓM 3: LÀM NEGATIVES (Lấy 3 nhóm còn lại trong Batch)
            neg_groups_list = []
            for i in range(1, 4): # Chạy từ 1 đến 3
                neg_feats = poi_features[i*group_size : (i+1)*group_size]
                neg_coords = coords[i*group_size : (i+1)*group_size]
                dist_neg = haversine_matrix_torch(neg_coords)
                neg_seq = neg_feats.unsqueeze(0)
                neg_group = group_encoder(neg_seq, dist_neg).mean(dim=1)
                neg_groups_list.append(neg_group)
                
            # Ép 3 tensor Negative lại thành ma trận (3, 64)
            negative_groups = torch.cat(neg_groups_list, dim=0)

            # Hàm Loss bây giờ sẽ ép AI phân biệt 1 Anchor với tận 3 Negatives
            loss = criterion(anchor_group, positive_group, negative_groups)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"🔄 Epoch [{epoch+1}/{num_epochs}] - Mức độ sai số (Loss): {epoch_loss:.4f}")

    print("✅ HOÀN TẤT HUẤN LUYỆN! AI đã hiểu được cấu trúc đô thị Đà Nẵng.")
    
    os.makedirs("reports/metrics", exist_ok=True)
    df_loss = pd.DataFrame({"Epoch": range(1, num_epochs+1), "Loss": loss_history})
    df_loss.to_csv("reports/metrics/training_loss.csv", index=False)

    os.makedirs("models_saved", exist_ok=True)
    torch.save(multimodal_encoder.state_dict(), "models_saved/multimodal_best.pth")
    torch.save(group_encoder.state_dict(), "models_saved/group_encoder_best.pth")
# =========================================================================
# 2. HÀM VẼ BIỂU ĐỒ LOSS (DÀNH CHO BÁO CÁO)
# =========================================================================
def plot_training_loss():
    print("\n📈 Đang vẽ biểu đồ Loss Curve...")
    try:
        # Cấu hình font
        plt.rcParams['font.family'] = 'sans-serif'
        sns.set_theme(style="whitegrid")

        df_loss = pd.read_csv("reports/metrics/training_loss.csv")
        plt.figure(figsize=(8, 5))
        plt.plot(df_loss['Epoch'], df_loss['Loss'], marker='o', color='#d62728', linewidth=2.5)
        
        plt.title("Biểu đồ Suy giảm Sai số (Training Loss) qua các Epoch", fontsize=14, fontweight='bold')
        plt.xlabel("Số vòng học (Epoch)", fontsize=12)
        plt.ylabel("Mức độ sai số (InfoNCE Loss)", fontsize=12)
        plt.xticks(df_loss['Epoch'])
        
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig("reports/figures/training_loss_curve.png", dpi=300, bbox_inches='tight')
        print("✅ Đã lưu ảnh: reports/figures/training_loss_curve.png")
    except Exception as e:
        print(f"Không thể vẽ biểu đồ Loss: {e}")

# =========================================================================
# 3. HÀM PHÂN CỤM T-SNE VÀ VẼ BẢN ĐỒ ĐẶC TRƯNG
# =========================================================================
def plot_tsne_clusters():
    print("\n🧠 Đang trích xuất Đặc trưng để Phân cụm (t-SNE)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Khởi tạo và nạp mô hình vừa học xong
    model = MultimodalEncoder().to(device)
    try:
        model.load_state_dict(torch.load("models_saved/multimodal_best.pth", map_location=device))
        model.eval()
    except:
        print("⚠️ Chưa tìm thấy weights. Sẽ chạy bằng mô hình gốc.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # KHÔNG shuffle để giữ nguyên thứ tự
    dataset = POIDataset(csv_file="dataset/processed/poi_processed_data.csv", image_transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False) 

    all_features = []

    print("🔄 Đang chạy dữ liệu qua Não bộ AI...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text']
            
            features = model(images=images, texts=texts)
            all_features.append(features.cpu())
            
    all_features = torch.cat(all_features, dim=0).numpy()
    df_raw = pd.read_csv("dataset/processed/poi_processed_data.csv")
    
    print("🌌 Đang ép Không gian 64 chiều xuống 2 chiều...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(all_features)
    
    df_raw['tsne_x'] = tsne_results[:, 0]
    df_raw['tsne_y'] = tsne_results[:, 1]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='tsne_x', y='tsne_y',
        hue='Category', # Đổi màu chấm theo từng Danh mục
        palette='tab10',
        data=df_raw,
        legend="full",
        alpha=0.8,
        s=60
    )
    plt.title("Biểu đồ Phân cụm Đặc trưng Đa phương thức (t-SNE)", fontsize=15, fontweight='bold')
    plt.xlabel("Chiều Đặc trưng 1")
    plt.ylabel("Chiều Đặc trưng 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
    
    plt.savefig("reports/figures/tsne_category_clusters.png", dpi=300, bbox_inches='tight')
    print("✅ Đã lưu ảnh: reports/figures/tsne_category_clusters.png")
    print("\n🎉 CHÚC MỪNG BẠN ĐÃ HOÀN THÀNH XUẤT SẮC TOÀN BỘ ĐỒ ÁN!")

# =========================================================================
# CHẠY TỰ ĐỘNG TẤT CẢ
# =========================================================================
if __name__ == "__main__":
    train_urban_ai()
    plot_training_loss()
    plot_tsne_clusters()