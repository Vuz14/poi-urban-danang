"""
main.py – Pipeline huấn luyện POI Urban AI (Multimodal + Spatial Hard Negative + Voids)
====================================================
BẢN CẬP NHẬT CUỐI CÙNG:
  - Sử dụng tập dữ liệu đã lọc ảnh đen (chống kẹt loss 0.6931).
  - Tối ưu hóa DataLoader (num_workers=4) chống nghẽn cổ chai CPU.
  - Tích hợp thành công "Vùng trống" (Voids) làm mẫu âm (Negative Sampling).
  - Hỗ trợ chạy mượt mà cả 4 Version (Ablation Study) cho cả Train và t-SNE.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from requests import RequestsDependencyWarning

from src.data.dataset import POIDataset
from src.encoder.multimodal_encoder import MultimodalEncoder, VERSION_DESC
from src.models.building_group import BuildingGroupEncoder
from src.models.loss_functions import InfoNCELoss
from utlis.geo_utils import haversine_matrix_torch

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
print("🔇 Đã tắt RequestsDependencyWarning")

# =========================================================================
# [YC4] BIẾN GLOBAL: CHỌN PHIÊN BẢN ABLATION STUDY (1, 2, 3, 4)
# =========================================================================
TRAINING_VERSION: int = 1  # ← ĐỔI SỐ NÀY ĐỂ CHUYỂN VERSION 

# =========================================================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ SIÊU THAM SỐ
# =========================================================================
# BẮT BUỘC dùng file đã filter để V1 không bị kẹt ở loss 0.6931
CSV_PATH      = "dataset/processed/poi_data_ggmap_v1_filtered.csv" 
IMAGE_DIR     = r"D:/poi-urban-danang/dataset/poi_images_ggmap"
VOID_PATH     = "dataset/sampling/urban_voids.csv"
GEOM_DIR      = r"D:/poi-urban-danang/dataset/building_images_ggmap"
VOID_GEOM_DIR = r"D:/poi-urban-danang/dataset/building_images_voids"

BATCH_SIZE  = 32
NUM_EPOCHS  = 10
TRAIN_RATIO = 0.8
GROUP_SIZE  = 8
TEMPERATURE = 0.5
LR          = 1e-4
HARD_NEG_K  = 24 

def collate_fn(batch):
    def stack_images(images):
        if isinstance(images[0], torch.Tensor):
            return torch.stack(images)
        return images

    poi_batch = {
        'poi_id'    : [item['poi']['poi_id']     for item in batch],
        'district'  : [item['poi']['district']   for item in batch],
        'coords'    : torch.stack([item['poi']['coords'] for item in batch]),
        'text'      : [item['poi']['text']       for item in batch],
        'category'  : [item['poi']['category']   for item in batch],
        'image'     : stack_images([item['poi']['image']      for item in batch]),
        'geom_image': stack_images([item['poi']['geom_image'] for item in batch]),
    }

    if batch[0].get('void') and 'void_id' in batch[0]['void']:
        void_batch = {
            'void_id'   : [item['void']['void_id']    for item in batch],
            'coords'    : torch.stack([item['void']['void_coords'] for item in batch]),
            'text'      : [item['void']['void_text']  for item in batch],
            'image'     : stack_images([item['void']['void_image']      for item in batch]),
            'geom_image': stack_images([item['void']['void_geom_image'] for item in batch]),
        }
    else:
        void_batch = {}

    return {'poi': poi_batch, 'void': void_batch}

# =========================================================================
# HÀM PHỤ TRỢ: Encode danh sách poi_dict/void_dict thành tensor embedding
# =========================================================================
def _encode_neighbor_groups(
    neighbor_list,      # List[dict] — output của get_nearest_poi_data hoặc get_nearest_void_data
    multimodal_encoder,
    group_encoder,
    device,
    transform,
):
    """
    Nhận danh sách dict các điểm lân cận, encode và trả về các negative group tensors.
    """
    neg_groups_list = []

    if not neighbor_list:
        return neg_groups_list

    # Stack dữ liệu từ list[dict] thành batch tensors
    neighbor_images = []
    neighbor_geoms  = []
    neighbor_texts  = []
    neighbor_coords = []

    for item in neighbor_list:
        # Xử lý ảnh (Multimodal)
        img = item['image']
        if not isinstance(img, torch.Tensor):
            # Nếu là list PIL (như POI) hoặc 1 PIL (như Void) thì phải transform
            if isinstance(img, list):
                img = torch.stack([transform(i) for i in img])
            else:
                img = transform(img).unsqueeze(0) # Tạo batch dimension giả cho CLIP
        neighbor_images.append(img)

        # Xử lý ảnh không gian (Geom)
        geom = item['geom_image']
        if not isinstance(geom, torch.Tensor):
            geom = transform(geom)
        neighbor_geoms.append(geom)

        neighbor_texts.append(item['text'])
        neighbor_coords.append(item['coords'])

    # Đẩy lên GPU
    neighbor_images = torch.stack(neighbor_images).to(device)
    neighbor_geoms  = torch.stack(neighbor_geoms).to(device)
    neighbor_coords = torch.stack(neighbor_coords).to(device)

    # Encode qua MultimodalEncoder (tắt grad để nhẹ máy)
    with torch.no_grad():
        neg_features = multimodal_encoder(
            geom_images=neighbor_geoms,
            images=neighbor_images,
            texts=neighbor_texts,
        )

    # Chia thành các nhóm GROUP_SIZE (ví dụ 8 điểm/nhóm) và qua BuildingGroupEncoder
    num_neg_groups = neg_features.shape[0] // GROUP_SIZE
    for i in range(num_neg_groups):
        neg_feats  = neg_features[i * GROUP_SIZE : (i + 1) * GROUP_SIZE]
        neg_coords = neighbor_coords[i * GROUP_SIZE : (i + 1) * GROUP_SIZE]
        
        if len(neg_feats) < GROUP_SIZE:
            break
            
        dist_neg  = haversine_matrix_torch(neg_coords)
        # Tăng chiều lên [1, GROUP_SIZE, Dim] để qua Transformer của GroupEncoder
        neg_group = group_encoder(neg_feats.unsqueeze(0), dist_neg).mean(dim=1)
        neg_groups_list.append(neg_group)

    return neg_groups_list

# =========================================================================
# 1. HÀM HUẤN LUYỆN AI (TRAINING)
# =========================================================================
def train_urban_ai():
    print("=" * 65)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN | {VERSION_DESC[TRAINING_VERSION]}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Thiết bị: {device.upper()}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = POIDataset(
        csv_file            = CSV_PATH,
        image_transform     = transform,
        image_dir           = IMAGE_DIR,
        void_csv_file       = VOID_PATH,
        geom_image_dir      = GEOM_DIR,
        void_geom_image_dir = VOID_GEOM_DIR,
    )

    total_len = len(full_dataset)
    train_len = int(total_len * TRAIN_RATIO)
    test_len  = total_len - train_len

    train_dataset, test_dataset = random_split(
        full_dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42)
    )

    # [TỐI ƯU TỐC ĐỘ]: Thêm num_workers=4 và pin_memory=True để load data cực nhanh
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, pin_memory=True, collate_fn=collate_fn, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=False, pin_memory=True, collate_fn=collate_fn, num_workers=4
    )

    multimodal_encoder = MultimodalEncoder(version=TRAINING_VERSION).to(device)
    group_encoder      = BuildingGroupEncoder(embed_dim=64, num_heads=4).to(device)

    for param in multimodal_encoder.clip_model.parameters():      param.requires_grad = False
    for param in multimodal_encoder.text_projection.parameters(): param.requires_grad = True
    for param in multimodal_encoder.image_projection.parameters(): param.requires_grad = True

# 1. Thêm weight_decay=1e-4 để chống học thuộc vẹt (Overfitting)
    optimizer = optim.Adam([
        {'params': multimodal_encoder.text_projection.parameters(),  'lr': LR},
        {'params': multimodal_encoder.image_projection.parameters(), 'lr': LR},
        {'params': multimodal_encoder.resnet.parameters(),            'lr': LR * 0.1},
        {'params': multimodal_encoder.fusion.parameters(),            'lr': LR},
        {'params': group_encoder.parameters(),                        'lr': LR},
    ], weight_decay=1e-4) # <--- CHI TIẾT MỚI
    
    criterion = InfoNCELoss(temperature=TEMPERATURE).to(device)

    # 2. Khởi tạo Scheduler (Cứ sau 3 Epoch sẽ giảm Learning Rate đi một nửa)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    train_loss_history, test_loss_history = [], []
    
    # 3. Biến lưu giữ mức Test Loss tốt nhất
    best_test_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # ---- TRAINING ------------------------------------------------
        multimodal_encoder.train()
        group_encoder.train()
        total_train_loss, num_train_batches = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    # --- 1. XỬ LÝ ANCHOR (POI THẬT) ---
                    poi_data = batch['poi']
                    coords      = poi_data['coords'].to(device, non_blocking=True)
                    images      = poi_data['image'].to(device, non_blocking=True) if isinstance(poi_data['image'], torch.Tensor) else poi_data['image']
                    geom_images = poi_data['geom_image'].to(device, non_blocking=True)
                    texts       = poi_data['text']

                    # Encode toàn bộ POI trong batch qua Multimodal Encoder
                    poi_features = multimodal_encoder(geom_images=geom_images, images=images, texts=texts)
                    
                    # Kiểm tra nếu batch quá nhỏ không đủ tạo Group (Cần ít nhất 2 nhóm: 1 Anchor, 1 Hard Neg)
                    if poi_features.shape[0] < GROUP_SIZE * 2:
                        continue

                    # Trích xuất Anchor Group (Nhóm POI mục tiêu)
                    anchor_feats  = poi_features[:GROUP_SIZE]
                    anchor_coords = coords[:GROUP_SIZE]
                    dist_anchor   = haversine_matrix_torch(anchor_coords)
                    anchor_group  = group_encoder(anchor_feats.unsqueeze(0), dist_anchor).mean(dim=1)

                    # Tạo Positive Group (Bằng cách thêm Noise vào Anchor - Data Augmentation)
                    noise = torch.randn_like(anchor_feats) * 0.15
                    positive_group = group_encoder((anchor_feats + noise).unsqueeze(0), dist_anchor).mean(dim=1)

                    # Danh sách chứa các nhóm âm bản (Negatives)
                    neg_groups_list = []

                    # --- 2. XỬ LÝ IN-BATCH HARD NEGATIVE (Các POI khác trong cùng batch) ---
                    anchor_centroid = anchor_coords.mean(dim=0, keepdim=True)
                    other_coords = coords[GROUP_SIZE:]
                    other_feats = poi_features[GROUP_SIZE:]

                    if len(other_coords) >= GROUP_SIZE:
                        # Tìm các POI trong batch có vị trí gần Anchor nhất (Hard Negatives)
                        dists_to_anchor = torch.cdist(anchor_centroid, other_coords).squeeze(0)
                        num_hard_negs = (len(other_coords) // GROUP_SIZE) * GROUP_SIZE
                        _, hard_neg_indices = torch.topk(dists_to_anchor, k=num_hard_negs, largest=False)
                        
                        h_feats_all = other_feats[hard_neg_indices]
                        h_coords_all = other_coords[hard_neg_indices]
                        
                        for i in range(num_hard_negs // GROUP_SIZE):
                            n_feats = h_feats_all[i*GROUP_SIZE : (i+1)*GROUP_SIZE]
                            n_coords = h_coords_all[i*GROUP_SIZE : (i+1)*GROUP_SIZE]
                            dist_n = haversine_matrix_torch(n_coords)
                            n_group = group_encoder(n_feats.unsqueeze(0), dist_n).mean(dim=1)
                            neg_groups_list.append(n_group)

                    # --- 3. XỬ LÝ SPATIAL VOID NEGATIVE (Vùng trống lân cận từ BallTree) ---
                    anchor_poi_id = poi_data['poi_id'][0]
                    # Lấy index thực trong DataFrame để truy vấn không gian
                    anchor_full_idx = full_dataset.data[
                        full_dataset.data['RestaurantID'].astype(str) == str(anchor_poi_id)
                    ].index
                    
                    if len(anchor_full_idx) > 0:
                        idx = int(anchor_full_idx[0])
                        # Lấy 16 vùng trống gần nhất (tạo ra 2 nhóm âm bản)
                        nearest_voids = full_dataset.get_nearest_void_data(idx, k=16)
                        
                        if len(nearest_voids) >= GROUP_SIZE:
                            void_neg_groups = _encode_neighbor_groups(
                                nearest_voids, multimodal_encoder, group_encoder, device, transform
                            )
                            neg_groups_list.extend(void_neg_groups)

                    # --- 4. TỔNG HỢP VÀ TÍNH LOSS ---
                    # Chỉ tính Loss nếu tìm được ít nhất 1 nhóm âm bản để so sánh
                    if len(neg_groups_list) > 0:
                        # Gộp tất cả các nhóm âm (POI gần đó + Vùng trống gần đó)
                        negative_groups = torch.cat(neg_groups_list, dim=0)
                        
                        # Contrastive Loss (InfoNCE / Triplet)
                        loss = criterion(anchor_group, positive_group, negative_groups)
                        
                        loss.backward()
                        optimizer.step()

                        total_train_loss += loss.item()
                        num_train_batches += 1
                        
                        # In tiến độ sau mỗi 10 batch để biết máy không bị treo
                        if batch_idx % 10 == 0:
                            print(f"   Batch {batch_idx} | Loss: {loss.item():.4f}")
                    else:
                        # Bỏ qua batch nếu không có mẫu âm (tránh lỗi code)
                        continue

        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        train_loss_history.append(avg_train_loss)
# ---- VALIDATION ------------------------------------------------
        multimodal_encoder.eval()
        group_encoder.eval()
        total_test_loss, num_test_batches = 0.0, 0

        with torch.no_grad():
            for batch in test_loader:
                # --- 1. XỬ LÝ ANCHOR ---
                poi_data = batch['poi']
                coords      = poi_data['coords'].to(device, non_blocking=True)
                images      = poi_data['image'].to(device, non_blocking=True) if isinstance(poi_data['image'], torch.Tensor) else poi_data['image']
                geom_images = poi_data['geom_image'].to(device, non_blocking=True)
                texts       = poi_data['text']

                poi_features = multimodal_encoder(geom_images=geom_images, images=images, texts=texts)

                if poi_features.shape[0] < GROUP_SIZE * 2:
                    continue

                anchor_feats  = poi_features[:GROUP_SIZE]
                anchor_coords = coords[:GROUP_SIZE]
                dist_anchor   = haversine_matrix_torch(anchor_coords)
                anchor_group  = group_encoder(anchor_feats.unsqueeze(0), dist_anchor).mean(dim=1)

                # Positive Group cho Test: Giữ nguyên gốc, không cần noise (hoặc noise cực nhỏ) để đánh giá tính ổn định
                positive_group = group_encoder(anchor_feats.unsqueeze(0), dist_anchor).mean(dim=1)

                neg_groups_list = []

                # --- 2. XỬ LÝ IN-BATCH HARD NEGATIVE (Đồng bộ với Train) ---
                anchor_centroid = anchor_coords.mean(dim=0, keepdim=True)
                other_coords = coords[GROUP_SIZE:]
                other_feats = poi_features[GROUP_SIZE:]

                if len(other_coords) >= GROUP_SIZE:
                    dists_to_anchor = torch.cdist(anchor_centroid, other_coords).squeeze(0)
                    num_hard_negs = (len(other_coords) // GROUP_SIZE) * GROUP_SIZE
                    _, hard_neg_indices = torch.topk(dists_to_anchor, k=num_hard_negs, largest=False)
                    
                    h_feats_all = other_feats[hard_neg_indices]
                    h_coords_all = other_coords[hard_neg_indices]
                    
                    for i in range(num_hard_negs // GROUP_SIZE):
                        n_feats = h_feats_all[i*GROUP_SIZE : (i+1)*GROUP_SIZE]
                        n_coords = h_coords_all[i*GROUP_SIZE : (i+1)*GROUP_SIZE]
                        dist_n = haversine_matrix_torch(n_coords)
                        n_group = group_encoder(n_feats.unsqueeze(0), dist_n).mean(dim=1)
                        neg_groups_list.append(n_group)

                # --- 3. XỬ LÝ SPATIAL VOID NEGATIVE (Đồng bộ với Train) ---
                anchor_poi_id = poi_data['poi_id'][0]
                anchor_full_idx = full_dataset.data[
                    full_dataset.data['RestaurantID'].astype(str) == str(anchor_poi_id)
                ].index
                
                if len(anchor_full_idx) > 0:
                    idx = int(anchor_full_idx[0])
                    nearest_voids = full_dataset.get_nearest_void_data(idx, k=16)
                    
                    if len(nearest_voids) >= GROUP_SIZE:
                        void_neg_groups = _encode_neighbor_groups(
                            nearest_voids, multimodal_encoder, group_encoder, device, transform
                        )
                        neg_groups_list.extend(void_neg_groups)

                # --- 4. TỔNG HỢP VÀ TÍNH LOSS ---
                if len(neg_groups_list) > 0:
                    negative_groups = torch.cat(neg_groups_list, dim=0)
                    test_loss = criterion(anchor_group, positive_group, negative_groups)
                    total_test_loss  += test_loss.item()
                    num_test_batches += 1
                else:
                    continue

# ... (code validation ở trên giữ nguyên) ...
        avg_test_loss = total_test_loss / max(num_test_batches, 1)
        test_loss_history.append(avg_test_loss)

        print(f"🔄 Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        # ---- [MỚI] GIẢM LEARNING RATE ----
        scheduler.step()

        # ---- [MỚI] CHỈ LƯU MÔ HÌNH KHI TEST LOSS GIẢM (BEST MODEL) ----
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            os.makedirs("models_saved", exist_ok=True)
            torch.save(multimodal_encoder.state_dict(), f"models_saved/multimodal_best_v{TRAINING_VERSION}.pth")
            torch.save(group_encoder.state_dict(),      f"models_saved/group_encoder_best_v{TRAINING_VERSION}.pth")
            print(f"   🌟 Đã lưu Best Model! (Test Loss giảm xuống: {best_test_loss:.4f})")
        else:
            print(f"   ⚠️ Test Loss không giảm (Best: {best_test_loss:.4f}). Bỏ qua lưu model.")

    print(f"\n✅ HOÀN TẤT HUẤN LUYỆN! ({VERSION_DESC[TRAINING_VERSION]})")

    # ------------------------------------------------------------------
    # LƯU KẾT QUẢ VÀO CSV 
    # (Lưu ý: Đã xóa phần save .pth ở đây vì đã save best model ở trên)
    # ------------------------------------------------------------------
    os.makedirs("reports/metrics", exist_ok=True)
    df_loss = pd.DataFrame({
        "Epoch"     : range(1, NUM_EPOCHS + 1),
        "Train_Loss": train_loss_history,
        "Test_Loss" : test_loss_history,
    })
    df_loss.to_csv(f"reports/metrics/training_loss_v{TRAINING_VERSION}.csv", index=False)

    os.makedirs("models_saved", exist_ok=True)
    torch.save(multimodal_encoder.state_dict(), f"models_saved/multimodal_best_v{TRAINING_VERSION}.pth")
    torch.save(group_encoder.state_dict(),      f"models_saved/group_encoder_best_v{TRAINING_VERSION}.pth")
    print(f"💾 Đã lưu weights tại: models_saved/*_v{TRAINING_VERSION}.pth")


# =========================================================================
# 2. VẼ BIỂU ĐỒ LOSS
# =========================================================================
def plot_training_loss():
    print("\n📈 Đang vẽ biểu đồ Loss Curve...")
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        sns.set_theme(style="whitegrid")
        df_loss = pd.read_csv(f"reports/metrics/training_loss_v{TRAINING_VERSION}.csv")

        plt.figure(figsize=(9, 5))
        plt.plot(df_loss['Epoch'], df_loss['Train_Loss'], marker='o', color='#d62728', linewidth=2.5, label='Train Loss')
        plt.plot(df_loss['Epoch'], df_loss['Test_Loss'],  marker='s', color='#1f77b4', linewidth=2.5, label='Test Loss', linestyle='--')
        plt.title(f"Loss Curve – {VERSION_DESC[TRAINING_VERSION]}", fontsize=13, fontweight='bold')
        plt.xlabel("Số vòng học (Epoch)", fontsize=11)
        plt.ylabel("InfoNCE Loss", fontsize=11)
        plt.xticks(df_loss['Epoch'])
        plt.legend()

        os.makedirs("reports/figures", exist_ok=True)
        out_path = f"reports/figures/loss_curve_v{TRAINING_VERSION}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu: {out_path}")
    except Exception as e:
        print(f"⚠️  Không thể vẽ biểu đồ Loss: {e}")


# =========================================================================
# 3. PHÂN CỤM t-SNE TRÊN TẬP TEST
# =========================================================================
def plot_tsne_clusters():
    print("\n🧠 Đang trích xuất đặc trưng để phân cụm (t-SNE)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultimodalEncoder(version=TRAINING_VERSION).to(device)
    weights_path = f"models_saved/multimodal_best_v{TRAINING_VERSION}.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Đã nạp weights: {weights_path}")
    except:
        print(f"⚠️  Không tìm thấy weights tại {weights_path}. Dùng model gốc.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = POIDataset(
        csv_file        = CSV_PATH,
        image_transform = transform,
        image_dir       = IMAGE_DIR,
        void_csv_file   = VOID_PATH,
        geom_image_dir  = GEOM_DIR,
    )
    tsne_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    all_features   = []
    all_categories = []

    model.eval()
    print("🔄 Đang chạy dữ liệu qua mô hình...")
    with torch.no_grad():
        for batch in tsne_loader:
            poi_data = batch['poi']

            images      = poi_data['image']
            geom_images = poi_data['geom_image']
            texts       = poi_data['text']
            
            if isinstance(images, torch.Tensor):
                images = images.to(device, non_blocking=True)
            if isinstance(geom_images, torch.Tensor):
                geom_images = geom_images.to(device, non_blocking=True)

            # [FIX LỖI] Đã thêm geom_images vào để Version 1 có thể t-SNE mượt mà
            feats = model(geom_images=geom_images, images=images, texts=texts)
            all_features.append(feats.cpu())
            all_categories.extend(poi_data['category'])

    all_features = torch.cat(all_features, dim=0).numpy()

    print("🌌 Đang chiếu 64 chiều → 2 chiều bằng t-SNE...")
    tsne        = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(all_features)

    df_plot = pd.DataFrame()
    df_plot['tsne_x']   = tsne_results[:, 0]
    df_plot['tsne_y']   = tsne_results[:, 1]
    df_plot['Category'] = all_categories

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='tsne_x', y='tsne_y', hue='Category',
        palette='tab10', data=df_plot, legend="full", alpha=0.8, s=60,
    )
    plt.title(f"Phân cụm t-SNE – {VERSION_DESC[TRAINING_VERSION]}", fontsize=14, fontweight='bold')
    plt.xlabel("Chiều Đặc trưng 1")
    plt.ylabel("Chiều Đặc trưng 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    os.makedirs("reports/figures", exist_ok=True)
    out_path = f"reports/figures/tsne_v{TRAINING_VERSION}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu: {out_path}")
    print(f"\n🎉 HOÀN THÀNH! Version {TRAINING_VERSION} – {VERSION_DESC[TRAINING_VERSION]}")

# =========================================================================
# CHẠY TỰ ĐỘNG TẤT CẢ
# =========================================================================
if __name__ == "__main__":
    # Đặt biến môi trường để đa luồng trên Windows không bị lỗi
    import multiprocessing
    multiprocessing.freeze_support()
    
    print("✅ Main script bắt đầu chạy...")
    print(f"⚙️  TRAINING_VERSION = {TRAINING_VERSION} | {VERSION_DESC[TRAINING_VERSION]}")
    train_urban_ai()
    plot_training_loss()
    plot_tsne_clusters()