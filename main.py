import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from torchvision import transforms
from requests import RequestsDependencyWarning

from src.data.dataset import POIDataset, custom_collate_fn
from src.encoder.multimodal_encoder import MultimodalEncoder, VERSION_DESC
from src.models.building_group import BuildingGroupEncoder
from src.models.loss_functions import InfoNCELoss
from utlis.geo_utils import haversine_matrix_torch

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
print("🔇 Đã tắt RequestsDependencyWarning")

# =========================================================================
# BIẾN GLOBAL & CẤU HÌNH
# =========================================================================
TRAINING_VERSION: int = 1  

# --- TRAIN: GOOGLE MAPS (Source Domain) ---
TRAIN_CSV       = "dataset/processed/master_nodes_google_maps_clean.csv" 
TRAIN_IMAGE_DIR = "dataset/poi_images_ggmap"
TRAIN_GEOM_DIR  = "dataset/building_images_google_maps"
VOID_PATH       = "dataset/sampling/urban_voids_google_maps.csv"
VOID_GEOM_DIR   = "dataset/building_images_google_maps"

# --- TEST: FOODY (Target Domain - Zero Shot) ---
TEST_CSV        = "dataset/processed/master_nodes_foody_clean.csv"
TEST_IMAGE_DIR  = "dataset/poi_images_foody"
TEST_GEOM_DIR   = "dataset/building_images_foody"

BATCH_SIZE  = 32
NUM_EPOCHS  = 10
GROUP_SIZE  = 8
TEMPERATURE = 0.5
LR          = 1e-4

# =========================================================================
# HÀM ĐÁNH GIÁ CHẤT LƯỢNG EMBEDDING (KHOA HỌC)
# =========================================================================
def evaluate_embeddings(features, labels, k=5):
    """
    Tính Silhouette Score (Độ gôm cụm) và Recall@K (Khả năng truy xuất).
    Đây là 2 chỉ số chuẩn mực nhất cho Contrastive Learning.
    """
    if len(features) < 2 or len(np.unique(labels)) < 2:
        return 0.0, 0.0

    features = np.array(features)
    labels = np.array(labels)

    # 1. Tính Silhouette Score (metric='cosine' vì dùng InfoNCE)
    try:
        sil_score = silhouette_score(features, labels, metric='cosine')
    except:
        sil_score = 0.0

    # 2. Tính Recall@K
    try:
        # Tìm k+1 hàng xóm gần nhất (Bỏ qua chính nó ở vị trí đầu tiên)
        n_neighbors = min(k + 1, len(features))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(features)
        _, indices = nn.kneighbors(features)

        hits = 0
        for i in range(len(features)):
            query_label = labels[i]
            # Các nhãn của k hàng xóm gần nhất (không tính chính nó)
            neighbor_labels = [labels[j] for j in indices[i][1:]]
            if query_label in neighbor_labels:
                hits += 1
        
        recall_at_k = hits / len(features)
    except:
        recall_at_k = 0.0

    return sil_score, recall_at_k

def _encode_neighbor_groups(neighbor_list, multimodal_encoder, group_encoder, device, transform):
    neg_groups_list = []
    if not neighbor_list: return neg_groups_list

    neighbor_images, neighbor_geoms, neighbor_texts, neighbor_coords = [], [], [], []

    for item in neighbor_list:
        img = item['image']
        if not isinstance(img, torch.Tensor):
            if isinstance(img, list): img = torch.stack([transform(i) for i in img])
            else: img = transform(img).unsqueeze(0)
        neighbor_images.append(img)

        geom = item['geom_image']
        if not isinstance(geom, torch.Tensor): geom = transform(geom)
        neighbor_geoms.append(geom)

        neighbor_texts.append(item['text'])
        neighbor_coords.append(item['coords'])

    neighbor_images = torch.stack(neighbor_images).to(device)
    neighbor_geoms  = torch.stack(neighbor_geoms).to(device)
    neighbor_coords = torch.stack(neighbor_coords).to(device)

    with torch.no_grad():
        neg_features = multimodal_encoder(geom_images=neighbor_geoms, images=neighbor_images, texts=neighbor_texts)

    num_neg_groups = neg_features.shape[0] // GROUP_SIZE
    for i in range(num_neg_groups):
        neg_feats  = neg_features[i * GROUP_SIZE : (i + 1) * GROUP_SIZE]
        neg_coords = neighbor_coords[i * GROUP_SIZE : (i + 1) * GROUP_SIZE]
        if len(neg_feats) < GROUP_SIZE: break
            
        dist_neg  = haversine_matrix_torch(neg_coords)
        neg_group = group_encoder(neg_feats.unsqueeze(0), dist_neg).mean(dim=1)
        neg_groups_list.append(neg_group)

    return neg_groups_list

# =========================================================================
# HÀM HUẤN LUYỆN CHÍNH
# =========================================================================
def train_urban_ai():
    print("=" * 65)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN ZERO-SHOT DOMAIN ADAPTATION | {VERSION_DESC[TRAINING_VERSION]}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Thiết bị: {device.upper()}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\n📚 Nạp tập Huấn luyện (Google Maps)...")
    train_dataset = POIDataset(
        csv_file            = TRAIN_CSV,
        image_transform     = transform,
        image_dir           = TRAIN_IMAGE_DIR,
        void_csv_file       = VOID_PATH,
        geom_image_dir      = TRAIN_GEOM_DIR,
        void_geom_image_dir = VOID_GEOM_DIR,
    )

    print("📚 Nạp tập Kiểm thử (Foody)...")
    test_dataset = POIDataset(
        csv_file            = TEST_CSV,
        image_transform     = transform,
        image_dir           = TEST_IMAGE_DIR,
        geom_image_dir      = TEST_GEOM_DIR,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, collate_fn=custom_collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=True, collate_fn=custom_collate_fn, num_workers=4)

    multimodal_encoder = MultimodalEncoder(version=TRAINING_VERSION).to(device)
    group_encoder      = BuildingGroupEncoder(embed_dim=64, num_heads=4).to(device)

    for param in multimodal_encoder.clip_model.parameters():      param.requires_grad = False
    for param in multimodal_encoder.text_projection.parameters(): param.requires_grad = True
    for param in multimodal_encoder.image_projection.parameters(): param.requires_grad = True

    optimizer = optim.Adam([
        {'params': multimodal_encoder.text_projection.parameters(),  'lr': LR},
        {'params': multimodal_encoder.image_projection.parameters(), 'lr': LR},
        {'params': multimodal_encoder.resnet.parameters(),           'lr': LR * 0.1},
        {'params': multimodal_encoder.fusion.parameters(),           'lr': LR},
        {'params': group_encoder.parameters(),                       'lr': LR},
    ], weight_decay=1e-4) 
    
    criterion = InfoNCELoss(temperature=TEMPERATURE).to(device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Khởi tạo DataFrame Log
    log_data = []
    best_test_loss = float('inf')
    best_epoch = 1

    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']

        # ---- TRAINING --------------------------------
        multimodal_encoder.train()
        group_encoder.train()
        total_train_loss, num_train_batches = 0.0, 0
        
        train_features, train_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            poi_data = batch['poi']
            coords      = poi_data['coords'].to(device, non_blocking=True)
            images      = poi_data['image'].to(device, non_blocking=True) if isinstance(poi_data['image'], torch.Tensor) else poi_data['image']
            geom_images = poi_data['geom_image'].to(device, non_blocking=True)
            texts       = poi_data['text']

            poi_features = multimodal_encoder(geom_images=geom_images, images=images, texts=texts)
            if poi_features.shape[0] < GROUP_SIZE * 2: continue

            anchor_feats  = poi_features[:GROUP_SIZE]
            anchor_coords = coords[:GROUP_SIZE]
            dist_anchor   = haversine_matrix_torch(anchor_coords)
            anchor_group  = group_encoder(anchor_feats.unsqueeze(0), dist_anchor).mean(dim=1)

            noise = torch.randn_like(anchor_feats) * 0.15
            positive_group = group_encoder((anchor_feats + noise).unsqueeze(0), dist_anchor).mean(dim=1)

            neg_groups_list = []
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

            anchor_poi_id = poi_data['poi_id'][0]
            anchor_full_idx = train_dataset.data[train_dataset.data['Global_ID'].astype(str) == str(anchor_poi_id)].index
            
            if len(anchor_full_idx) > 0:
                idx = int(anchor_full_idx[0])
                nearest_voids = train_dataset.get_nearest_void_data(idx, k=16)
                if len(nearest_voids) >= GROUP_SIZE:
                    void_neg_groups = _encode_neighbor_groups(nearest_voids, multimodal_encoder, group_encoder, device, transform)
                    neg_groups_list.extend(void_neg_groups)

            if len(neg_groups_list) > 0:
                negative_groups = torch.cat(neg_groups_list, dim=0)
                
                anchor_group = F.normalize(anchor_group, p=2, dim=1)
                positive_group = F.normalize(positive_group, p=2, dim=1)
                negative_groups = F.normalize(negative_groups, p=2, dim=1)
                
                # Thu thập Feature và Label để tính Silhouette & Recall
                cat = poi_data.get('category') if 'category' in poi_data else poi_data.get('Category')
                if isinstance(cat, (list, tuple, pd.Series)): cat = cat[0]
                train_features.append(anchor_group.detach().cpu().squeeze().numpy())
                train_labels.append(str(cat))

                loss = criterion(anchor_group, positive_group, negative_groups)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                num_train_batches += 1

        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        tr_sil, tr_r5 = evaluate_embeddings(train_features, train_labels, k=5)

        # ---- VALIDATION (Zero-Shot Foody) ----------------------------
        multimodal_encoder.eval()
        group_encoder.eval()
        total_test_loss, num_test_batches = 0.0, 0
        test_features, test_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                poi_data = batch['poi']
                coords      = poi_data['coords'].to(device, non_blocking=True)
                images      = poi_data['image'].to(device, non_blocking=True) if isinstance(poi_data['image'], torch.Tensor) else poi_data['image']
                geom_images = poi_data['geom_image'].to(device, non_blocking=True)
                texts       = poi_data['text']

                poi_features = multimodal_encoder(geom_images=geom_images, images=images, texts=texts)
                if poi_features.shape[0] < GROUP_SIZE * 2: continue

                anchor_feats  = poi_features[:GROUP_SIZE]
                anchor_coords = coords[:GROUP_SIZE]
                dist_anchor   = haversine_matrix_torch(anchor_coords)
                anchor_group  = group_encoder(anchor_feats.unsqueeze(0), dist_anchor).mean(dim=1)
                positive_group = group_encoder(anchor_feats.unsqueeze(0), dist_anchor).mean(dim=1)

                neg_groups_list = []
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

                if len(neg_groups_list) > 0:
                    negative_groups = torch.cat(neg_groups_list, dim=0)
                    
                    anchor_group = F.normalize(anchor_group, p=2, dim=1)
                    positive_group = F.normalize(positive_group, p=2, dim=1)
                    negative_groups = F.normalize(negative_groups, p=2, dim=1)
                    
                    # Thu thập Feature và Label
                    cat = poi_data.get('category') if 'category' in poi_data else poi_data.get('Category')
                    if isinstance(cat, (list, tuple, pd.Series)): cat = cat[0]
                    test_features.append(anchor_group.detach().cpu().squeeze().numpy())
                    test_labels.append(str(cat))

                    test_loss = criterion(anchor_group, positive_group, negative_groups)
                    total_test_loss += test_loss.item()
                    num_test_batches += 1

        avg_test_loss = total_test_loss / max(num_test_batches, 1)
        te_sil, te_r5 = evaluate_embeddings(test_features, test_labels, k=5)

        print(f"🔄 Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | LR: {current_lr:.6f} | GG Maps Loss: {avg_train_loss:.4f} | Foody Loss: {avg_test_loss:.4f}")
        print(f"   ► TRAIN (GG Maps): Silhouette={tr_sil:.4f}, Recall@5={tr_r5:.4f}")
        print(f"   ► TEST  (Foody)  : Silhouette={te_sil:.4f}, Recall@5={te_r5:.4f}")

        scheduler.step()

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            os.makedirs("models_saved", exist_ok=True)
            torch.save(multimodal_encoder.state_dict(), f"models_saved/multimodal_best_v{TRAINING_VERSION}.pth")
            torch.save(group_encoder.state_dict(),      f"models_saved/group_encoder_best_v{TRAINING_VERSION}.pth")
            print(f"   🌟 Đã lưu Best Model tại Epoch {best_epoch}!")

        # Lưu thông tin Epoch này
        log_data.append({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": avg_train_loss, "train_silhouette": tr_sil, "train_recall_5": tr_r5,
            "test_loss": avg_test_loss,   "test_silhouette": te_sil,  "test_recall_5": te_r5,
        })

    # =========================================================================
    # LƯU THỐNG KÊ LOG
    # =========================================================================
    os.makedirs("reports/metrics", exist_ok=True)
    df_loss = pd.DataFrame(log_data)
    df_loss['is_best_model'] = ["Yes" if e == best_epoch else "No" for e in df_loss['epoch']]
    
    csv_path = f"reports/metrics/training_loss_v{TRAINING_VERSION}.csv"
    df_loss.to_csv(csv_path, index=False)
    
    print("\n✅ HOÀN TẤT HUẤN LUYỆN ZERO-SHOT!")
    print(f"📊 Bảng dữ liệu Clustering & Retrieval Metrics đã được xuất tại: {csv_path}")

def plot_training_loss():
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        sns.set_theme(style="whitegrid")
        df_loss = pd.read_csv(f"reports/metrics/training_loss_v{TRAINING_VERSION}.csv")
        
        # Biểu đồ Loss
        plt.figure(figsize=(9, 5))
        plt.plot(df_loss['epoch'], df_loss['train_loss'], marker='o', color='#d62728', label='GG Maps (Train) Loss')
        plt.plot(df_loss['epoch'], df_loss['test_loss'],  marker='s', color='#1f77b4', label='Foody (Test) Loss', linestyle='--')
        best_row = df_loss[df_loss['is_best_model'] == 'Yes'].iloc[0]
        plt.scatter(best_row['epoch'], best_row['test_loss'], color='gold', s=200, edgecolors='black', marker='*', zorder=5, label=f'Best Epoch ({int(best_row["epoch"])})')
        plt.title(f"Zero-shot Domain Adaptation Curve – {VERSION_DESC[TRAINING_VERSION]}")
        plt.xlabel("Epoch")
        plt.ylabel("InfoNCE Loss")
        plt.legend()
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig(f"reports/figures/loss_curve_v{TRAINING_VERSION}.png", dpi=300, bbox_inches='tight')
        
    except Exception as e: print(f"⚠️ Lỗi vẽ biểu đồ: {e}")

def plot_tsne_clusters():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalEncoder(version=TRAINING_VERSION).to(device)
    try: model.load_state_dict(torch.load(f"models_saved/multimodal_best_v{TRAINING_VERSION}.pth", map_location=device))
    except: pass
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    test_dataset = POIDataset(csv_file=TEST_CSV, image_transform=transform, image_dir=TEST_IMAGE_DIR, geom_image_dir=TEST_GEOM_DIR)
    tsne_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)

    all_features, all_categories = [], []
    model.eval()
    with torch.no_grad():
        for batch in tsne_loader:
            poi_data = batch['poi']
            images = poi_data['image'].to(device) if isinstance(poi_data['image'], torch.Tensor) else poi_data['image']
            geom_images = poi_data['geom_image'].to(device)
            feats = model(geom_images=geom_images, images=images, texts=poi_data['text'])
            feats = F.normalize(feats, p=2, dim=1) 
            all_features.append(feats.cpu())
            
            cats = poi_data.get('category') if 'category' in poi_data else poi_data.get('Category')
            all_categories.extend(cats)

    if all_features:
        tsne_results = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(torch.cat(all_features, dim=0).numpy())
        df_plot = pd.DataFrame({'tsne_x': tsne_results[:, 0], 'tsne_y': tsne_results[:, 1], 'Category': all_categories})
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='tsne_x', y='tsne_y', hue='Category', data=df_plot, legend="full", alpha=0.8)
        plt.title(f"Phân cụm t-SNE Foody (Zero-shot) – {VERSION_DESC[TRAINING_VERSION]}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f"reports/figures/tsne_v{TRAINING_VERSION}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train_urban_ai()
    plot_training_loss()
    plot_tsne_clusters()