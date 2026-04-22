import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import gc
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from requests import RequestsDependencyWarning
from tqdm import tqdm
from src.data.dataset import POIDataset, custom_collate_fn
from src.encoder.multimodal_encoder import MultimodalEncoder, VERSION_DESC
from src.models.building_group import BuildingGroupEncoder
from src.models.loss_functions import InfoNCELoss, MarginInfoNCELoss, AdaptiveTripletLoss
from utlis.geo_utils import haversine_matrix_torch

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)


# =========================================================================
# BIẾN GLOBAL & CẤU HÌNH 1,2, 3,
# =========================================================================
VERSIONS_TO_TRAIN = [  4] 

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
EMBEDDING_DIM = 256
BATCH_SIZE  = 16
NUM_EPOCHS  = 10
GROUP_SIZE  = 4
TEMPERATURE = 0.1
MARGIN      = 0.02
LR          = 1e-4
POSITIVE_NOISE_STD = 0.35
POSITIVE_FEATURE_DROPOUT = 0.15
DATA_LOADER_WORKERS = 0
MIN_GROUP_PURITY = 0.75

# =========================================================================
# HÀM ĐÁNH GIÁ CHẤT LƯỢNG EMBEDDING (KHOA HỌC)
# =========================================================================
def _normalize_label(label):
    if pd.isna(label):
        return "Unknown"

    clean_label = str(label).strip()
    if not clean_label or clean_label.lower() in {"nan", "none", "null", "unknown"}:
        return "Unknown"
    return clean_label

def _prepare_labeled_embeddings(features, labels, min_samples_per_class=2, drop_unknown=True):
    if len(features) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), np.array([], dtype=object)

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray([_normalize_label(label) for label in labels], dtype=object)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    valid_mask = ~np.isnan(features).any(axis=1)
    if drop_unknown:
        valid_mask &= labels != "Unknown"

    features = features[valid_mask]
    labels = labels[valid_mask]

    if len(features) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), np.array([], dtype=object)

    label_counts = pd.Series(labels).value_counts()
    keep_labels = label_counts[label_counts >= min_samples_per_class].index.tolist()
    keep_mask = np.isin(labels, keep_labels)
    return features[keep_mask], labels[keep_mask]

def _prepare_projection_features(features):
    features = np.asarray(features, dtype=np.float32)
    if len(features) < 2:
        return features

    scaled_features = StandardScaler().fit_transform(features)
    pca_dims = min(32, scaled_features.shape[0] - 1, scaled_features.shape[1])
    if pca_dims >= 2 and pca_dims < scaled_features.shape[1]:
        scaled_features = PCA(n_components=pca_dims, random_state=42).fit_transform(scaled_features)

    return scaled_features

def evaluate_embeddings(features, labels, k=5):
    # # --- THÊM 2 DÒNG NÀY VÀO ---
    # unique_labels = np.unique(labels) if len(labels) > 0 else []
    # print(f"\n👉 [DEBUG] Số vector thu được: {len(features)} | Số nhãn khác nhau: {len(unique_labels)} -> {unique_labels}")
    # ---------------------------
    if len(features) < 2 or len(np.unique(labels)) < 2:
        return 0.0, 0.0

    features = np.array(features)
    labels = np.array(labels)

    # Nếu có giá trị NaN (Not a Number), mô hình có thể bị nổ gradient
    if np.isnan(features).any():
        print("⚠️ Lỗi: Features chứa giá trị NaN!")
        return 0.0, 0.0

    # 1. Tính Silhouette Score
    sil_score = 0.0
    try:
        # Nếu mỗi sample đều là 1 class riêng biệt thì thuật toán sẽ lỗi
        if len(np.unique(labels)) < len(labels): 
            sil_score = silhouette_score(features, labels, metric='cosine')
    except Exception as e:
        print(f"⚠️ Lỗi khi tính Silhouette: {e}")
        sil_score = 0.0

    # 2. Tính Recall@K
    recall_at_k = 0.0
    try:
        n_neighbors = min(k + 1, len(features))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(features)
        _, indices = nn.kneighbors(features)

        hits = 0
        for i in range(len(features)):
            query_label = labels[i]
            neighbor_labels = [labels[j] for j in indices[i][1:]]
            if query_label in neighbor_labels:
                hits += 1
        
        recall_at_k = hits / len(features)
    except Exception as e:
        print(f"⚠️ Lỗi khi tính Recall: {e}")
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
        neg_group = group_encoder(neg_feats.unsqueeze(0), dist_neg)
        neg_groups_list.append(neg_group)

    return neg_groups_list

def _build_positive_group(anchor_feats, dist_anchor, group_encoder, noise_std=POSITIVE_NOISE_STD):
    noisy_feats = anchor_feats + torch.randn_like(anchor_feats) * noise_std
    dropout_mask = (torch.rand_like(noisy_feats) > POSITIVE_FEATURE_DROPOUT).float()
    noisy_feats = noisy_feats * dropout_mask
    return group_encoder(noisy_feats.unsqueeze(0), dist_anchor)

def _contrast_stats(anchor_group, positive_group, negative_groups):
    pos_sim = F.cosine_similarity(anchor_group, positive_group, dim=-1).mean().item()
    neg_sim = F.cosine_similarity(anchor_group.unsqueeze(1), negative_groups.unsqueeze(0), dim=-1).mean().item()
    return pos_sim, neg_sim

def _select_spatial_group_indices(coords, group_size, candidate_indices=None, reference_coord=None):
    if candidate_indices is None:
        candidate_indices = torch.arange(coords.shape[0], device=coords.device)
    elif not isinstance(candidate_indices, torch.Tensor):
        candidate_indices = torch.tensor(candidate_indices, device=coords.device, dtype=torch.long)
    else:
        candidate_indices = candidate_indices.to(device=coords.device, dtype=torch.long)

    if candidate_indices.numel() < group_size:
        return None

    candidate_coords = coords[candidate_indices]
    pairwise = torch.cdist(candidate_coords, candidate_coords)

    if reference_coord is None:
        knn = torch.topk(pairwise, k=group_size, largest=False).indices
        compactness = pairwise.gather(1, knn).mean(dim=1)
        seed_pos = torch.argmin(compactness)
    else:
        ref = reference_coord.to(coords.device).reshape(1, -1)
        seed_pos = torch.argmin(torch.cdist(candidate_coords, ref).squeeze(1))

    seed_coord = candidate_coords[seed_pos].unsqueeze(0)
    seed_dists = torch.cdist(seed_coord, candidate_coords).squeeze(0)
    chosen_pos = torch.topk(seed_dists, k=group_size, largest=False).indices
    chosen_indices = candidate_indices[chosen_pos]

    chosen_coords = coords[chosen_indices]
    centroid = chosen_coords.mean(dim=0, keepdim=True)
    centroid_order = torch.argsort(torch.cdist(centroid, chosen_coords).squeeze(0))
    return chosen_indices[centroid_order]

def _build_negative_groups_from_batch(poi_features, coords, group_encoder, anchor_indices):
    remaining = torch.arange(coords.shape[0], device=coords.device)
    remaining = remaining[~torch.isin(remaining, anchor_indices)]
    neg_groups_list = []
    anchor_centroid = coords[anchor_indices].mean(dim=0)

    while remaining.numel() >= GROUP_SIZE:
        neg_indices = _select_spatial_group_indices(
            coords,
            GROUP_SIZE,
            candidate_indices=remaining,
            reference_coord=anchor_centroid,
        )
        if neg_indices is None:
            break

        neg_feats = poi_features[neg_indices]
        neg_coords = coords[neg_indices]
        dist_neg = haversine_matrix_torch(neg_coords)
        neg_group = group_encoder(neg_feats.unsqueeze(0), dist_neg)
        neg_groups_list.append(neg_group)

        remaining = remaining[~torch.isin(remaining, neg_indices)]

    return neg_groups_list

def _resolve_group_label_info(categories, group_indices):
    if isinstance(categories, str):
        label = _normalize_label(categories)
        purity = 1.0 if label != "Unknown" else 0.0
        return label, purity

    labels = [_normalize_label(categories[int(idx)]) for idx in group_indices.detach().cpu().tolist()]
    valid_labels = [label for label in labels if label != "Unknown"]
    if not valid_labels:
        return "Unknown", 0.0

    label_counts = pd.Series(valid_labels).value_counts()
    majority_label = label_counts.index[0]
    purity = float(label_counts.iloc[0] / len(labels))
    return majority_label, purity

def _resolve_group_label(categories, group_indices):
    label, _ = _resolve_group_label_info(categories, group_indices)
    return label

def _collect_group_embeddings_from_loader(loader, multimodal_encoder, group_encoder, device):
    all_features, all_categories = [], []

    multimodal_encoder.eval()
    group_encoder.eval()
    with torch.no_grad():
        for batch in loader:
            poi_data = batch['poi']
            coords = poi_data['coords'].to(device, non_blocking=True)
            images = poi_data['image'].to(device, non_blocking=True) if isinstance(poi_data['image'], torch.Tensor) else poi_data['image']
            geom_images = poi_data['geom_image'].to(device, non_blocking=True)
            texts = poi_data['text']

            poi_features = multimodal_encoder(geom_images=geom_images, images=images, texts=texts)
            if poi_features.shape[0] < GROUP_SIZE:
                continue

            cats = poi_data.get('Category', poi_data.get('category', 'Unknown'))
            remaining = torch.arange(poi_features.shape[0], device=device)
            while remaining.numel() >= GROUP_SIZE:
                group_indices = _select_spatial_group_indices(coords, GROUP_SIZE, candidate_indices=remaining)
                if group_indices is None:
                    break

                group_feats = poi_features[group_indices]
                group_coords = coords[group_indices]
                dist_group = haversine_matrix_torch(group_coords)
                group_embedding = group_encoder(group_feats.unsqueeze(0), dist_group)
                group_embedding = F.normalize(group_embedding, p=2, dim=1)

                group_category, group_purity = _resolve_group_label_info(cats, group_indices)
                if group_purity >= MIN_GROUP_PURITY and group_category != "Unknown":
                    all_features.append(group_embedding.cpu().squeeze(0))
                    all_categories.append(str(group_category))

                remaining = remaining[~torch.isin(remaining, group_indices)]

    if not all_features:
        return np.empty((0, EMBEDDING_DIM)), []

    return torch.stack(all_features).numpy(), all_categories

def _compute_epoch_retrieval_metrics(dataset, multimodal_encoder, group_encoder, device):
    eval_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=DATA_LOADER_WORKERS,
    )
    features, labels = _collect_group_embeddings_from_loader(eval_loader, multimodal_encoder, group_encoder, device)
    return evaluate_embeddings(features, labels, k=5)

def _load_best_models(version, out_paths, device):
    multimodal_encoder = MultimodalEncoder(embed_dim=EMBEDDING_DIM, version=version).to(device)
    group_encoder = BuildingGroupEncoder(embed_dim=EMBEDDING_DIM, num_heads=4).to(device)

    multimodal_encoder.load_state_dict(torch.load(f"{out_paths['models']}/multimodal_best.pth", map_location=device))
    group_encoder.load_state_dict(torch.load(f"{out_paths['models']}/group_encoder_best.pth", map_location=device))

    multimodal_encoder.eval()
    group_encoder.eval()
    return multimodal_encoder, group_encoder

def _collect_group_embeddings(version, out_paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multimodal_encoder, group_encoder = _load_best_models(version, out_paths, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = POIDataset(csv_file=TEST_CSV, image_transform=transform, image_dir=TEST_IMAGE_DIR, geom_image_dir=TEST_GEOM_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=DATA_LOADER_WORKERS)

    all_features, all_categories = _collect_group_embeddings_from_loader(
        loader,
        multimodal_encoder,
        group_encoder,
        device,
    )

    del multimodal_encoder
    del group_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_features, all_categories
# =========================================================================
# HÀM TẠO THƯ MỤC TỰ ĐỘNG CHO TỪNG VERSION
# =========================================================================
def get_output_paths(version):
    base_dir = f"results/v{version}"
    paths = {
        "models": f"{base_dir}/models_saved",
        "metrics": f"{base_dir}/reports/metrics",
        "figures": f"{base_dir}/reports/figures"
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

# =========================================================================
# HÀM HUẤN LUYỆN CHÍNH
# =========================================================================
def train_urban_ai(version, out_paths):
    print("=" * 65)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN ZERO-SHOT DOMAIN ADAPTATION | {VERSION_DESC[version]}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Thiết bị: {device.upper()}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\n📚 Nạp tập Huấn luyện (Google Maps)...")
    train_dataset = POIDataset(csv_file=TRAIN_CSV, image_transform=transform, image_dir=TRAIN_IMAGE_DIR, void_csv_file=VOID_PATH, geom_image_dir=TRAIN_GEOM_DIR, void_geom_image_dir=VOID_GEOM_DIR)
    
    print("📚 Nạp tập Kiểm thử (Foody)...")
    test_dataset = POIDataset(csv_file=TEST_CSV, image_transform=transform, image_dir=TEST_IMAGE_DIR, geom_image_dir=TEST_GEOM_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, collate_fn=custom_collate_fn, num_workers=DATA_LOADER_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=True, collate_fn=custom_collate_fn, num_workers=DATA_LOADER_WORKERS)

    # KHỞI TẠO MÔ HÌNH THEO VERSION TRUYỀN VÀO
    multimodal_encoder = MultimodalEncoder(embed_dim=EMBEDDING_DIM, version=version).to(device)
    group_encoder      = BuildingGroupEncoder(embed_dim=EMBEDDING_DIM, num_heads=4).to(device)
    
    criterion = MarginInfoNCELoss(temperature=TEMPERATURE, margin=MARGIN).to(device)
    # CẤU HÌNH OPTIMIZER LINH HOẠT TỰ ĐỘNG NHẬN DIỆN VERSION
    # =========================================================================
    # 1. Các tham số cơ bản (Version nào cũng có)
    optim_params = [
        {'params': multimodal_encoder.resnet.parameters(), 'lr': LR * 0.1},
        {'params': group_encoder.parameters(),             'lr': LR},
        # THÊM THAM SỐ CỦA HÀM LOSS VÀO ĐÂY ĐỂ MÁY TỰ HỌC NHIỆT ĐỘ
        # {'params': criterion.parameters(),                 'lr': LR}, 
    ]
    # 2. Đóng băng CLIP (chỉ làm nếu Version > 1)
    if hasattr(multimodal_encoder, 'clip_model'):
        for param in multimodal_encoder.clip_model.parameters(): 
            param.requires_grad = False

    # 3. Thêm Text / Image Projection vào Optimizer nếu có
    if hasattr(multimodal_encoder, 'text_projection'):
        for param in multimodal_encoder.text_projection.parameters(): param.requires_grad = True
        optim_params.append({'params': multimodal_encoder.text_projection.parameters(), 'lr': LR})

    if hasattr(multimodal_encoder, 'image_projection'):
        for param in multimodal_encoder.image_projection.parameters(): param.requires_grad = True
        optim_params.append({'params': multimodal_encoder.image_projection.parameters(), 'lr': LR})

    # 4. Thêm mạng Fusion và Gated Attention vào Optimizer (V2, V3, V4)
    if hasattr(multimodal_encoder, 'fusion_proj'):
        optim_params.append({'params': multimodal_encoder.fusion_proj.parameters(), 'lr': LR})

    if hasattr(multimodal_encoder, 'gate'):
        optim_params.append({'params': multimodal_encoder.gate.parameters(), 'lr': LR * 0.5})

    # Khởi tạo Adam với các tham số đã thu thập
    optimizer = optim.Adam(optim_params, weight_decay=1e-4)
    
    # criterion = InfoNCELoss(temperature=TEMPERATURE).to(device)
    # Sử dụng hàm Loss nâng cấp có Margin để ép các cụm t-SNE tách xa nhau
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    log_data = []
    best_test_loss = float('inf')
    best_epoch = 1

    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']

        # ---- TRAINING ----
        multimodal_encoder.train()
        group_encoder.train()
        total_train_loss, num_train_batches = 0.0, 0
        total_train_pos_sim, total_train_neg_sim = 0.0, 0.0

        from tqdm import tqdm
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", ncols=100)
        
        for batch in train_bar:
            optimizer.zero_grad()
            poi_data = batch['poi']
            coords      = poi_data['coords'].to(device, non_blocking=True)
            images      = poi_data['image'].to(device, non_blocking=True) if isinstance(poi_data['image'], torch.Tensor) else poi_data['image']
            geom_images = poi_data['geom_image'].to(device, non_blocking=True)
            texts       = poi_data['text']

            poi_features = multimodal_encoder(geom_images=geom_images, images=images, texts=texts)
            if poi_features.shape[0] < GROUP_SIZE * 2: continue

            anchor_indices = _select_spatial_group_indices(coords, GROUP_SIZE)
            if anchor_indices is None:
                continue

            anchor_feats  = poi_features[anchor_indices]
            anchor_coords = coords[anchor_indices]
            dist_anchor   = haversine_matrix_torch(anchor_coords)
            anchor_group  = group_encoder(anchor_feats.unsqueeze(0), dist_anchor)
            positive_group = _build_positive_group(anchor_feats, dist_anchor, group_encoder)

            neg_groups_list = _build_negative_groups_from_batch(
                poi_features,
                coords,
                group_encoder,
                anchor_indices,
            )

            anchor_poi_id = poi_data['poi_id'][int(anchor_indices[0].item())]
            anchor_full_idx = train_dataset.data[train_dataset.data['RestaurantID'].astype(str) == str(anchor_poi_id)].index
            
            if len(anchor_full_idx) > 0:
                idx = int(anchor_full_idx[0])
                nearest_voids = train_dataset.get_nearest_void_data(idx, k=16)
                if len(nearest_voids) >= GROUP_SIZE:
                    void_neg_groups = _encode_neighbor_groups(nearest_voids, multimodal_encoder, group_encoder, device, transform)
                    neg_groups_list.extend(void_neg_groups)

            if len(neg_groups_list) > 0:
                negative_groups = torch.cat(neg_groups_list, dim=0)
                
                if anchor_group.dim() == 1: anchor_group = anchor_group.unsqueeze(0)
                anchor_group = F.normalize(anchor_group, p=2, dim=1)
                if positive_group.dim() == 1: positive_group = positive_group.unsqueeze(0)
                positive_group = F.normalize(positive_group, p=2, dim=1)
                if negative_groups.dim() == 1: negative_groups = negative_groups.unsqueeze(0)
                negative_groups = F.normalize(negative_groups, p=2, dim=1)
                
                loss = criterion(anchor_group, positive_group, negative_groups)
                pos_sim, neg_sim = _contrast_stats(anchor_group, positive_group, negative_groups)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_train_pos_sim += pos_sim
                total_train_neg_sim += neg_sim
                num_train_batches += 1
                train_bar.set_postfix({
                    "Loss": f"{loss.item():.6f}",
                    "PosSim": f"{pos_sim:.4f}",
                    "NegSim": f"{neg_sim:.4f}"
                })

        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        avg_train_pos_sim = total_train_pos_sim / max(num_train_batches, 1)
        avg_train_neg_sim = total_train_neg_sim / max(num_train_batches, 1)

        # ---- VALIDATION ----
        multimodal_encoder.eval()
        group_encoder.eval()
        total_test_loss, num_test_batches = 0.0, 0
        total_test_pos_sim, total_test_neg_sim = 0.0, 0.0

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Test]", ncols=100)
            for batch in test_bar:
                poi_data = batch['poi']
                coords      = poi_data['coords'].to(device, non_blocking=True)
                images      = poi_data['image'].to(device, non_blocking=True) if isinstance(poi_data['image'], torch.Tensor) else poi_data['image']
                geom_images = poi_data['geom_image'].to(device, non_blocking=True)
                texts       = poi_data['text']

                poi_features = multimodal_encoder(geom_images=geom_images, images=images, texts=texts)
                if poi_features.shape[0] < GROUP_SIZE * 2: continue

                anchor_indices = _select_spatial_group_indices(coords, GROUP_SIZE)
                if anchor_indices is None:
                    continue

                anchor_feats  = poi_features[anchor_indices]
                anchor_coords = coords[anchor_indices]
                dist_anchor   = haversine_matrix_torch(anchor_coords)
                anchor_group  = group_encoder(anchor_feats.unsqueeze(0), dist_anchor)
                positive_group = _build_positive_group(anchor_feats, dist_anchor, group_encoder)

                neg_groups_list = _build_negative_groups_from_batch(
                    poi_features,
                    coords,
                    group_encoder,
                    anchor_indices,
                )

                if len(neg_groups_list) > 0:
                    negative_groups = torch.cat(neg_groups_list, dim=0)
                    
                    if anchor_group.dim() == 1: anchor_group = anchor_group.unsqueeze(0)
                    anchor_group = F.normalize(anchor_group, p=2, dim=1)
                    if positive_group.dim() == 1: positive_group = positive_group.unsqueeze(0)
                    positive_group = F.normalize(positive_group, p=2, dim=1)
                    if negative_groups.dim() == 1: negative_groups = negative_groups.unsqueeze(0)
                    negative_groups = F.normalize(negative_groups, p=2, dim=1)
                    
                    test_loss = criterion(anchor_group, positive_group, negative_groups)
                    pos_sim, neg_sim = _contrast_stats(anchor_group, positive_group, negative_groups)
                    total_test_loss += test_loss.item()
                    total_test_pos_sim += pos_sim
                    total_test_neg_sim += neg_sim
                    num_test_batches += 1
                    test_bar.set_postfix({
                        "Loss": f"{test_loss.item():.6f}",
                        "PosSim": f"{pos_sim:.4f}",
                        "NegSim": f"{neg_sim:.4f}"
                    })

        avg_test_loss = total_test_loss / max(num_test_batches, 1)
        avg_test_pos_sim = total_test_pos_sim / max(num_test_batches, 1)
        avg_test_neg_sim = total_test_neg_sim / max(num_test_batches, 1)
        tr_sil, tr_r5 = _compute_epoch_retrieval_metrics(train_dataset, multimodal_encoder, group_encoder, device)
        te_sil, te_r5 = _compute_epoch_retrieval_metrics(test_dataset, multimodal_encoder, group_encoder, device)

        print(f"🔄 Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | LR: {current_lr:.6f} | GG Maps Loss: {avg_train_loss:.6f} | Foody Loss: {avg_test_loss:.6f}")
        print(f"   ► Contrast Stats : Train Pos={avg_train_pos_sim:.4f}, Train Neg={avg_train_neg_sim:.4f} | Test Pos={avg_test_pos_sim:.4f}, Test Neg={avg_test_neg_sim:.4f}")
        print(f"   ► TRAIN (GG Maps): Silhouette={tr_sil:.4f}, Recall@5={tr_r5:.4f}")
        print(f"   ► TEST  (Foody)  : Silhouette={te_sil:.4f}, Recall@5={te_r5:.4f}")

        scheduler.step()

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            # LƯU VÀO THƯ MỤC CỦA VERSION TƯƠNG ỨNG
            torch.save(multimodal_encoder.state_dict(), f"{out_paths['models']}/multimodal_best.pth")
            torch.save(group_encoder.state_dict(),      f"{out_paths['models']}/group_encoder_best.pth")
            print(f"   🌟 Đã lưu Best Model tại Epoch {best_epoch}!")

        log_data.append({
            "epoch": epoch + 1, "learning_rate": current_lr,
            "train_loss": avg_train_loss, "train_silhouette": tr_sil, "train_recall_5": tr_r5,
            "test_loss": avg_test_loss,   "test_silhouette": te_sil,  "test_recall_5": te_r5,
            "train_pos_sim": avg_train_pos_sim, "train_neg_sim": avg_train_neg_sim,
            "test_pos_sim": avg_test_pos_sim, "test_neg_sim": avg_test_neg_sim,
        })

    # LƯU BẢNG LOG VÀO THƯ MỤC VERSION TƯƠNG ỨNG
    df_loss = pd.DataFrame(log_data)
    df_loss['is_best_model'] = ["Yes" if e == best_epoch else "No" for e in df_loss['epoch']]
    csv_path = f"{out_paths['metrics']}/training_loss_v{version}.csv"
    df_loss.to_csv(csv_path, index=False)
    
    print(f"\n✅ HOÀN TẤT HUẤN LUYỆN ZERO-SHOT VERSION {version}!")
    print(f"📊 Bảng dữ liệu đã lưu tại: {csv_path}")

    del multimodal_encoder
    del group_encoder
    del optimizer
    del criterion
    del train_loader
    del test_loader
    gc.collect()
    torch.cuda.empty_cache()

def plot_training_loss(version, out_paths):
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        sns.set_theme(style="whitegrid")
        df_loss = pd.read_csv(f"{out_paths['metrics']}/training_loss_v{version}.csv")
        
        plt.figure(figsize=(9, 5))
        plt.plot(df_loss['epoch'], df_loss['train_loss'], marker='o', color='#d62728', label='GG Maps (Train) Loss')
        plt.plot(df_loss['epoch'], df_loss['test_loss'],  marker='s', color='#1f77b4', label='Foody (Test) Loss', linestyle='--')
        
        best_row = df_loss[df_loss['is_best_model'] == 'Yes'].iloc[0]
        plt.scatter(best_row['epoch'], best_row['test_loss'], color='gold', s=200, edgecolors='black', marker='*', zorder=5, label=f'Best Epoch ({int(best_row["epoch"])})')
        
        plt.title(f"Zero-shot Domain Adaptation Curve – {VERSION_DESC[version]}")
        plt.xlabel("Epoch")
        plt.ylabel("InfoNCE Loss")
        plt.legend()
        plt.savefig(f"{out_paths['figures']}/loss_curve_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e: print(f"⚠️ Lỗi vẽ biểu đồ: {e}")

def plot_silhouette_scores(version, out_paths):
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        sns.set_theme(style="whitegrid")
        df_metrics = pd.read_csv(f"{out_paths['metrics']}/training_loss_v{version}.csv")

        plt.figure(figsize=(9, 5))
        plt.plot(df_metrics['epoch'], df_metrics['train_silhouette'], marker='o', color='#2ca02c', label='GG Maps (Train)')
        plt.plot(df_metrics['epoch'], df_metrics['test_silhouette'], marker='s', color='#ff7f0e', linestyle='--', label='Foody (Test)')

        best_row = df_metrics.loc[df_metrics['test_silhouette'].idxmax()]
        plt.scatter(best_row['epoch'], best_row['test_silhouette'], color='gold', s=180, edgecolors='black', marker='*', zorder=5, label=f'Best Test Silhouette ({int(best_row["epoch"])})')

        plt.axhline(0.0, color='gray', linewidth=1, linestyle=':')
        plt.title(f"Silhouette Score by Epoch - {VERSION_DESC[version]}")
        plt.xlabel("Epoch")
        plt.ylabel("Silhouette Score")
        plt.legend()
        plt.savefig(f"{out_paths['figures']}/silhouette_curve_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"âš ï¸ Lá»—i váº½ Silhouette Score plot: {e}")

def plot_tsne_clusters(version, out_paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalEncoder(version=version).to(device)
    try: 
        model.load_state_dict(torch.load(f"{out_paths['models']}/multimodal_best.pth", map_location=device))
    except: pass
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_dataset = POIDataset(csv_file=TEST_CSV, image_transform=transform, image_dir=TEST_IMAGE_DIR, geom_image_dir=TEST_GEOM_DIR)
    tsne_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=DATA_LOADER_WORKERS)

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
            
            cats = poi_data.get('Category', poi_data.get('category', 'Unknown'))
            all_categories.extend(cats)

    if all_features:
        tsne_results = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(torch.cat(all_features, dim=0).numpy())
        df_plot = pd.DataFrame({'tsne_x': tsne_results[:, 0], 'tsne_y': tsne_results[:, 1], 'Category': all_categories})
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='tsne_x', y='tsne_y', hue='Category', data=df_plot, legend="full", alpha=0.8)
        plt.title(f"Phân cụm t-SNE Foody (Zero-shot) – {VERSION_DESC[version]}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f"{out_paths['figures']}/tsne_clusters_v{version}.jpg", dpi=300, bbox_inches='tight')
        plt.close()

def plot_group_tsne_clusters(version, out_paths):
    try:
        features, categories = _collect_group_embeddings(version, out_paths)
        if len(features) == 0:
            return

        perplexity = max(5, min(30, len(features) - 1))
        tsne_results = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(features)
        df_plot = pd.DataFrame({'tsne_x': tsne_results[:, 0], 'tsne_y': tsne_results[:, 1], 'Category': categories})
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='tsne_x', y='tsne_y', hue='Category', data=df_plot, legend="full", alpha=0.85)
        plt.title(f"t-SNE Group Embeddings on Foody - {VERSION_DESC[version]}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f"{out_paths['figures']}/tsne_group_clusters_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"âš ï¸ Lá»—i váº½ t-SNE group embeddings: {e}")

def plot_umap_clusters(version, out_paths):
    try:
        import umap
    except ImportError:
        print("âš ï¸ ChÆ°a cÃ i `umap-learn`, bá» qua UMAP visualization.")
        return

    try:
        features, categories = _collect_group_embeddings(version, out_paths)
        if len(features) < 2:
            return

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, len(features) - 1)),
            min_dist=0.15,
            metric='cosine',
            random_state=42
        )
        umap_results = reducer.fit_transform(features)
        df_plot = pd.DataFrame({'umap_x': umap_results[:, 0], 'umap_y': umap_results[:, 1], 'Category': categories})
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='umap_x', y='umap_y', hue='Category', data=df_plot, legend="full", alpha=0.88)
        plt.title(f"UMAP Group Embeddings on Foody - {VERSION_DESC[version]}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f"{out_paths['figures']}/umap_clusters_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"âš ï¸ Lá»—i váº½ UMAP: {e}")

def evaluate_embeddings(features, labels, k=5):
    features, labels = _prepare_labeled_embeddings(features, labels, min_samples_per_class=2, drop_unknown=True)
    if len(features) < 2 or len(np.unique(labels)) < 2:
        return 0.0, 0.0

    sil_score = 0.0
    try:
        if len(np.unique(labels)) < len(labels):
            sil_score = silhouette_score(features, labels, metric='cosine')
    except Exception as e:
        print(f"âš ï¸ Lá»—i khi tÃ­nh Silhouette: {e}")
        sil_score = 0.0

    recall_at_k = 0.0
    try:
        n_neighbors = min(k + 1, len(features))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(features)
        _, indices = nn.kneighbors(features)

        hits = 0
        for i in range(len(features)):
            query_label = labels[i]
            neighbor_labels = [labels[j] for j in indices[i][1:]]
            if query_label in neighbor_labels:
                hits += 1

        recall_at_k = hits / len(features)
    except Exception as e:
        print(f"âš ï¸ Lá»—i khi tÃ­nh Recall: {e}")
        recall_at_k = 0.0

    return sil_score, recall_at_k

def plot_training_loss(version, out_paths):
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        sns.set_theme(style="whitegrid")
        df_loss = pd.read_csv(f"{out_paths['metrics']}/training_loss_v{version}.csv")

        plt.figure(figsize=(9, 5))
        plt.plot(df_loss['epoch'], df_loss['train_loss'], marker='o', color='#d62728', label='GG Maps (Train) Loss')
        plt.plot(df_loss['epoch'], df_loss['test_loss'], marker='s', color='#1f77b4', linestyle='--', label='Foody (Test) Loss')

        best_row = df_loss[df_loss['is_best_model'] == 'Yes'].iloc[0]
        plt.scatter(best_row['epoch'], best_row['test_loss'], color='gold', s=200, edgecolors='black', marker='*', zorder=5, label=f'Best Epoch ({int(best_row["epoch"])})')

        train_min = df_loss['train_loss'].replace(0, np.nan).min()
        if pd.notna(train_min) and train_min > 0:
            loss_ratio = df_loss['test_loss'].max() / train_min
            if loss_ratio > 100:
                plt.yscale('symlog', linthresh=1e-4)

        plt.title(f"Zero-shot Domain Adaptation Curve â€“ {VERSION_DESC[version]}")
        plt.xlabel("Epoch")
        plt.ylabel("InfoNCE Loss")
        plt.legend()
        plt.savefig(f"{out_paths['figures']}/loss_curve_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"âš ï¸ Lá»—i váº½ biá»ƒu Ä‘á»“ loss: {e}")

def plot_recall_at_5(version, out_paths):
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        sns.set_theme(style="whitegrid")
        df_metrics = pd.read_csv(f"{out_paths['metrics']}/training_loss_v{version}.csv")

        plt.figure(figsize=(9, 5))
        plt.plot(df_metrics['epoch'], df_metrics['train_recall_5'], marker='o', color='#9467bd', label='GG Maps (Train)')
        plt.plot(df_metrics['epoch'], df_metrics['test_recall_5'], marker='s', color='#17becf', linestyle='--', label='Foody (Test)')

        best_row = df_metrics.loc[df_metrics['test_recall_5'].idxmax()]
        plt.scatter(best_row['epoch'], best_row['test_recall_5'], color='gold', s=180, edgecolors='black', marker='*', zorder=5, label=f'Best Test Recall@5 ({int(best_row["epoch"])})')

        plt.ylim(0.0, 1.05)
        plt.title(f"Recall@5 by Epoch - {VERSION_DESC[version]}")
        plt.xlabel("Epoch")
        plt.ylabel("Recall@5")
        plt.legend()
        plt.savefig(f"{out_paths['figures']}/recall_at_5_curve_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"âš ï¸ Lá»—i váº½ Recall@5 plot: {e}")

def plot_similarity_curves(version, out_paths):
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        sns.set_theme(style="whitegrid")
        df_metrics = pd.read_csv(f"{out_paths['metrics']}/training_loss_v{version}.csv")

        plt.figure(figsize=(10, 5.5))
        plt.plot(df_metrics['epoch'], df_metrics['train_pos_sim'], marker='o', color='#2ca02c', label='Train PosSim')
        plt.plot(df_metrics['epoch'], df_metrics['train_neg_sim'], marker='o', color='#d62728', linestyle='--', label='Train NegSim')
        plt.plot(df_metrics['epoch'], df_metrics['test_pos_sim'], marker='s', color='#1f77b4', label='Test PosSim')
        plt.plot(df_metrics['epoch'], df_metrics['test_neg_sim'], marker='s', color='#ff7f0e', linestyle='--', label='Test NegSim')

        best_row = df_metrics[df_metrics['is_best_model'] == 'Yes'].iloc[0]
        plt.axvline(best_row['epoch'], color='goldenrod', linewidth=1.25, linestyle=':', label=f'Best Epoch ({int(best_row["epoch"])})')
        plt.axhline(0.0, color='gray', linewidth=1, linestyle=':')

        plt.title(f"Positive vs Negative Similarity - {VERSION_DESC[version]}")
        plt.xlabel("Epoch")
        plt.ylabel("Cosine Similarity")
        plt.legend(ncol=2)
        plt.savefig(f"{out_paths['figures']}/pos_neg_similarity_curve_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"âš ï¸ Lá»—i váº½ PosSim/NegSim plot: {e}")

def plot_group_tsne_clusters(version, out_paths):
    try:
        features, categories = _collect_group_embeddings(version, out_paths)
        features, categories = _prepare_labeled_embeddings(features, categories, min_samples_per_class=2, drop_unknown=True)
        if len(features) < 3:
            return

        features = _prepare_projection_features(features)
        perplexity = max(5, min(30, len(features) - 1))
        tsne_results = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init='pca',
            learning_rate='auto'
        ).fit_transform(features)

        df_plot = pd.DataFrame({'tsne_x': tsne_results[:, 0], 'tsne_y': tsne_results[:, 1], 'Category': categories})
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='tsne_x', y='tsne_y', hue='Category', data=df_plot, legend="full", alpha=0.85)
        plt.title(f"t-SNE Group Embeddings on Foody - {VERSION_DESC[version]}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f"{out_paths['figures']}/tsne_group_clusters_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"âš ï¸ Lá»—i váº½ t-SNE group embeddings: {e}")

def plot_umap_clusters(version, out_paths):
    try:
        import umap
    except ImportError:
        print("âš ï¸ ChÆ°a cÃ i `umap-learn`, bá» qua UMAP visualization.")
        return

    try:
        features, categories = _collect_group_embeddings(version, out_paths)
        features, categories = _prepare_labeled_embeddings(features, categories, min_samples_per_class=2, drop_unknown=True)
        if len(features) < 3:
            return

        features = _prepare_projection_features(features)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, len(features) - 1)),
            min_dist=0.15,
            metric='cosine',
            random_state=42
        )
        umap_results = reducer.fit_transform(features)

        df_plot = pd.DataFrame({'umap_x': umap_results[:, 0], 'umap_y': umap_results[:, 1], 'Category': categories})
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='umap_x', y='umap_y', hue='Category', data=df_plot, legend="full", alpha=0.88)
        plt.title(f"UMAP Group Embeddings on Foody - {VERSION_DESC[version]}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f"{out_paths['figures']}/umap_clusters_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"âš ï¸ Lá»—i váº½ UMAP: {e}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    # CHẠY LẦN LƯỢT CÁC VERSION ĐƯỢC CHỈ ĐỊNH TẠI VERSIONS_TO_TRAIN
    for v in VERSIONS_TO_TRAIN:
        print(f"\n{'='*80}")
        print(f"🚀 BẮT ĐẦU CHUỖI HUẤN LUYỆN ĐỘC LẬP CHO VERSION {v}")
        print(f"{'='*80}")
        
        # 1. Tạo thư mục chứa riêng cho Version này
        paths = get_output_paths(v)
        
        # 2. Train và lưu vào thư mục riêng
        train_urban_ai(v, paths)
        
        # 3. Trực quan hóa và lưu vào thư mục riêng
        print(f"🎨 Đang vẽ biểu đồ kết quả cho Version {v}...")
        plot_training_loss(v, paths)
        plot_silhouette_scores(v, paths)
        plot_recall_at_5(v, paths)
        plot_similarity_curves(v, paths)
        plot_tsne_clusters(v, paths)
        plot_group_tsne_clusters(v, paths)
        plot_umap_clusters(v, paths)
        
        print(f"📦 Toàn bộ báo cáo, file csv, hình ảnh và Model V{v} đã được đóng gói tại: results/v{v}/")
