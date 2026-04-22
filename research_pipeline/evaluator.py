import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gc
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from config import *
from src.data.dataset import POIDataset, custom_collate_fn
from src.encoder.multimodal_encoder import MultimodalEncoder
from src.models.building_group import BuildingGroupEncoder
from utlis.geo_utils import _normalize_label, _select_spatial_group_indices, haversine_matrix_torch, _resolve_group_label_info

def _prepare_labeled_embeddings(features, labels, min_samples_per_class=2, drop_unknown=True):
    if len(features) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), np.array([], dtype=object)
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray([_normalize_label(label) for label in labels], dtype=object)
    if features.ndim == 1: features = features.reshape(1, -1)
    
    valid_mask = ~np.isnan(features).any(axis=1)
    if drop_unknown: valid_mask &= labels != "Unknown"
    
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
    if len(features) < 2: return features
    scaled_features = StandardScaler().fit_transform(features)
    pca_dims = min(32, scaled_features.shape[0] - 1, scaled_features.shape[1])
    if pca_dims >= 2 and pca_dims < scaled_features.shape[1]:
        scaled_features = PCA(n_components=pca_dims, random_state=42).fit_transform(scaled_features)
    return scaled_features

def evaluate_embeddings(features, labels, k=5):
    features, labels = _prepare_labeled_embeddings(features, labels, min_samples_per_class=2, drop_unknown=True)
    if len(features) < 2 or len(np.unique(labels)) < 2: return 0.0, 0.0

    sil_score = 0.0
    try:
        if len(np.unique(labels)) < len(labels):
            sil_score = silhouette_score(features, labels, metric='cosine')
    except Exception as e:
        print(f"⚠️ Lỗi khi tính Silhouette: {e}")
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
            if query_label in neighbor_labels: hits += 1
        recall_at_k = hits / len(features)
    except Exception as e:
        print(f"⚠️ Lỗi khi tính Recall: {e}")
        recall_at_k = 0.0

    return sil_score, recall_at_k

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
            if poi_features.shape[0] < GROUP_SIZE: continue

            cats = poi_data.get('Category', poi_data.get('category', 'Unknown'))
            remaining = torch.arange(poi_features.shape[0], device=device)
            while remaining.numel() >= GROUP_SIZE:
                group_indices = _select_spatial_group_indices(coords, GROUP_SIZE, candidate_indices=remaining)
                if group_indices is None: break

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

    if not all_features: return np.empty((0, EMBEDDING_DIM)), []
    return torch.stack(all_features).numpy(), all_categories

def _compute_epoch_retrieval_metrics(dataset, multimodal_encoder, group_encoder, device):
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=DATA_LOADER_WORKERS)
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
    all_features, all_categories = _collect_group_embeddings_from_loader(loader, multimodal_encoder, group_encoder, device)
    
    del multimodal_encoder, group_encoder
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return all_features, all_categories