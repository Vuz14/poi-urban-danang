import numpy as np
import torch
import math
import pandas as pd
from config import POSITIVE_NOISE_STD, POSITIVE_FEATURE_DROPOUT, GROUP_SIZE

def haversine_matrix_torch(coords):
    """Tính ma trận khoảng cách Haversine (tính bằng mét) cho một tensor tọa độ."""
    coords_rad = torch.deg2rad(coords)
    lat = coords_rad[:, 0]
    lon = coords_rad[:, 1]
    dlat = lat.unsqueeze(1) - lat.unsqueeze(0)
    dlon = lon.unsqueeze(1) - lon.unsqueeze(0)
    a = torch.sin(dlat / 2)**2 + torch.cos(lat.unsqueeze(1)) * torch.cos(lat.unsqueeze(0)) * torch.sin(dlon / 2)**2
    c = 2 * torch.asin(torch.sqrt(a + 1e-10))
    r = 6371000 
    return r * c

def simple_poisson_disk_sampling(width, height, radius=2000, k=30):
    """Thuật toán Bridson cho Poisson Disk Sampling 2D cơ bản."""
    cell_size = radius / math.sqrt(2)
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    points = []
    spawn_points = []
    spawn_points.append((np.random.uniform(0, width), np.random.uniform(0, height)))
    
    while spawn_points:
        spawn_idx = np.random.randint(0, len(spawn_points))
        spawn_center = spawn_points[spawn_idx]
        accepted = False
        for _ in range(k):
            angle = 2 * math.pi * np.random.uniform(0, 1)
            r = np.random.uniform(radius, 2 * radius)
            px, py = spawn_center[0] + r * math.cos(angle), spawn_center[1] + r * math.sin(angle)
            if 0 <= px < width and 0 <= py < height:
                g_x, g_y = int(px / cell_size), int(py / cell_size)
                valid = True
                for i in range(max(0, g_x - 1), min(grid_width, g_x + 2)):
                    for j in range(max(0, g_y - 1), min(grid_height, g_y + 2)):
                        if grid[i][j] is not None:
                            dist = math.hypot(grid[i][j][0] - px, grid[i][j][1] - py)
                            if dist < radius:
                                valid = False
                                break
                    if not valid: break
                if valid:
                    points.append((px, py))
                    spawn_points.append((px, py))
                    grid[g_x][g_y] = (px, py)
                    accepted = True
                    break
        if not accepted:
            spawn_points.pop(spawn_idx)
    return np.array(points)

def _normalize_label(label):
    if pd.isna(label): return "Unknown"
    clean_label = str(label).strip()
    if not clean_label or clean_label.lower() in {"nan", "none", "null", "unknown"}:
        return "Unknown"
    return clean_label

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
            coords, GROUP_SIZE, candidate_indices=remaining, reference_coord=anchor_centroid
        )
        if neg_indices is None: break

        neg_feats = poi_features[neg_indices]
        neg_coords = coords[neg_indices]
        dist_neg = haversine_matrix_torch(neg_coords)
        neg_group = group_encoder(neg_feats.unsqueeze(0), dist_neg)
        neg_groups_list.append(neg_group)
        remaining = remaining[~torch.isin(remaining, neg_indices)]
    return neg_groups_list

def _build_positive_group(anchor_feats, dist_anchor, group_encoder, noise_std=POSITIVE_NOISE_STD):
    noisy_feats = anchor_feats + torch.randn_like(anchor_feats) * noise_std
    dropout_mask = (torch.rand_like(noisy_feats) > POSITIVE_FEATURE_DROPOUT).float()
    noisy_feats = noisy_feats * dropout_mask
    return group_encoder(noisy_feats.unsqueeze(0), dist_anchor)

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

def _resolve_group_label_info(categories, group_indices):
    if isinstance(categories, str):
        label = _normalize_label(categories)
        purity = 1.0 if label != "Unknown" else 0.0
        return label, purity
    labels = [_normalize_label(categories[int(idx)]) for idx in group_indices.detach().cpu().tolist()]
    valid_labels = [label for label in labels if label != "Unknown"]
    if not valid_labels: return "Unknown", 0.0
    label_counts = pd.Series(valid_labels).value_counts()
    majority_label = label_counts.index[0]
    purity = float(label_counts.iloc[0] / len(labels))
    return majority_label, purity

def _resolve_group_label(categories, group_indices):
    label, _ = _resolve_group_label_info(categories, group_indices)
    return label