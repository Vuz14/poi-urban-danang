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
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
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
CSV_PATH      = "/kaggle/input/datasets/nhttngy/csv-data/poi_data_ggmap_v1_filtered.csv" 
IMAGE_DIR     = r"/kaggle/input/datasets/nhttngy/images/poi_images_ggmap"
VOID_PATH     = "/kaggle/input/datasets/nhttngy/csv-data/urban_voids.csv"
GEOM_DIR      = r"/kaggle/input/datasets/nhttngy/images-moi/building_images_ggmap"
VOID_GEOM_DIR = r"/kaggle/input/datasets/nhttngy/building-image-voids/building_images_voids"

BATCH_SIZE  = 128
NUM_EPOCHS  = 10
TRAIN_RATIO = 0.8
GROUP_SIZE  = 8
TEMPERATURE = 0.05
LR          = 1e-4
HARD_NEG_K  = 4 
MAX_NEG_GROUPS = 8
NEG_HARD_RATIO = 0.25
PAIR_THRESHOLD = 0.65
SEED = 42
USE_VOID_NEGATIVES = False


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def _select_balanced_negatives(
    anchor_group,
    neg_groups_list,
    max_neg_groups=MAX_NEG_GROUPS,
    hard_ratio=NEG_HARD_RATIO,
    deterministic=False,
):
    """
    Chọn số lượng negative cố định để loss ổn định hơn giữa các batch:
    - Một phần hard negatives (gần anchor nhất theo cosine)
    - Một phần random negatives để tránh quá khó và giảm dao động.
    """
    if not neg_groups_list:
        return None

    all_negs = torch.cat(neg_groups_list, dim=0)
    total_negs = all_negs.shape[0]

    if total_negs <= max_neg_groups:
        # Giữ số lượng negatives cố định để loss ổn định giữa các batch.
        if total_negs == max_neg_groups:
            return all_negs
        if total_negs == 0:
            return None
        if deterministic:
            pad_idx = torch.arange(max_neg_groups - total_negs, device=all_negs.device) % total_negs
        else:
            pad_idx = torch.randint(0, total_negs, (max_neg_groups - total_negs,), device=all_negs.device)
        padded_negs = all_negs[pad_idx]
        return torch.cat([all_negs, padded_negs], dim=0)

    anchor_expand = anchor_group.expand(total_negs, -1)
    sims = torch.nn.functional.cosine_similarity(anchor_expand, all_negs, dim=-1)

    hard_k = max(1, int(max_neg_groups * hard_ratio))
    rand_k = max_neg_groups - hard_k

    _, hard_idx = torch.topk(sims, k=hard_k, largest=True)

    mask = torch.ones(total_negs, dtype=torch.bool, device=all_negs.device)
    mask[hard_idx] = False
    remaining_idx = torch.where(mask)[0]

    if rand_k > 0 and remaining_idx.numel() > 0:
        if deterministic:
            rand_idx = remaining_idx[:rand_k]
        else:
            perm = torch.randperm(remaining_idx.numel(), device=all_negs.device)
            rand_idx = remaining_idx[perm[:rand_k]]
        selected_idx = torch.cat([hard_idx, rand_idx], dim=0)
    else:
        selected_idx = hard_idx

    return all_negs[selected_idx]


def _build_group_embedding(group_encoder, feats, coords):
    dist = haversine_matrix_torch(coords)
    return group_encoder(feats.unsqueeze(0), dist).mean(dim=1)


def _compute_pair_metrics(anchor_group, positive_group, negative_groups, threshold=PAIR_THRESHOLD):
    """
    Tính Accuracy/Precision/Recall/F1 ở mức pair-wise binary:
    - Positive pair: (anchor, positive) có nhãn 1
    - Negative pair: (anchor, negative_i) có nhãn 0
    """
    if anchor_group.dim() == 1:
        anchor_group = anchor_group.unsqueeze(0)
    if positive_group.dim() == 1:
        positive_group = positive_group.unsqueeze(0)
    if negative_groups.dim() == 1:
        negative_groups = negative_groups.unsqueeze(0)

    sim_pos = torch.nn.functional.cosine_similarity(anchor_group, positive_group, dim=-1).reshape(-1)
    sim_neg_matrix = torch.nn.functional.cosine_similarity(
        anchor_group.unsqueeze(1),
        negative_groups.unsqueeze(0),
        dim=-1,
    )
    sim_neg = sim_neg_matrix.reshape(-1)

    scores = torch.cat([sim_pos.reshape(-1), sim_neg], dim=0)
    labels = torch.cat([
        torch.ones(sim_pos.numel(), dtype=torch.long, device=scores.device),
        torch.zeros(sim_neg.numel(), dtype=torch.long, device=scores.device),
    ], dim=0)

    preds = (scores >= threshold).long()

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())

    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg_matrix], dim=1)
    rank_correct = int((logits.argmax(dim=1) == 0).sum().item())
    rank_total = int(logits.shape[0])

    return (
        tp,
        fp,
        tn,
        fn,
        scores.detach().cpu().tolist(),
        labels.detach().cpu().tolist(),
        rank_correct,
        rank_total,
    )


def _finalize_classification_metrics(tp, fp, tn, fn, labels, scores, rank_correct, rank_total):
    total = tp + fp + tn + fn
    pair_accuracy = (tp + tn) / max(total, 1)
    rank_accuracy = rank_correct / max(rank_total, 1)
    accuracy = 0.5 * pair_accuracy + 0.5 * rank_accuracy
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    if labels and len(set(labels)) > 1:
        auc = float(roc_auc_score(labels, scores))
    else:
        auc = float("nan")

    return auc, accuracy, f1, precision, recall, specificity

# =========================================================================
# 1. HÀM HUẤN LUYỆN AI (TRAINING)
# =========================================================================
def train_urban_ai():
    print("=" * 65)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN | {VERSION_DESC[TRAINING_VERSION]}")
    print("=" * 65)

    set_seed(SEED)
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
    optimizer = optim.AdamW([
        {'params': multimodal_encoder.text_projection.parameters(),  'lr': LR},
        {'params': multimodal_encoder.image_projection.parameters(), 'lr': LR},
        {'params': multimodal_encoder.resnet.parameters(),            'lr': LR * 0.1},
        {'params': multimodal_encoder.fusion.parameters(),            'lr': LR},
        {'params': group_encoder.parameters(),                        'lr': LR},
    ], weight_decay=5e-4)
    
    criterion = InfoNCELoss(temperature=TEMPERATURE, label_smoothing=0.05).to(device)

    # Warmup ngắn + cosine decay để giảm dao động loss ở các epoch đầu.
    warmup_epochs = 2
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(NUM_EPOCHS - warmup_epochs, 1), eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    train_loss_history, test_loss_history = [], []
    train_auc_history, test_auc_history = [], []
    train_acc_history, test_acc_history = [], []
    train_f1_history, test_f1_history = [], []
    train_precision_history, test_precision_history = [], []
    train_recall_history, test_recall_history = [], []
    train_spec_history, test_spec_history = [], []
    
    # 3. Biến lưu giữ mức Test Loss tốt nhất
    best_test_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # ---- TRAINING ------------------------------------------------
        multimodal_encoder.train()
        group_encoder.train()
        total_train_loss, num_train_batches = 0.0, 0
        train_tp = train_fp = train_tn = train_fn = 0
        train_scores_epoch, train_labels_epoch = [], []
        train_rank_correct = train_rank_total = 0

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

                    usable = (poi_features.shape[0] // GROUP_SIZE) * GROUP_SIZE
                    poi_features = poi_features[:usable]
                    coords = coords[:usable]
                    poi_ids = poi_data['poi_id'][:usable]
                    num_groups = usable // GROUP_SIZE
                    if num_groups < 2:
                        continue

                    group_feats = poi_features.view(num_groups, GROUP_SIZE, -1)
                    group_coords = coords.view(num_groups, GROUP_SIZE, -1)
                    group_embeds = [
                        _build_group_embedding(group_encoder, group_feats[i], group_coords[i])
                        for i in range(num_groups)
                    ]
                    group_losses = []

                    for g in range(num_groups):
                        anchor_feats = group_feats[g]
                        anchor_coords = group_coords[g]
                        anchor_group = group_embeds[g]
                        noise = torch.randn_like(anchor_feats) * 0.03
                        positive_group = _build_group_embedding(group_encoder, anchor_feats + noise, anchor_coords)

                        # Dùng toàn bộ group còn lại trong batch làm negative.
                        # KHÔNG detach để anchor/negative cùng nhận gradient đẩy tách nhau.
                        neg_groups_list = [group_embeds[i] for i in range(num_groups) if i != g]

                        if USE_VOID_NEGATIVES:
                            anchor_poi_id = poi_ids[g * GROUP_SIZE]
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

                        if len(neg_groups_list) == 0:
                            continue

                        negative_groups = _select_balanced_negatives(
                            anchor_group, neg_groups_list, deterministic=False
                        )
                        if negative_groups is None or negative_groups.shape[0] < 2:
                            continue

                        group_loss = criterion(anchor_group, positive_group, negative_groups)
                        group_losses.append(group_loss)

                        tp, fp, tn, fn, scores, labels, rank_correct, rank_total = _compute_pair_metrics(
                            anchor_group, positive_group, negative_groups
                        )
                        train_tp += tp
                        train_fp += fp
                        train_tn += tn
                        train_fn += fn
                        train_scores_epoch.extend(scores)
                        train_labels_epoch.extend(labels)
                        train_rank_correct += rank_correct
                        train_rank_total += rank_total

                    if not group_losses:
                        continue

                    loss = torch.stack(group_losses).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(multimodal_encoder.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(group_encoder.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_train_loss += loss.item()
                    num_train_batches += 1

                    if batch_idx % 10 == 0:
                        print(f"   Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        train_loss_history.append(avg_train_loss)
        train_auc, train_acc, train_f1, train_precision, train_recall, train_spec = _finalize_classification_metrics(
            train_tp, train_fp, train_tn, train_fn,
            train_labels_epoch, train_scores_epoch,
            train_rank_correct, train_rank_total,
        )
        train_auc_history.append(train_auc)
        train_acc_history.append(train_acc)
        train_f1_history.append(train_f1)
        train_precision_history.append(train_precision)
        train_recall_history.append(train_recall)
        train_spec_history.append(train_spec)
# ---- VALIDATION ------------------------------------------------
        multimodal_encoder.eval()
        group_encoder.eval()
        total_test_loss, num_test_batches = 0.0, 0
        test_tp = test_fp = test_tn = test_fn = 0
        test_scores_epoch, test_labels_epoch = [], []
        test_rank_correct = test_rank_total = 0

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

                usable = (poi_features.shape[0] // GROUP_SIZE) * GROUP_SIZE
                poi_features = poi_features[:usable]
                coords = coords[:usable]
                poi_ids = poi_data['poi_id'][:usable]
                num_groups = usable // GROUP_SIZE
                if num_groups < 2:
                    continue

                group_feats = poi_features.view(num_groups, GROUP_SIZE, -1)
                group_coords = coords.view(num_groups, GROUP_SIZE, -1)
                group_embeds = [
                    _build_group_embedding(group_encoder, group_feats[i], group_coords[i])
                    for i in range(num_groups)
                ]
                eval_group_losses = []

                for g in range(num_groups):
                    anchor_feats = group_feats[g]
                    anchor_coords = group_coords[g]
                    anchor_group = group_embeds[g]
                    positive_group = anchor_group.clone()

                    neg_groups_list = [group_embeds[i] for i in range(num_groups) if i != g]

                    if USE_VOID_NEGATIVES:
                        anchor_poi_id = poi_ids[g * GROUP_SIZE]
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

                    if len(neg_groups_list) == 0:
                        continue

                    negative_groups = _select_balanced_negatives(
                        anchor_group, neg_groups_list, deterministic=True
                    )
                    if negative_groups is None or negative_groups.shape[0] < 2:
                        continue

                    eval_group_losses.append(criterion(anchor_group, positive_group, negative_groups))
                    tp, fp, tn, fn, scores, labels, rank_correct, rank_total = _compute_pair_metrics(
                        anchor_group, positive_group, negative_groups
                    )
                    test_tp += tp
                    test_fp += fp
                    test_tn += tn
                    test_fn += fn
                    test_scores_epoch.extend(scores)
                    test_labels_epoch.extend(labels)
                    test_rank_correct += rank_correct
                    test_rank_total += rank_total

                if eval_group_losses:
                    test_loss = torch.stack(eval_group_losses).mean()
                    total_test_loss += test_loss.item()
                    num_test_batches += 1

# ... (code validation ở trên giữ nguyên) ...
        avg_test_loss = total_test_loss / max(num_test_batches, 1)
        test_loss_history.append(avg_test_loss)
        test_auc, test_acc, test_f1, test_precision, test_recall, test_spec = _finalize_classification_metrics(
            test_tp, test_fp, test_tn, test_fn,
            test_labels_epoch, test_scores_epoch,
            test_rank_correct, test_rank_total,
        )
        test_auc_history.append(test_auc)
        test_acc_history.append(test_acc)
        test_f1_history.append(test_f1)
        test_precision_history.append(test_precision)
        test_recall_history.append(test_recall)
        test_spec_history.append(test_spec)

        print(
            f"🔄 Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
            f"Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | "
            f"Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}"
        )

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
        "epoch"          : range(1, NUM_EPOCHS + 1),
        "train_loss"     : train_loss_history,
        "train_auc"      : train_auc_history,
        "train_acc"      : train_acc_history,
        "train_f1"       : train_f1_history,
        "train_precision": train_precision_history,
        "train_recall"   : train_recall_history,
        "train_spec"     : train_spec_history,
        "test_loss"      : test_loss_history,
        "test_auc"       : test_auc_history,
        "test_acc"       : test_acc_history,
        "test_f1"        : test_f1_history,
        "test_precision" : test_precision_history,
        "test_recall"    : test_recall_history,
        "test_spec"      : test_spec_history,
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
        plt.plot(df_loss['epoch'], df_loss['train_loss'], marker='o', color='#d62728', linewidth=2.5, label='Train Loss')
        plt.plot(df_loss['epoch'], df_loss['test_loss'],  marker='s', color='#1f77b4', linewidth=2.5, label='Test Loss', linestyle='--')
        plt.title(f"Loss Curve – {VERSION_DESC[TRAINING_VERSION]}", fontsize=13, fontweight='bold')
        plt.xlabel("Số vòng học (Epoch)", fontsize=11)
        plt.ylabel("InfoNCE Loss", fontsize=11)
        plt.xticks(df_loss['epoch'])
        plt.legend()

        os.makedirs("reports/figures", exist_ok=True)
        out_path = f"reports/figures/loss_curve_v{TRAINING_VERSION}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu: {out_path}")

        metric_cols = [
            ('Loss', 'train_loss', 'test_loss'),
            ('AUC', 'train_auc', 'test_auc'),
            ('Accuracy', 'train_acc', 'test_acc'),
            ('F1-score', 'train_f1', 'test_f1'),
            ('Precision', 'train_precision', 'test_precision'),
            ('Recall', 'train_recall', 'test_recall'),
            ('Specificity', 'train_spec', 'test_spec'),
        ]

        fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
        axes = axes.flatten()

        for ax, (metric_name, train_col, test_col) in zip(axes, metric_cols):
            ax.plot(df_loss['epoch'], df_loss[train_col], marker='o', linewidth=2.0, label=f'Train {metric_name}')
            ax.plot(df_loss['epoch'], df_loss[test_col], marker='s', linewidth=2.0, linestyle='--', label=f'Test {metric_name}')
            ax.set_title(metric_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_xticks(df_loss['epoch'])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for ax in axes[len(metric_cols):]:
            ax.axis('off')

        fig.suptitle(f"All Training/Test Metrics – {VERSION_DESC[TRAINING_VERSION]}", fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        metric_path = f"reports/figures/metrics_curve_v{TRAINING_VERSION}.png"
        fig.savefig(metric_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Đã lưu: {metric_path}")
    except Exception as e:
        print(f"⚠️  Không thể vẽ biểu đồ metrics: {e}")


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