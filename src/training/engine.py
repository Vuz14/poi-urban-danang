import torch
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import gc
from tqdm import tqdm

from config import *
from src.data.dataset import POIDataset, custom_collate_fn
from src.encoder.multimodal_encoder import MultimodalEncoder, VERSION_DESC
from src.models.building_group import BuildingGroupEncoder
from src.models.loss_functions import MarginInfoNCELoss
from utlis.geo_utils import _select_spatial_group_indices, haversine_matrix_torch, _build_positive_group, _build_negative_groups_from_batch
from research_pipeline.evaluator import _compute_epoch_retrieval_metrics

def _contrast_stats(anchor_group, positive_group, negative_groups):
    pos_sim = F.cosine_similarity(anchor_group, positive_group, dim=-1).mean().item()
    neg_sim = F.cosine_similarity(anchor_group.unsqueeze(1), negative_groups.unsqueeze(0), dim=-1).mean().item()
    return pos_sim, neg_sim

def train_urban_ai(version, out_paths):
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

    multimodal_encoder = MultimodalEncoder(embed_dim=EMBEDDING_DIM, version=version).to(device)
    group_encoder      = BuildingGroupEncoder(embed_dim=EMBEDDING_DIM, num_heads=4).to(device)
    criterion = MarginInfoNCELoss(temperature=TEMPERATURE, margin=MARGIN).to(device)

    optim_params = [
        {'params': multimodal_encoder.resnet.parameters(), 'lr': LR * 0.1},
        {'params': group_encoder.parameters(),             'lr': LR},
    ]

    if hasattr(multimodal_encoder, 'clip_model'):
        for param in multimodal_encoder.clip_model.parameters(): param.requires_grad = False
    if hasattr(multimodal_encoder, 'text_projection'):
        for param in multimodal_encoder.text_projection.parameters(): param.requires_grad = True
        optim_params.append({'params': multimodal_encoder.text_projection.parameters(), 'lr': LR})
    if hasattr(multimodal_encoder, 'image_projection'):
        for param in multimodal_encoder.image_projection.parameters(): param.requires_grad = True
        optim_params.append({'params': multimodal_encoder.image_projection.parameters(), 'lr': LR})
    if hasattr(multimodal_encoder, 'fusion_proj'):
        optim_params.append({'params': multimodal_encoder.fusion_proj.parameters(), 'lr': LR})
    if hasattr(multimodal_encoder, 'gate'):
        optim_params.append({'params': multimodal_encoder.gate.parameters(), 'lr': LR * 0.5})

    optimizer = optim.Adam(optim_params, weight_decay=1e-4)
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
            if anchor_indices is None: continue

            anchor_feats  = poi_features[anchor_indices]
            anchor_coords = coords[anchor_indices]
            dist_anchor   = haversine_matrix_torch(anchor_coords)
            anchor_group  = group_encoder(anchor_feats.unsqueeze(0), dist_anchor)
            positive_group = _build_positive_group(anchor_feats, dist_anchor, group_encoder)

            neg_groups_list = _build_negative_groups_from_batch(poi_features, coords, group_encoder, anchor_indices)

            # Tắt phần quét Pandas (Hard Negative) để giải phóng CPU/IO
            # anchor_poi_id = poi_data['poi_id'][int(anchor_indices[0].item())]
            # anchor_full_idx = train_dataset.data[train_dataset.data['RestaurantID'].astype(str) == str(anchor_poi_id)].index
            # if len(anchor_full_idx) > 0:
            #     idx = int(anchor_full_idx[0])
            #     nearest_voids = train_dataset.get_nearest_void_data(idx, k=16)
            #     if len(nearest_voids) >= GROUP_SIZE:
            #         void_neg_groups = _encode_neighbor_groups(nearest_voids, multimodal_encoder, group_encoder, device, transform)
            #         neg_groups_list.extend(void_neg_groups)

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
                train_bar.set_postfix({"Loss": f"{loss.item():.6f}", "PosSim": f"{pos_sim:.4f}", "NegSim": f"{neg_sim:.4f}"})

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
                if anchor_indices is None: continue

                anchor_feats  = poi_features[anchor_indices]
                anchor_coords = coords[anchor_indices]
                dist_anchor   = haversine_matrix_torch(anchor_coords)
                anchor_group  = group_encoder(anchor_feats.unsqueeze(0), dist_anchor)
                positive_group = _build_positive_group(anchor_feats, dist_anchor, group_encoder)

                neg_groups_list = _build_negative_groups_from_batch(poi_features, coords, group_encoder, anchor_indices)

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
                    test_bar.set_postfix({"Loss": f"{test_loss.item():.6f}", "PosSim": f"{pos_sim:.4f}", "NegSim": f"{neg_sim:.4f}"})

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

    df_loss = pd.DataFrame(log_data)
    df_loss['is_best_model'] = ["Yes" if e == best_epoch else "No" for e in df_loss['epoch']]
    csv_path = f"{out_paths['metrics']}/training_loss_v{version}.csv"
    df_loss.to_csv(csv_path, index=False)
    
    print(f"\n✅ HOÀN TẤT HUẤN LUYỆN ZERO-SHOT VERSION {version}!")
    print(f"📊 Bảng dữ liệu đã lưu tại: {csv_path}")

    del multimodal_encoder, group_encoder, optimizer, criterion, train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()