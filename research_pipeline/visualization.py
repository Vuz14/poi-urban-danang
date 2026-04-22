import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader

from config import *
from src.encoder.multimodal_encoder import MultimodalEncoder, VERSION_DESC
from src.data.dataset import POIDataset, custom_collate_fn
from research_pipeline.evaluator  import _collect_group_embeddings, _prepare_labeled_embeddings, _prepare_projection_features

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

        plt.title(f"Zero-shot Domain Adaptation Curve – {VERSION_DESC[version]}")
        plt.xlabel("Epoch")
        plt.ylabel("InfoNCE Loss")
        plt.legend()
        plt.savefig(f"{out_paths['figures']}/loss_curve_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e: print(f"⚠️ Lỗi vẽ biểu đồ loss: {e}")

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
    except Exception as e: print(f"⚠️ Lỗi vẽ Silhouette Score plot: {e}")

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
    except Exception as e: print(f"⚠️ Lỗi vẽ Recall@5 plot: {e}")

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
    except Exception as e: print(f"⚠️ Lỗi vẽ PosSim/NegSim plot: {e}")

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
        features, categories = _prepare_labeled_embeddings(features, categories, min_samples_per_class=2, drop_unknown=True)
        if len(features) < 3: return

        features = _prepare_projection_features(features)
        perplexity = max(5, min(30, len(features) - 1))
        tsne_results = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto').fit_transform(features)

        df_plot = pd.DataFrame({'tsne_x': tsne_results[:, 0], 'tsne_y': tsne_results[:, 1], 'Category': categories})
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='tsne_x', y='tsne_y', hue='Category', data=df_plot, legend="full", alpha=0.85)
        plt.title(f"t-SNE Group Embeddings on Foody - {VERSION_DESC[version]}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f"{out_paths['figures']}/tsne_group_clusters_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e: print(f"⚠️ Lỗi vẽ t-SNE group embeddings: {e}")

def plot_umap_clusters(version, out_paths):
    try: import umap
    except ImportError: return

    try:
        features, categories = _collect_group_embeddings(version, out_paths)
        features, categories = _prepare_labeled_embeddings(features, categories, min_samples_per_class=2, drop_unknown=True)
        if len(features) < 3: return

        features = _prepare_projection_features(features)
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, max(2, len(features) - 1)), min_dist=0.15, metric='cosine', random_state=42)
        umap_results = reducer.fit_transform(features)

        df_plot = pd.DataFrame({'umap_x': umap_results[:, 0], 'umap_y': umap_results[:, 1], 'Category': categories})
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='umap_x', y='umap_y', hue='Category', data=df_plot, legend="full", alpha=0.88)
        plt.title(f"UMAP Group Embeddings on Foody - {VERSION_DESC[version]}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f"{out_paths['figures']}/umap_clusters_v{version}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e: print(f"⚠️ Lỗi vẽ UMAP: {e}")