import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

def train_on_google_maps(train_dataloader, save_path="models/gmap_model"):
    """Huấn luyện mô hình sử dụng Google Maps data với Contrastive Loss"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,
        warmup_steps=100,
        output_path=save_path
    )
    return model

def extract_cross_domain_embeddings(model_path, gmap_texts, foody_texts):
    """Đóng băng model và trích xuất embedding cho cả 2 Domain"""
    model = SentenceTransformer(model_path)
    
    # Freeze model weights (Zero-shot cho Foody)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    with torch.no_grad():
        gmap_embeddings = model.encode(gmap_texts, convert_to_tensor=True)
        foody_embeddings = model.encode(foody_texts, convert_to_tensor=True)
        
    return gmap_embeddings.cpu().numpy(), foody_embeddings.cpu().numpy()