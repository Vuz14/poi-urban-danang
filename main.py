import torch
import torch.optim as optim
from src.models.building_group import BuildingGroupEncoder
from src.models.region_model import RegionEncoder
from src.models.loss_functions import InfoNCELoss, AdaptiveTripletLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Khởi chạy huấn luyện trên: {device}")
    
    # 1. Khởi tạo Models
    embed_dim = 64
    group_model = BuildingGroupEncoder(embed_dim=embed_dim).to(device)
    region_model = RegionEncoder(embed_dim=embed_dim).to(device)
    
    # 2. Loss & Optimizer
    criterion_group = InfoNCELoss(temperature=0.05)
    criterion_region = AdaptiveTripletLoss(lamba=50.0)
    
    # Khai báo optimizer gom chung tham số
    optimizer = optim.Adam(
        list(group_model.parameters()) + list(region_model.parameters()), 
        lr=1e-3
    )
    
    # 3. Dummy Data Loop (Minh họa tensor shape)
    # Giả lập 1 Region chứa 5 Building Groups, mỗi Group có 10 thực thể (Buildings/POIs)
    # Features: (1, 10, 64), Dist_matrix: (10, 10)
    
    group_features = torch.randn(5, 1, 10, 64).to(device) 
    dist_matrices = torch.abs(torch.randn(5, 10, 10)).to(device)
    
    group_model.train()
    region_model.train()
    
    optimizer.zero_grad()
    
    # --- Bước A: Trích xuất Group Embeddings ---
    group_embeddings = []
    for i in range(5):
        # Cho qua Distance-Biased Transformer
        g_emb = group_model(group_features[i], dist_matrices[i])
        group_embeddings.append(g_emb)
        
    group_tensor = torch.stack(group_embeddings, dim=1) # (1, 5, 64)
    
    # --- Bước B: Trích xuất Region Embedding ---
    region_embedding = region_model(group_tensor) # (1, 64)
    
    print("Shape của Region Representation xuất ra:", region_embedding.shape)
    print("Kiến trúc Mạng đã sẵn sàng để áp dụng Loss đa tầng.")

if __name__ == "__main__":
    main()