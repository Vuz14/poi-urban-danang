import pandas as pd
import numpy as np
import torch
import osmnx as ox
import networkx as nx
from tqdm import tqdm
import os

# 1. CẤU HÌNH ĐƯỜNG DẪN (Đã sửa theo máy của bạn)
CSV_PATH = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\processed\poi_processed_data.csv"
OUTPUT_DIR = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "street_dist_matrix.pt")

def compute_street_distances():
    # Kiểm tra file đầu vào
    if not os.path.exists(CSV_PATH):
        print(f"❌ Lỗi: Không tìm thấy file tại {CSV_PATH}")
        return

    print("🌍 Bước 2.1: Tải mạng lưới giao thông Đà Nẵng (OSM)...")
    # Tải đồ thị đường bộ toàn thành phố
    G = ox.graph_from_place('Da Nang, Vietnam', network_type='drive')
    
    print(f"📂 Bước 2.2: Đang đọc dữ liệu từ CSV...")
    df = pd.read_csv(CSV_PATH)
    n = len(df)
    
    print(f"📍 Bước 2.3: Ánh xạ {n} POIs vào nút giao thông gần nhất...")
    # Lấy tọa độ Lon/Lat từ file của bạn
    nodes = ox.distance.nearest_nodes(G, X=df['Lon'].values, Y=df['Lat'].values)
    
    # Khởi tạo ma trận (n x n)
    dist_matrix = np.zeros((n, n))
    
    print("🚗 Bước 2.4: Tính toán Dijkstra (Shortest Path)...")
    # Vòng lặp tính toán
    for i in tqdm(range(n)):
        try:
            # Tính khoảng cách từ nút i đến mọi nút khác có thể đến được
            lengths = nx.single_source_dijkstra_path_length(G, nodes[i], weight='length')
            for j in range(n):
                # Nếu có đường đi thì lấy độ dài (mét), nếu không thì phạt 15km
                dist_matrix[i][j] = lengths.get(nodes[j], 15000.0)
        except Exception:
            dist_matrix[i, :] = 15000.0

    print(f"💾 Bước 2.5: Đang lưu ma trận Tensor...")
    # Chuyển sang PyTorch Tensor và lưu
    dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32)
    torch.save(dist_tensor, OUTPUT_FILE)
    
    print(f"\n✅ THÀNH CÔNG!")
    print(f"📍 Ma trận khoảng cách đã lưu tại: {OUTPUT_FILE}")
    print(f"📊 Kích thước ma trận: {dist_tensor.shape}")

if __name__ == "__main__":
    compute_street_distances()