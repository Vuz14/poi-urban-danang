import os
import osmnx as ox
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import networkx as nx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def compute_street_distances():
    MASTER_NODES_PATH = os.path.join(PROJECT_ROOT, "dataset/processed/master_nodes.csv")
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "dataset/processed/street_dist_matrix.pt")
    TEMP_MMAP = os.path.join(PROJECT_ROOT, "dataset/processed/temp_dist.mmap")
    
    if not os.path.exists(MASTER_NODES_PATH):
        print(f"❌ Không tìm thấy file: {MASTER_NODES_PATH}")
        return

    df_master = pd.read_csv(MASTER_NODES_PATH)
    n_total = len(df_master) # Khoảng 133k
    
    # Xác định các hàng là POI thực tế (loại bỏ urban_voids khỏi danh sách làm gốc)
    is_poi = df_master['Source'] != 'urban_voids'
    poi_indices = df_master.index[is_poi].tolist()
    n_pois = len(poi_indices) # Khoảng 6k

    print(f"📦 Cấu hình ma trận tối ưu: [{n_pois} POIs x {n_total} Total Nodes]")
    
    # 1. Tải đồ thị mạng lưới đường
    try:
        G = ox.graph_from_place('Da Nang, Vietnam', network_type='drive')
    except Exception:
        G = ox.graph_from_place('Hai Chau District, Da Nang, Vietnam', network_type='drive')
        
    # Ánh xạ tọa độ sang IDs trên đồ thị OSMnx
    nodes_on_graph = ox.distance.nearest_nodes(G, X=df_master['Lon'].tolist(), Y=df_master['Lat'].tolist())
    
    # 2. Sử dụng Memory Mapping để chống tràn RAM
    # Thay vì khởi tạo mảng 3GB trên RAM, ta khởi tạo trên ổ cứng
    dist_matrix_mmap = np.memmap(TEMP_MMAP, dtype='float32', mode='w+', shape=(n_pois, n_total))
    dist_matrix_mmap[:] = 15000.0 # Giá trị mặc định 15km

    print(f"🚗 Đang tính toán Dijkstra cho {n_pois} điểm gốc...")
    
    for idx_in_matrix, i in enumerate(tqdm(poi_indices)):
        try:
            # Tính toán đường đi ngắn nhất từ 1 POI đến TẤT CẢ các node khác trong đồ thị
            source_node = nodes_on_graph[i]
            lengths = nx.single_source_dijkstra_path_length(G, source_node, weight='length')
            
            # Cập nhật khoảng cách cho tất cả 133k điểm trong hàng tương ứng
            for j in range(n_total):
                target_node = nodes_on_graph[j]
                if target_node in lengths:
                    dist_matrix_mmap[idx_in_matrix, j] = lengths[target_node]
            
            # Lưu dữ liệu xuống đĩa sau mỗi 100 vòng lặp để giải phóng bộ nhớ đệm
            if idx_in_matrix % 100 == 0:
                dist_matrix_mmap.flush()
                
        except Exception:
            continue

    # 3. Chuyển sang định dạng PyTorch Tensor để dùng trong Model
    print("\n💾 Đang đóng gói dữ liệu vào Tensor...")
    # Chuyển đổi từ memmap sang tensor (lúc này máy sẽ cần khoảng 3GB RAM trống để load lên và save)
    final_tensor = torch.from_numpy(np.array(dist_matrix_mmap))
    torch.save(final_tensor, OUTPUT_FILE)
    
    # Dọn dẹp file tạm
    del dist_matrix_mmap
    if os.path.exists(TEMP_MMAP):
        os.remove(TEMP_MMAP)
        
    print(f"✅ Hoàn thành! Kích thước file: {n_pois}x{n_total}")
    print(f"📁 Lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    compute_street_distances()