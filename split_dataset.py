import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Lấy đường dẫn gốc của project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def clean_and_filter_dataset(csv_path, geom_dir, output_path):
    print(f"🧹 Đang kiểm tra tính toàn vẹn của ảnh tại: {geom_dir}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file {csv_path}. Hãy chạy pds_sampler.py trước!")
        return

    df = pd.read_csv(csv_path)
    initial_count = len(df)
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        global_id = row['Global_ID']
        # Tên file lấy thẳng từ Global_ID
        img_path = os.path.join(geom_dir, f"{global_id}.png")
        
        # 1. Kiểm tra file có tồn tại không
        if not os.path.exists(img_path):
            continue
            
        # 2. Kiểm tra ảnh đen (0,0,0) - Xảy ra khi rớt biển hoặc lỗi GPS
        try:
            img = Image.open(img_path).convert('RGB')
            img_arr = np.array(img)
            if np.sum(img_arr) == 0: # Ảnh đen tuyệt đối
                continue
            valid_indices.append(idx)
        except Exception as e:
            continue

    df_clean = df.iloc[valid_indices].copy()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Đã lọc xong! Giữ lại {len(df_clean)}/{initial_count} điểm hợp lệ.")
    print(f"📁 File dữ liệu sạch được lưu tại: {output_path}")

if __name__ == "__main__":
    # Chạy dọn dẹp cho tập Source (Google Maps)
    clean_and_filter_dataset(
        csv_path=os.path.join(PROJECT_ROOT, "dataset/processed/master_nodes_google_maps.csv"),
        geom_dir=os.path.join(PROJECT_ROOT, "dataset/building_images_google_maps"),
        output_path=os.path.join(PROJECT_ROOT, "dataset/processed/master_nodes_google_maps_clean.csv")
    )