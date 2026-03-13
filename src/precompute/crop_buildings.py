import osmnx as ox
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 1. CẤU HÌNH ĐƯỜNG DẪN (Theo máy của bạn)
CSV_PATH = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\processed\poi_processed_data.csv"
OUTPUT_DIR = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\building_images"

def crop_building_polygons():
    # Tạo thư mục chứa ảnh nếu chưa có
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.read_csv(CSV_PATH)
    print(f"🏢 Đang trích xuất đa giác cho {len(df)} POIs...")

    # Tắt hiển thị cửa sổ biểu đồ để chạy nhanh hơn
    plt.ioff() 

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        poi_id = row['RestaurantID']
        lat, lon = row['Lat'], row['Lon']
        file_name = os.path.join(OUTPUT_DIR, f"building_{poi_id}.png")

        # Kiểm tra nếu ảnh đã có thì bỏ qua (resume)
        if os.path.exists(file_name):
            continue

        try:
            # Lấy dữ liệu đa giác tòa nhà trong bán kính 100m
            # tags={'building': True} giúp lọc riêng các khối nhà
            buildings = ox.features_from_point((lat, lon), tags={'building': True}, dist=100)

            # Tạo khung vẽ 2.24x2.24 inch với 100 DPI = 224x224 pixel
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            fig.patch.set_facecolor('black') # Nền đen
            ax.set_facecolor('black')

            if not buildings.empty:
                # Vẽ tòa nhà màu trắng
                buildings.plot(ax=ax, facecolor='white', edgecolor='none')
            
            # Xóa các trục tọa độ và lề thừa
            ax.set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
            
            # Lưu ảnh
            plt.savefig(file_name, facecolor='black', pad_inches=0)
            plt.close(fig)

            # Dùng PIL để chuẩn hóa tuyệt đối về 224x224 (ResNet standard)
            img = Image.open(file_name).convert('RGB').resize((224, 224))
            img.save(file_name)

        except Exception:
            # Nếu khu vực không có dữ liệu tòa nhà, tạo ảnh đen hoàn toàn
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            img.save(file_name)
            plt.close()

    print(f"\n✅ Xong! Ảnh đã nằm tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    crop_building_polygons() 