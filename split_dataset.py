import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Khai báo đường dẫn
IMAGE_DIR = "D:/poi-urban-danang/dataset/building_images_ggmap"
CSV_PATH = "D:/poi-urban-danang/dataset/processed/poi_data_ggmap.csv"
# Đổi lại đường dẫn output cho giống log bạn vừa chạy
OUTPUT_CSV = "D:/poi-urban-danang/dataset/processed/poi_data_ggmap_v1_filtered.csv" 
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df = pd.read_csv(CSV_PATH)
valid_indices = []

print("Đang quét và lọc bỏ các POI có ảnh tòa nhà bị đen...")

# Cờ để in thử 1 file xem đường dẫn đúng chưa
debug_printed = False 

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # FIX LỖI Ở ĐÂY: Thay dấu ':' thành '_'
    raw_poi_id = str(row.get('place_id', str(idx)))
    safe_poi_id = raw_poi_id.replace(':', '_') 
    
    img_name = f"building_gg_{safe_poi_id}.png" 
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    if not debug_printed:
        print(f"\n[DEBUG] Đang tìm file thử: {img_path}")
        debug_printed = True

    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert('L') 
            if np.array(img).max() > 0:  
                valid_indices.append(idx)
        except Exception as e:
            pass 

df_filtered = df.iloc[valid_indices]
df_filtered.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Đã tạo file mới tại: {OUTPUT_CSV}")
print(f"Số lượng POI hợp lệ để chạy V1: {len(df_filtered)}")