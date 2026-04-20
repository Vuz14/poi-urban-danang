import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def download_all(csv_file, save_folder, id_col, url_col):
    """
    Hàm tải toàn bộ ảnh từ file CSV và lưu vào thư mục chỉ định.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    if not os.path.exists(csv_file):
        print(f"⚠️ Không tìm thấy file: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    print(f"🚀 Đang xử lý file: {os.path.basename(csv_file)} ({len(df)} dòng)...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        poi_id = row.get(id_col)
        url = row.get(url_col)
        
        # Nếu không có link ảnh hoặc bị gán N/A thì bỏ qua
        if pd.isna(url) or str(url).upper() == 'N/A':
            continue
            
        # Xử lý trường hợp có nhiều link ảnh (thường gặp ở Google Maps)
        url = str(url).split(',')[0].strip()
        
        # Làm sạch ID để đặt tên file (tránh ký tự đặc biệt)
        clean_id = str(poi_id).replace(':', '_').replace('/', '_')
        path = os.path.join(save_folder, f"{clean_id}.jpg")
        
        if os.path.exists(path):
            continue  # Đã tải rồi, skip cho nhanh
            
        try:
            # Gửi request tải ảnh với timeout để tránh treo chương trình
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                # Resize về 224x224 chuẩn bị cho ResNet/CLIP để tối ưu dung lượng
                img = img.resize((224, 224))
                img.save(path, "JPEG", quality=90)
        except Exception as e:
            # Bỏ qua các lỗi tải ảnh cá biệt (link die, timeout, etc.)
            continue

if __name__ == "__main__":
    # 1. Cấu hình đường dẫn cho Google Maps (Source Domain)
    GGMAP_CSV = "dataset/processed/master_nodes_google_maps_clean.csv"
    GGMAP_IMG_DIR = "dataset/poi_images_ggmap"
    
    # 2. Cấu hình đường dẫn cho Foody (Target Domain)
    FOODY_CSV = "dataset/processed/master_nodes_foody_clean.csv"
    FOODY_IMG_DIR = "dataset/poi_images_foody"

    print("--- 📸 BẮT ĐẦU TẢI ẢNH POI ---")
    
    # Tải ảnh cho Google Maps
    # Lưu ý: Cột ID là 'Global_ID' dựa trên cấu hình master_nodes
    download_all(
        csv_file=GGMAP_CSV, 
        save_folder=GGMAP_IMG_DIR, 
        id_col="Global_ID", 
        url_col="URL"
    )

    # Tải ảnh cho Foody
    # Lưu ý: Foody dùng 'RestaurantID' và link ảnh nằm ở 'Image_URL'
    download_all(
        csv_file=FOODY_CSV, 
        save_folder=FOODY_IMG_DIR, 
        id_col="RestaurantID", 
        url_col="Image_URL"
    )

    print("\n✅ Hoàn tất quá trình tải ảnh!")