import os, requests, pandas as pd
from PIL import Image
from io import BytesIO

def download_all(csv_file, save_folder, id_col, url_col):
    os.makedirs(save_folder, exist_ok=True)
    df = pd.read_csv(csv_file)
    
    print(f"🚀 Đang xử lý file: {os.path.basename(csv_file)}...")
    
    for _, row in df.iterrows():
        poi_id = row.get(id_col)
        url = row.get(url_col)
        
        # Nếu không có link ảnh thì bỏ qua
        if pd.isna(url):
            continue
            
        # Xử lý trường hợp Google Maps có nhiều link ảnh cách nhau bằng dấu phẩy
        url = str(url).split(',')[0].strip()
        
        # Làm sạch ID (thay dấu : thành _ để Windows không báo lỗi tên file)
        clean_id = str(poi_id).replace(':', '_')
        path = os.path.join(save_folder, f"{clean_id}.jpg")
        
        if os.path.exists(path):
            continue  # đã tải rồi, skip cho nhanh
            
        try:
            r = Image.open(BytesIO(requests.get(url, timeout=5).content)).convert("RGB")
            # Resize luôn về 224x224 chuẩn bị cho CLIP/ResNet để đỡ tốn dung lượng
            r = r.resize((224, 224))
            r.save(path)
        except Exception as e:
            pass  # bỏ qua nếu lỗi mạng hoặc link hỏng

# Truyền đúng tên cột cho Google Maps
download_all("D:/poi-urban-danang/dataset/processed/poi_data_ggmap.csv", 
             "D:/poi-urban-danang/dataset/poi_images_ggmap",
             id_col="place_id", 
             url_col="image_urls")

# Truyền đúng tên cột cho Foody
download_all("D:/poi-urban-danang/dataset/processed/poi_data_foody.csv", 
             "D:/poi-urban-danang/dataset/poi_images_foody",
             id_col="RestaurantID", 
             url_col="Image_URL")
             
print("✅ Hoàn tất tải ảnh!")