import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import numpy as np

class POIDataset(Dataset):
    def __init__(self, csv_file, image_transform=None):
        """
        Khởi tạo Dataset đọc dữ liệu từ file CSV đã tiền xử lý.
        """
        self.data = pd.read_csv(csv_file)
        self.image_transform = image_transform
        
        # Đảm bảo Lat/Lon là kiểu số thực float32
        self.data['Lat'] = self.data['Lat'].astype(np.float32)
        self.data['Lon'] = self.data['Lon'].astype(np.float32)

    def __len__(self):
        return len(self.data)

    def _download_image(self, url):
        """Tải ảnh từ URL, nếu lỗi trả về ảnh đen mặc định"""
        try:
            if pd.notna(url):
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    return Image.open(BytesIO(response.content)).convert("RGB")
        except:
            pass
        # Trả về ảnh đen (Zeros) kích thước 224x224 nếu link hỏng
        return Image.new('RGB', (224, 224), (0, 0, 0))

    def __getitem__(self, idx):
        # 1. Lấy Tọa độ Không gian (Lat, Lon)
        lat = self.data.iloc[idx]['Lat']
        lon = self.data.iloc[idx]['Lon']
        coords = torch.tensor([lat, lon], dtype=torch.float32)

        # 2. Lấy Văn bản tổng hợp (Tên + Giá + Review)
        text = str(self.data.iloc[idx]['LLM_Input_Text'])

        # 3. Lấy Hình ảnh
        img_url = self.data.iloc[idx]['Image_URL']
        image = self._download_image(img_url)
        if self.image_transform:
            image = self.image_transform(image)

        # Trả về 1 Dictionary chứa Đầy đủ thông tin của 1 POI
        return {
            'poi_id': self.data.iloc[idx]['RestaurantID'],
            'district': self.data.iloc[idx]['District'],
            'coords': coords,
            'text': text,
            'image': image
        }

# ==========================================
# ĐOẠN CODE TEST THỬ DATASET
# ==========================================
if __name__ == "__main__":
    from torchvision import transforms

    # Transform chuẩn cho ảnh đưa vào mô hình CLIP / ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Khởi tạo dataset (Chỉ đường dẫn đến file CSV của bạn)
    dataset = POIDataset(csv_file="dataset/processed/poi_processed_data.csv", image_transform=transform)
    
    # Tạo DataLoader (Bơm từng mẻ 4 POI một lúc vào mô hình)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Tổng số POI trong Dataset: {len(dataset)}")
    
    # Lấy thử 1 Batch đầu tiên ra xem
    for batch in dataloader:
        print("--- THÔNG TIN BATCH ĐẦU TIÊN ---")
        print("ID các quán:", batch['poi_id'].tolist())
        print("Tọa độ (Tensor):", batch['coords'].shape) # Sẽ là [4, 2]
        print("Hình ảnh (Tensor):", batch['image'].shape) # Sẽ là [4, 3, 224, 224]
        print("Văn bản 1 quán:", batch['text'][0][:100], "...")
        break