"""
dataset.py - POIDataset chỉ dùng Google Maps (ggmap) + Spatial Hard Negative Sampling
=====================================================
THAY ĐỔI SO VỚI PHIÊN BẢN CŨ:
  - [YC1] Xóa toàn bộ logic "foody"
  - [YC2] Hỗ trợ Multi-image: mỗi POI có thể có tối đa MAX_IMAGES ảnh
  - [YC3] Bổ sung Dynamic Negative Sampling từ file urban_voids.csv
  - [MỚI] Bổ sung đọc ảnh đa giác tòa nhà (geom_image) cho ResNet
  - [HARD-NEG] Thay thế random void bằng Spatial Hard Negative:
               Pre-computed KNN (K=30) trên toàn bộ POI theo tọa độ Haversine.
               Bổ sung method get_nearest_poi_data(anchor_idx, k) để training loop
               lấy k POI lân cận gần nhất làm negative samples.
"""

import os
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.neighbors import BallTree

# [HARD-NEG] Import sklearn NearestNeighbors để tính KNN spatial
from sklearn.neighbors import NearestNeighbors


class POIDataset(Dataset):
    # Số lượng ảnh tối đa mỗi POI (pad ảnh đen nếu thiếu)
    MAX_IMAGES = 5

    def __init__(
        self,
        csv_file: str,
        image_transform=None,
        image_dir: str = None,
        void_csv_file: str = None,
        geom_image_dir: str = None,  # Thư mục chứa ảnh tòa nhà crop (building_gg_ID.png)
        void_geom_image_dir: str = None,
    ):
        self.data = pd.read_csv(csv_file)
        self.image_transform = image_transform
        self.image_dir = image_dir
        self.geom_image_dir = geom_image_dir
        self.void_geom_image_dir = void_geom_image_dir

        # ------------------------------------------------------------------
        # ĐỌC DỮ LIỆU VÙNG TRỐNG (Negative Sample)
        # ------------------------------------------------------------------
        if void_csv_file and os.path.exists(void_csv_file):
            self.void_data = pd.read_csv(void_csv_file)
            print(f"✅ Đã tải {len(self.void_data)} điểm Vùng trống từ {void_csv_file}")
            void_coords_rad = np.deg2rad(self.void_data[['Lat', 'Lon']].values)
            self.void_tree = BallTree(void_coords_rad, metric='haversine')
        else:
            self.void_data = None
            self.void_tree = None
            print("⚠️ Không tìm thấy file Vùng trống (void_csv_file).")

        # ------------------------------------------------------------------
        # CHUẨN HÓA DỮ LIỆU GGMAP
        # ------------------------------------------------------------------
        if 'image_urls' in self.data.columns:
            self.data['Image_URL'] = self.data['image_urls'].apply(
                lambda x: str(x).split(',')[0].strip() if pd.notna(x) and x else None
            )

        self.data['RestaurantID'] = self.data['place_id']
        self.data['District']     = self.data.get('district', self.data.get('category', 'Unknown'))
        self.data['Category']     = self.data.get('category', 'Unknown')
        self.data['Lat']          = self.data['lat'].astype(np.float32)
        self.data['Lon']          = self.data['lng'].astype(np.float32)

        # ------------------------------------------------------------------
        # [HARD-NEG] PRE-COMPUTE SPATIAL NEAREST NEIGHBORS (K=30)
        # ------------------------------------------------------------------
        coords = self.data[['Lat', 'Lon']].values
        nn_model = NearestNeighbors(n_neighbors=31, metric='haversine')  # 31 để loại self
        nn_model.fit(np.radians(coords))
        distances, indices = nn_model.kneighbors(np.radians(coords))
        self.nearest_indices = indices[:, 1:31]   # loại index 0 (chính nó), lấy 30 POI gần nhất
        
        print(f"✅ Precomputed nearest neighbors cho {len(self.data)} POI (K=30)")
        print(f"✅ POIDataset khởi tạo: {len(self.data)} POI | Source = Google Maps (ggmap)")
        print(f"   image_dir = {self.image_dir}")
        print(f"   geom_image_dir = {self.geom_image_dir}")
        print(f"   MAX_IMAGES per POI = {self.MAX_IMAGES}")

    def __len__(self):
        return len(self.data)

    def _load_multi_images(self, poi_id: str):
        """Tải ảnh cho POI thật từ thư mục local"""
        images = []
        if self.image_dir:
            clean_id = str(poi_id).replace(':', '_')
            for i in range(1, self.MAX_IMAGES + 1):
                path_numbered = os.path.join(self.image_dir, f"{clean_id}_{i}.jpg")
                path_single   = os.path.join(self.image_dir, f"{clean_id}.jpg")

                img = None
                if os.path.exists(path_numbered):
                    try: img = Image.open(path_numbered).convert("RGB")
                    except: pass
                elif i == 1 and os.path.exists(path_single):
                    try: img = Image.open(path_single).convert("RGB")
                    except: pass

                if img is None:
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
                images.append(img)
        else:
            images = [Image.new('RGB', (224, 224), (0, 0, 0))] * self.MAX_IMAGES

        return images

    def _get_dummy_images(self):
        """Tạo ảnh đen hoàn toàn cho Vùng trống"""
        return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(self.MAX_IMAGES)]

    def __getitem__(self, idx: int):
        # ==========================================
        # 1. LẤY DỮ LIỆU POI THẬT (POSITIVE)
        # ==========================================
        row    = self.data.iloc[idx]
        poi_id = row['RestaurantID']
        coords = torch.tensor([row['Lat'], row['Lon']], dtype=torch.float32)
        text   = str(row.get('LLM_Input_Text', ''))

        # Lấy Multi-image cho CLIP
        pil_images = self._load_multi_images(poi_id)
        if self.image_transform:
            tensor_imgs = torch.stack([self.image_transform(img) for img in pil_images])
        else:
            tensor_imgs = pil_images

        # Lấy ảnh đa giác tòa nhà cho ResNet (FIX LỖI TÊN FILE VÀ DẤU `:`)
        safe_poi_id = str(poi_id).replace(':', '_')
        if self.geom_image_dir:
            geom_path = os.path.join(self.geom_image_dir, f"building_gg_{safe_poi_id}.png")
            if os.path.exists(geom_path):
                try: 
                    geom_img = Image.open(geom_path)
                    # Xử lý nền trong suốt thành nền trắng
                    if geom_img.mode in ('RGBA', 'LA') or (geom_img.mode == 'P' and 'transparency' in geom_img.info):
                        alpha = geom_img.convert('RGBA').split()[-1]
                        bg = Image.new("RGB", geom_img.size, (255, 255, 255))
                        bg.paste(geom_img, mask=alpha)
                        geom_img = bg
                    else:
                        geom_img = geom_img.convert('RGB')
                except: 
                    geom_img = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                geom_img = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            geom_img = Image.new('RGB', (224, 224), (0, 0, 0))

        tensor_geom = self.image_transform(geom_img) if self.image_transform else geom_img

        poi_dict = {
            'poi_id'    : str(poi_id),
            'district'  : str(row.get('District', 'Unknown')),
            'coords'    : coords,
            'text'      : text,
            'image'     : tensor_imgs,
            'geom_image': tensor_geom,
            'category'  : str(row.get('Category', 'Unknown')),
        }

        # ==========================================
        # 2. BỐC NGẪU NHIÊN VÙNG TRỐNG (NEGATIVE)
        # Tự động gán Dummy Data như trong Checklist
        # ==========================================
        if self.void_data is not None:
            void_idx = random.randint(0, len(self.void_data) - 1)
            void_row = self.void_data.iloc[void_idx]
            void_id  = str(void_row['Global_ID'])
            void_coords = torch.tensor([void_row['Lat'], void_row['Lon']], dtype=torch.float32)

            # Dummy Multi-image (Ảnh đen thui)
            void_pil_images = self._get_dummy_images()
            if self.image_transform:
                void_tensor_imgs = torch.stack([self.image_transform(img) for img in void_pil_images])
            else:
                void_tensor_imgs = void_pil_images

            # Dummy Geom Image cho Void (Fix lỗi đọc nhầm file geom_path của POI)
            if self.void_geom_image_dir:
                void_geom_path = os.path.join(self.void_geom_image_dir, f"building_{void_id}.png")
                if os.path.exists(void_geom_path):
                    try: 
                        void_geom_img = Image.open(void_geom_path)
                        if void_geom_img.mode in ('RGBA', 'LA') or (void_geom_img.mode == 'P' and 'transparency' in void_geom_img.info):
                            alpha = void_geom_img.convert('RGBA').split()[-1]
                            bg = Image.new("RGB", void_geom_img.size, (255, 255, 255))
                            bg.paste(void_geom_img, mask=alpha)
                            void_geom_img = bg
                        else:
                            void_geom_img = void_geom_img.convert('RGB')
                    except: 
                        void_geom_img = Image.new('RGB', (224, 224), (0, 0, 0))
                else:
                    void_geom_img = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                void_geom_img = Image.new('RGB', (224, 224), (0, 0, 0))

            void_tensor_geom = self.image_transform(void_geom_img) if self.image_transform else void_geom_img

            void_dict = {
                'void_id'        : void_id,
                'void_coords'    : void_coords,
                'void_text'      : "", # Dummy Text rỗng
                'void_image'     : void_tensor_imgs,
                'void_geom_image': void_tensor_geom
            }
        else:
            void_dict = {}

        return {'poi': poi_dict, 'void': void_dict}

    def get_nearest_poi_data(self, anchor_idx: int, k: int = 30):
        """Trả về list chứa dict 'poi' của k POI lân cận gần nhất."""
        k = min(k, self.nearest_indices.shape[1])
        neighbor_indices = self.nearest_indices[anchor_idx][:k]
        return [self.__getitem__(int(i))['poi'] for i in neighbor_indices]
    
    def get_nearest_void_data(self, poi_idx, k=16):
        """
        Hàm tìm k điểm void gần một POI nhất bằng BallTree
        """
        if self.void_tree is None:
            return []

        # Lấy tọa độ POI hiện tại (chuyển sang radian)
        poi_lat = self.data.iloc[poi_idx]['lat']
        poi_lng = self.data.iloc[poi_idx]['lng']
        poi_coord_rad = np.deg2rad([[poi_lat, poi_lng]])

        # Tìm k lân cận trong cây BallTree
        # dists: khoảng cách, indices: chỉ số của void trong self.void_data
        _, indices = self.void_tree.query(poi_coord_rad, k=k)
        
        nearest_void_indices = indices[0]
        
        results = []
        for v_idx in nearest_void_indices:
            # Gọi hàm lấy dữ liệu của 1 điểm void (ảnh, tọa độ, v.v.)
            results.append(self.get_void_item(v_idx))
        return results

    def get_void_item(self, idx):
        """
        Hàm bổ trợ để lấy dữ liệu của một điểm Void cụ thể
        """
        row = self.void_data.iloc[idx]
        # Giả sử bạn có logic nạp ảnh cho void ở đây
        # Nếu chưa có ảnh cho void, hãy trả về dummy/tensor trắng hoặc ảnh building
        return {
            'void_id': row['Global_ID'],
            'coords': torch.tensor([row['Lat'], row['Lon']], dtype=torch.float),
            'text': "Urban void area, vacant land",
            'image': torch.zeros((3, 224, 224)), # Placeholder nếu chưa có ảnh
            'geom_image': torch.zeros((3, 224, 224)) # Placeholder
        }
# =========================================================================
# HÀM COLLATE CUSTOM: Gom batch sao cho POI và Void được tách bạch
# =========================================================================
def custom_collate_fn(batch):
    poi_batch = {
        'poi_id'    : [item['poi']['poi_id'] for item in batch],
        'district'  : [item['poi']['district'] for item in batch],
        'coords'    : torch.stack([item['poi']['coords'] for item in batch]),
        'text'      : [item['poi']['text'] for item in batch],
        'image'     : torch.stack([item['poi']['image'] for item in batch]) if torch.is_tensor(batch[0]['poi']['image']) else [item['poi']['image'] for item in batch],
        'geom_image': torch.stack([item['poi']['geom_image'] for item in batch]) if torch.is_tensor(batch[0]['poi']['geom_image']) else [item['poi']['geom_image'] for item in batch],
        'category'  : [item['poi']['category'] for item in batch]
    }

    if batch[0].get('void') and 'void_id' in batch[0]['void']:
        void_batch = {
            'void_id'        : [item['void']['void_id'] for item in batch],
            'coords'         : torch.stack([item['void']['void_coords'] for item in batch]),
            'text'           : [item['void']['void_text'] for item in batch],
            'image'          : torch.stack([item['void']['void_image'] for item in batch]) if torch.is_tensor(batch[0]['void']['void_image']) else [item['void']['void_image'] for item in batch],
            'geom_image'     : torch.stack([item['void']['void_geom_image'] for item in batch]) if torch.is_tensor(batch[0]['void']['void_geom_image']) else [item['void']['void_geom_image'] for item in batch]
        }
    else:
        void_batch = {}

    return {'poi': poi_batch, 'void': void_batch}