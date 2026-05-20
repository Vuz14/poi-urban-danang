import os
import random
import builtins
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors

def _safe_print(message):
    try:
        builtins.print(message)
    except UnicodeEncodeError:
        builtins.print(str(message).encode("ascii", errors="replace").decode("ascii"))

print = _safe_print

class POIDataset(Dataset):
    # Số lượng ảnh tối đa mỗi POI (pad ảnh đen nếu thiếu)
    MAX_IMAGES = 5

    def __init__(
        self,
        csv_file: str,
        image_transform=None,
        image_dir: str = None,
        void_csv_file: str = None,
        geom_image_dir: str = None,
        void_geom_image_dir: str = None,
    ):
        self.data = pd.read_csv(csv_file)
        # Chỉ lấy POI thật làm data chính (bỏ qua void nếu chúng nằm chung trong file master)
        if 'Source' in self.data.columns:
            self.data = self.data[~self.data['Source'].str.contains('void')].reset_index(drop=True)

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
            self.data['Image_URL'] = self.data['mage_urls'].apply(
                lambda x: str(x).split(',')[0].strip() if pd.notna(x) and x else None
            )

        self.data['RestaurantID'] = self.data.get('Place_id', self.data.index)
        self.data['District']     = self.data.get('District', self.data.get('Category', 'Unknown'))
        self.data['Category']     = self.data.get('Category', 'Unknown')
        self.data['Lat']          = self.data.get('Lat', self.data.get('lat')).astype(np.float32)
        self.data['Lon']          = self.data.get('Lon', self.data.get('lng')).astype(np.float32)

        # ------------------------------------------------------------------
        # [HARD-NEG] PRE-COMPUTE SPATIAL NEAREST NEIGHBORS (K=30)
        # ------------------------------------------------------------------
        coords = self.data[['Lat', 'Lon']].values
        nn_model = NearestNeighbors(n_neighbors=31, metric='haversine')
        nn_model.fit(np.radians(coords))
        distances, indices = nn_model.kneighbors(np.radians(coords))
        self.nearest_indices = indices[:, 1:31]
        
        print(f"✅ Precomputed nearest neighbors cho {len(self.data)} POI (K=30)")
        print(f"✅ POIDataset khởi tạo: {len(self.data)} POI")

    def __len__(self):
        return len(self.data)

    def _load_multi_images(self, poi_id: str):
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
        return [Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(self.MAX_IMAGES)]

    def _load_geom_image(self, global_id, base_dir):
        """Hàm load ảnh tòa nhà đã chuẩn hóa tên"""
        if not base_dir or pd.isna(global_id):
            return Image.new('RGB', (224, 224), (17, 17, 17)) # Nền xám tối an toàn
            
        geom_path = os.path.join(base_dir, f"{global_id}.png")
        if os.path.exists(geom_path):
            try: 
                geom_img = Image.open(geom_path)
                if geom_img.mode in ('RGBA', 'LA') or (geom_img.mode == 'P' and 'transparency' in geom_img.info):
                    alpha = geom_img.convert('RGBA').split()[-1]
                    bg = Image.new("RGB", geom_img.size, (255, 255, 255))
                    bg.paste(geom_img, mask=alpha)
                    geom_img = bg
                else:
                    geom_img = geom_img.convert('RGB')
                return geom_img
            except: 
                pass
        return Image.new('RGB', (224, 224), (17, 17, 17))

    def __getitem__(self, idx: int):
        # ==========================================
        # 1. LẤY DỮ LIỆU POI THẬT (POSITIVE)
        # ==========================================
        row    = self.data.iloc[idx]
        poi_id = row['RestaurantID']
        global_id = row.get('Global_ID', f"google_maps_{str(poi_id).replace(':','_')}")
        coords = torch.tensor([row['Lat'], row['Lon']], dtype=torch.float32)
        text   = str(row.get('LLM_Input_Text', ''))

        pil_images = self._load_multi_images(poi_id)
        if self.image_transform:
            tensor_imgs = torch.stack([self.image_transform(img) for img in pil_images])
        else:
            tensor_imgs = pil_images

        # Load Geom Image chuẩn (dùng Global_ID)
        geom_img = self._load_geom_image(global_id, self.geom_image_dir)
        tensor_geom = self.image_transform(geom_img) if self.image_transform else geom_img

        poi_dict = {
            'poi_id'    : str(poi_id),
            'global_id' : str(global_id),
            'district'  : str(row.get('District', 'Unknown')),
            'coords'    : coords,
            'text'      : text,
            'image'     : tensor_imgs,
            'geom_image': tensor_geom,
            'category'  : str(row.get('Category', 'Unknown')),
        }

        # ==========================================
        # 2. BỐC NGẪU NHIÊN VÙNG TRỐNG (NEGATIVE)
        # ==========================================
        if self.void_data is not None:
            void_idx = random.randint(0, len(self.void_data) - 1)
            void_row = self.void_data.iloc[void_idx]
            void_global_id  = str(void_row['Global_ID'])
            void_coords = torch.tensor([void_row['Lat'], void_row['Lon']], dtype=torch.float32)

            void_pil_images = self._get_dummy_images()
            if self.image_transform:
                void_tensor_imgs = torch.stack([self.image_transform(img) for img in void_pil_images])
            else:
                void_tensor_imgs = void_pil_images

            # Load Geom Image cho Void (dùng Global_ID)
            void_geom_img = self._load_geom_image(void_global_id, self.void_geom_image_dir)
            void_tensor_geom = self.image_transform(void_geom_img) if self.image_transform else void_geom_img

            void_dict = {
                'void_id'        : void_global_id,
                'void_coords'    : void_coords,
                'void_text'      : "Urban void area, vacant land", 
                'void_image'     : void_tensor_imgs,
                'void_geom_image': void_tensor_geom
            }
        else:
            void_dict = {}

        return {'poi': poi_dict, 'void': void_dict}

    def get_nearest_poi_data(self, anchor_idx: int, k: int = 30):
        k = min(k, self.nearest_indices.shape[1])
        neighbor_indices = self.nearest_indices[anchor_idx][:k]
        return [self.__getitem__(int(i))['poi'] for i in neighbor_indices]
    
    def get_nearest_void_data(self, poi_idx, k=16):
        if self.void_tree is None:
            return []
        poi_lat = self.data.iloc[poi_idx]['Lat']
        poi_lng = self.data.iloc[poi_idx]['Lon']
        poi_coord_rad = np.deg2rad([[poi_lat, poi_lng]])
        _, indices = self.void_tree.query(poi_coord_rad, k=k)
        
        nearest_void_indices = indices[0]
        results = []
        for v_idx in nearest_void_indices:
            results.append(self.get_void_item(v_idx))
        return results

    def get_void_item(self, idx):
        row = self.void_data.iloc[idx]
        void_global_id = row['Global_ID']
        
        # Load đúng ảnh Void để truyền vào model thay vì zeros
        void_geom_img = self._load_geom_image(void_global_id, self.void_geom_image_dir)
        void_tensor_geom = self.image_transform(void_geom_img) if self.image_transform else void_geom_img
        
        void_pil_images = self._get_dummy_images()
        void_tensor_imgs = torch.stack([self.image_transform(img) for img in void_pil_images]) if self.image_transform else void_pil_images

        return {
            'void_id': void_global_id,
            'coords': torch.tensor([row['Lat'], row['Lon']], dtype=torch.float),
            'text': "Urban void area, vacant land",
            'image': void_tensor_imgs,
            'geom_image': void_tensor_geom 
        }

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
