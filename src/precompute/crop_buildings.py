import osmnx as ox
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

POI_PATH = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\processed\poi_processed_data.csv"
VOID_PATH = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\sampling\urban_voids.csv"
OUTPUT_DIR = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\building_images"

def crop_logic(lat, lon, file_name):
    """Giữ nguyên logic vẽ đa giác cũ của bạn"""
    try:
        buildings = ox.features_from_point((lat, lon), tags={'building': True}, dist=100)
        fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        if not buildings.empty:
            buildings.plot(ax=ax, facecolor='white', edgecolor='none')
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
        plt.savefig(file_name, facecolor='black', pad_inches=0)
        plt.close(fig)
        Image.open(file_name).convert('RGB').resize((224, 224)).save(file_name)
    except Exception:
        Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(file_name)
        plt.close()

def run_sync_cropping():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    plt.ioff()

    # Bước 3.1: Cắt ảnh cho POI (Tên file: building_ID.png)
    df_poi = pd.read_csv(POI_PATH)
    print("🏢 Đang cắt ảnh cho các quán ăn...")
    for _, row in tqdm(df_poi.iterrows(), total=len(df_poi)):
        fname = os.path.join(OUTPUT_DIR, f"building_{row['RestaurantID']}.png")
        if not os.path.exists(fname):
            crop_logic(row['Lat'], row['Lon'], fname)

    # Bước 3.2: Cắt ảnh cho Vùng trống (Tên file: void_INDEX.png)
    df_void = pd.read_csv(VOID_PATH)
    print("🌌 Đang cắt ảnh cho các vùng trống đô thị...")
    for idx, row in tqdm(df_void.iterrows(), total=len(df_void)):
        fname = os.path.join(OUTPUT_DIR, f"void_{idx}.png")
        if not os.path.exists(fname):
            crop_logic(row['Lat'], row['Lon'], fname)

    print(f"✅ Hoàn tất! Ảnh POI và Voids đã nằm chung tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_sync_cropping()