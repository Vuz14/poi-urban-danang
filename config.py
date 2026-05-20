# config.py
import os

# --- CẤU HÌNH PHIÊN BẢN ---
VERSIONS_TO_TRAIN = [4] 

# --- ĐƯỜNG DẪN DỮ LIỆU ---
TRAIN_CSV       = "dataset/processed/master_nodes_google_maps_clean.csv" 
TRAIN_IMAGE_DIR = "dataset/poi_images_ggmap"
TRAIN_GEOM_DIR  = "dataset/building_images_google_maps"
VOID_PATH       = "dataset/sampling/urban_voids_google_maps.csv"
VOID_GEOM_DIR   = "dataset/building_images_google_maps"

TEST_CSV        = "dataset/processed/master_nodes_foody_clean.csv"
TEST_IMAGE_DIR  = "dataset/poi_images_foody"
TEST_GEOM_DIR   = "dataset/building_images_foody"

# --- SIÊU THAM SỐ (HYPERPARAMETERS) ---
EMBEDDING_DIM = 256
BATCH_SIZE  = 24
NUM_EPOCHS  = 10
GROUP_SIZE  = 4
TEMPERATURE = 0.1
MARGIN      = 0.02
LR          = 1e-4
POSITIVE_NOISE_STD = 0.35
POSITIVE_FEATURE_DROPOUT = 0.2
# Windows multiprocessing can hide dataset exceptions as
# "DataLoader worker exited unexpectedly". Keep this at 0 for stable runs/debugging.
DATA_LOADER_WORKERS = 0
MIN_GROUP_PURITY = 0.5
