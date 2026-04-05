import os
import glob
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def crop_and_resize_buildings():
    input_dir = "dataset/building_images"
    output_dir = "dataset/building_images_224"
    
    if not os.path.exists(input_dir):
        print(f"❌ Không tìm thấy thư mục {input_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    image_paths.extend(glob.glob(os.path.join(input_dir, "*.jpg")))
    
    print(f"📦 Đã tìm thấy {len(image_paths)} ảnh tòa nhà. Bắt đầu xử lý...")
    
    # Transform pipeline: Resize cạnh nhỏ nhất về 224, sau đó cắt phần trung tâm 224x224
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    
    success = 0
    failed = 0
    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                out_img = transform(img)
                out_img.save(out_path, quality=95)
                success += 1
        except Exception as e:
            print(f"⚠️ Lỗi xử lý {filename}: {e}")
            failed += 1
            
    print(f"✅ Xong! Thành công: {success}, Lỗi: {failed}")
    print(f"📁 Toàn bộ ảnh chuẩn (224x224) được lưu tại: {output_dir}")

if __name__ == "__main__":
    crop_and_resize_buildings()