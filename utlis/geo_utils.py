import numpy as np
import torch
import math

def haversine_matrix_torch(coords):
    """
    Tính ma trận khoảng cách Haversine (tính bằng mét) cho một tensor tọa độ.
    coords: Tensor kích thước (N, 2) với [vĩ độ (lat), kinh độ (lon)]
    """
    # Chuyển đổi sang radian
    coords_rad = torch.deg2rad(coords)
    lat = coords_rad[:, 0]
    lon = coords_rad[:, 1]
    
    # Tính chênh lệch
    dlat = lat.unsqueeze(1) - lat.unsqueeze(0)
    dlon = lon.unsqueeze(1) - lon.unsqueeze(0)
    
    # Công thức Haversine
    a = torch.sin(dlat / 2)**2 + torch.cos(lat.unsqueeze(1)) * torch.cos(lat.unsqueeze(0)) * torch.sin(dlon / 2)**2
    c = 2 * torch.asin(torch.sqrt(a + 1e-10))
    
    r = 6371000 # Bán kính Trái đất tính bằng mét
    return r * c

def simple_poisson_disk_sampling(width, height, radius=100, k=30):
    """
    Thuật toán Bridson cho Poisson Disk Sampling 2D cơ bản.
    Tạo các random points điền vào vùng trống.
    """
    cell_size = radius / math.sqrt(2)
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    
    points = []
    spawn_points = []
    
    # Khởi tạo điểm đầu tiên
    spawn_points.append((np.random.uniform(0, width), np.random.uniform(0, height)))
    
    while spawn_points:
        spawn_idx = np.random.randint(0, len(spawn_points))
        spawn_center = spawn_points[spawn_idx]
        accepted = False
        
        for _ in range(k):
            angle = 2 * math.pi * np.random.uniform(0, 1)
            r = np.random.uniform(radius, 2 * radius)
            px, py = spawn_center[0] + r * math.cos(angle), spawn_center[1] + r * math.sin(angle)
            
            if 0 <= px < width and 0 <= py < height:
                g_x, g_y = int(px / cell_size), int(py / cell_size)
                valid = True
                
                # Kiểm tra va chạm với các điểm lân cận
                for i in range(max(0, g_x - 1), min(grid_width, g_x + 2)):
                    for j in range(max(0, g_y - 1), min(grid_height, g_y + 2)):
                        if grid[i][j] is not None:
                            dist = math.hypot(grid[i][j][0] - px, grid[i][j][1] - py)
                            if dist < radius:
                                valid = False
                                break
                    if not valid: break
                
                if valid:
                    points.append((px, py))
                    spawn_points.append((px, py))
                    grid[g_x][g_y] = (px, py)
                    accepted = True
                    break
                    
        if not accepted:
            spawn_points.pop(spawn_idx)
            
    return np.array(points)