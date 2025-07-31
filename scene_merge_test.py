import numpy as np
from collections import Counter

def get_grid_coords(dims, resolution, vox_origin):
    """
    计算体素网格中每个体素的中心坐标
    :param dims: 体素网格的尺寸 [x, y, z]
    :param resolution: 体素分辨率
    :param vox_origin: 体素网格在世界坐标系中的原点
    :return: 体素中心坐标数组，形状为 (N, 3)
    """
    g_xx = np.arange(0, dims[0])
    g_yy = np.arange(0, dims[1])
    g_zz = np.arange(0, dims[2])

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing='ij')
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(float)
    coords_grid = (coords_grid * resolution) + resolution / 2
    coords_grid += vox_origin
    return coords_grid

def merge_voxels_to_world(voxels, voxel_size=0.08, vox_origins=None):
    """
    将多帧体素数据映射到世界坐标系并重新构造体素网格
    :param voxels: 多帧体素数据，形状为 [30, 1, 12, 60, 60, 36]
    :param voxel_size: 体素分辨率，单位：米
    :param vox_origins: 每帧体素网格在世界坐标系中的原点，形状为 [30, 3]
    :return: 重新构造后的体素网格
    """
    num_frames = voxels.shape[0]
    if vox_origins is None:
        vox_origins = np.zeros((num_frames, 3))

    all_coords = []
    all_classes = []

    for i in range(num_frames):
        frame_voxels = voxels[i].squeeze(0)
        pred_classes = np.argmax(frame_voxels, axis=0)
        dims = pred_classes.shape
        coords = get_grid_coords(dims, voxel_size, vox_origins[i])
        all_coords.append(coords)
        all_classes.append(pred_classes.flatten())

    all_coords = np.vstack(all_coords)
    all_classes = np.hstack(all_classes)
    combined_data = np.hstack([all_coords, all_classes[:, np.newaxis]])

    min_x, min_y, min_z = np.min(all_coords, axis=0)
    max_x, max_y, max_z = np.max(all_coords, axis=0)

    new_dims = [
        int(np.ceil((max_x - min_x) / voxel_size)),
        int(np.ceil((max_y - min_y) / voxel_size)),
        int(np.ceil((max_z - min_z) / voxel_size))
    ]

    new_voxels = np.zeros(new_dims, dtype=np.uint8)
    voxel_class_counts = {}

    for i in range(combined_data.shape[0]):
        x_idx = int((combined_data[i, 0] - min_x) / voxel_size)
        y_idx = int((combined_data[i, 1] - min_y) / voxel_size)
        z_idx = int((combined_data[i, 2] - min_z) / voxel_size)

        if (x_idx, y_idx, z_idx) not in voxel_class_counts:
            voxel_class_counts[(x_idx, y_idx, z_idx)] = []
        voxel_class_counts[(x_idx, y_idx, z_idx)].append(combined_data[i, 3])

    for (x_idx, y_idx, z_idx), class_list in voxel_class_counts.items():
        counter = Counter(class_list)
        most_common_class = counter.most_common(1)[0][0]
        new_voxels[x_idx, y_idx, z_idx] = most_common_class

    return new_voxels


# 示例调用
voxels = np.random.rand(30, 1, 12, 60, 60, 36)
vox_origins = np.random.rand(30, 3) * 10  # 随机生成每帧的原点
new_voxels = merge_voxels_to_world(voxels, voxel_size=0.08, vox_origins=vox_origins)
print(f"重新构造后的体素网格形状: {new_voxels.shape}")