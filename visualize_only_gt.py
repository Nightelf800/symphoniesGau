import os
import pickle
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from rich.progress import track
from mayavi import mlab
from threading import Thread

from scipy.ndimage import zoom
from visual.syndata_visualize import draw_gt
from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks, build_from_configs, evaluation
from ssc_pl.utils.helper import vox2pix

from xvfbwrapper import Xvfb

COLORS = np.array([
    [100, 150, 245, 255],
    [100, 230, 245, 255],
    [30, 60, 150, 255],
    [80, 30, 180, 255],
    [100, 80, 250, 255],
    [255, 30, 30, 255],
    [255, 40, 200, 255],
    [150, 30, 90, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [175, 0, 75, 255],
    [255, 200, 0, 255],
    [255, 120, 50, 255],
    [0, 175, 0, 255],
    [135, 60, 0, 255],
    [150, 240, 80, 255],
    [255, 240, 150, 255],
    [255, 0, 0, 255],
]).astype(np.uint8)

KITTI360_COLORS = np.concatenate((
    COLORS[0:6],
    COLORS[8:15],
    COLORS[16:],
    np.array([[250, 150, 0, 255], [50, 255, 255, 255]]).astype(np.uint8),
), 0)

NYU_COLORS = np.array(
    [
        [22, 191, 206, 255],  # 0 empty    浅青色
        [214, 38, 40, 255],  # 1 ceiling   红色
        [43, 160, 43, 255],  # 2 floor     绿色
        [158, 216, 229, 255],  # 3 wall    淡青色
        [114, 158, 206, 255],  # 4 window   灰蓝色
        [204, 204, 91, 255],  # 5 chair     浅黄色
        [255, 186, 119, 255],  # 6 bed      橙色
        [147, 102, 188, 255],  # 7 sofa     紫色
        [30, 119, 181, 255],  # 8 table     深蓝色
        [188, 188, 33, 255],  # 9 tvs       亮黄色
        [255, 127, 12, 255],  # 10 furn     橙色
        [196, 175, 214, 255],  # 11 objs       淡紫色
        [153, 153, 153, 255],  # 12     灰色
    ]
)

COLORS_MAPS = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furn', 'objs', '12']

def gt_visual_cal_fov_mask(outputs):
    cam_pose = np.linalg.inv(outputs['cam_pose'])
    voxel_origin = outputs['voxel_origin']
    cam_K = np.array(((518.8579, 0, 320), (0, 518.8579, 240), (0, 0, 1)))
    voxel_size = 0.04  # meters
    scene_size = (4, 4, 2)  # meters
    img_shape = (640, 480)

    projected_pix, fov_mask, pix_z = vox2pix(cam_pose, cam_K, voxel_origin,
                                             voxel_size, img_shape, scene_size)
    return fov_mask



def log_metrics(evaluator, prefix=None):
    metrics = evaluator.compute()
    iou_per_class = metrics.pop('iou_per_class')
    print(f'metrics: {metrics}')
    if prefix:
        metrics = {'/'.join((prefix, k)): v.item() for k, v in metrics.items()}
    print(f'metrics: {metrics}')
    evaluator.reset()


def main():
    output_file_path = "/home/ubuntu/hdd3/ylc/datasets/honor_coll_data/cleaned_preprocess_voxels_sam/"

    print(f'----------files------------')
    pkl_files = []
    for root, dirs, files in os.walk(output_file_path):
        for file in files:
            # 判断文件后缀是否为.pkl
            if file.lower().endswith(".pkl"):
                # 拼接绝对路径并添加到列表
                pkl_file_path = os.path.join(root, file)
                pkl_files.append(pkl_file_path)
    pkl_files_sorted = sorted(pkl_files, key=os.path.basename)
    print(f'pkl files total: {len(pkl_files)}')
    print(f'----------visual------------')

    for file in pkl_files_sorted:
        with open(file, 'rb') as f:
            outputs = pickle.load(f)
        id_name = file.split('/')[-1].split('.')[0]
        print(f'id_name: {id_name}')
        print(f'outputs.keys: {outputs.keys()}')

        cam_pose = np.linalg.inv(outputs['cam_pose'])
        vox_origin = outputs['voxel_origin']
        cam_K = outputs['intrinsic']
        target = outputs['target_1_4']

        params = dict(
            img_size=(640, 480),
            f=cam_K[0, 0],
            voxel_size=0.08,
            d=0.5,
            colors=NYU_COLORS,
        )

        # print(f'config.data.datasets.type: {config.data.datasets.type}')
        # print(f'vox_origin: {vox_origin}')
        draw_gt(target, cam_pose, vox_origin, **params,
             save_path=f'./outputs/visual_honor_gt', file_name=f'{id_name}.png')
        # draw_gt(target, cam_pose, vox_origin, **params, need_update_view=False)


if __name__ == '__main__':
    main()

