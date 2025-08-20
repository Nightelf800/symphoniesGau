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
from visual.syndata_visualize import draw
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


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(config: DictConfig):
    files = ([os.path.join(config.path, f)
              for f in os.listdir(config.path)] if os.path.isdir(config.path) else [config.path])
    test_evaluator = build_from_configs(evaluation, config.evaluator).cuda()

    print(f'----------files------------')
    print(f'files: {files}')

    for file in track(files):
        with open(file, 'rb') as f:
            outputs = pickle.load(f)
        id_name = file.split('/')[-1].split('.')[0]
        print(f'id_name: {id_name}')
        print(f'outputs.keys: {outputs.keys()}')

        cam_K = outputs['cam_K']
        cam_pose = outputs['cam_pose'] if 'cam_pose' in outputs else outputs[
            'T_velo_2_cam']  # compatible with MonoScene
        vox_origin = outputs['voxel_origin'] if 'voxel_origin' in outputs else np.array([0, -25.6, -2])
        fov_mask = outputs['fov_mask_1'] if 'fov_mask_1' in outputs else gt_visual_cal_fov_mask(outputs)
        pred = outputs['pred'] if 'pred' in outputs else None
        preds_ori = outputs['preds_ori'] if 'preds_ori' in outputs else None
        target = outputs['target'] if 'target' in outputs else outputs.pop('target_1_4').transpose(0, 2, 1)
        # if test_evaluator:
        #     test_evaluator.update({'ssc_logits': torch.tensor(preds_ori.astype(np.int32)).unsqueeze(0).cuda()},
        #                           {'target': torch.tensor(target.astype(np.int32)).unsqueeze(0).cuda()})
        # log_metrics(test_evaluator, 'val')

        # 计算缩放因子
        # 使用zoom函数进行下采样
        # zoom_factors = (50 / 60, 50 / 60, 25 / 36)
        # target = zoom(target, zoom_factors, order=0)  # order=3 是三次插值
        # 直接下采样
        # target = target[::2, ::2, ::2]

        # print(f'------------output-------------')
        # print(f'output.keys: {outputs.keys()}')
        # print('output[pred].shape: {}'.format(outputs['pred'].shape))
        # print('output[target].shape: {}'.format(outputs['target'].shape))

        data_type = config.data.datasets.type
        if data_type == 'SemanticKITTI':
            params = dict(
                img_size=(1220, 370),
                f=707.0912,
                voxel_size=0.2,
                d=7,
                colors=COLORS,
            )
        elif data_type == 'KITTI360':
            # Otherwise the trained model would output distorted results, due to unreasonably labeling
            # a large number of voxels as "ignored" in the annotations.
            pred[target == 255] = 0
            params = dict(
                img_size=(1408, 376),
                f=552.55426,
                voxel_size=0.2,
                d=7,
                colors=KITTI360_COLORS,
            )
        elif data_type == 'NYUv2':
            pred[target == 255] = 0
            params = dict(
                img_size=(640, 480),
                f=518.8579,
                voxel_size=0.04,
                d=0.75,
                colors=NYU_COLORS,
            )
        elif data_type == 'SYNData':
            pred[target == 255] = 0
            params = dict(
                img_size=(640, 480),
                f=cam_K[0, 0],
                voxel_size=0.08,
                d=0.5,
                colors=NYU_COLORS,
            )
        elif data_type == 'ScanNet':
            pred[target == 255] = 0
            params = dict(
                img_size=(640, 480),
                f=116.9621,
                voxel_size=0.08,
                d=0.3,
                colors=NYU_COLORS,
            )
        else:
            raise NotImplementedError

        print(f'config.data.datasets.type: {config.data.datasets.type}')
        print(f'vox_origin: {vox_origin}')
        file_name = file.split(os.sep)[-1].split(".")[0]
        for i, vol in enumerate((target, pred)):
            if vol is None:
                continue
            # print(f'{i} | vol: {vol.shape}')
            # draw(vol, cam_pose, vox_origin, fov_mask, **params,
            #      save_path=f'./outputs/visual', file_name=f'{id_name}_{i}.png')
            draw(data_type, vol, cam_pose, vox_origin, fov_mask, **params, need_update_view=False)


if __name__ == '__main__':
    main()

