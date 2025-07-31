import os
import os.path as osp
import pickle
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from collections import Counter
from rich.progress import track
from visualize import draw_scene

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks, build_from_configs, evaluation

class_names = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
               'table', 'tvs', 'furn', 'objs']
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


def log_metrics(evaluator, prefix=None, scene=False):
    metrics = evaluator.compute()
    iou_per_class = metrics.pop('iou_per_class')
    if prefix:
        metrics = {'/'.join((prefix, k)): v.item() for k, v in metrics.items()}
    if scene:
        print(f'*' * 10, 'scene', '*' * 10)
    else:
        print(f'*' * 20)
    print(f'metrics: {metrics}')
    print(f'*' * 20)
    evaluator.reset()


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
    # 若 vox_origin 是 PyTorch 张量，转换为 NumPy 数组
    if isinstance(vox_origin, torch.Tensor):
        vox_origin = vox_origin.cpu().numpy()
    coords_grid += vox_origin
    return coords_grid


def merge_voxels_to_world(voxels, target_shape=None, voxel_size=0.08, vox_origins=None, label=False):
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
        frame_voxels = voxels[i]
        if not label:
            frame_voxels = torch.softmax(frame_voxels, dim=0).detach().cpu().numpy()
            pred_classes = np.argmax(frame_voxels, axis=0)
        else:
            pred_classes = frame_voxels.detach().cpu().numpy()

        dims = pred_classes.shape
        coords = get_grid_coords(dims, voxel_size, vox_origins[i])
        all_coords.append(coords)
        all_classes.append(pred_classes.flatten())

    all_coords = np.vstack(all_coords)
    all_classes = np.hstack(all_classes)
    combined_data = np.hstack([all_coords, all_classes[:, np.newaxis]])

    min_x, min_y, min_z = np.min(all_coords, axis=0)
    max_x, max_y, max_z = np.max(all_coords, axis=0)

    if target_shape:
        new_voxels = np.zeros(target_shape, dtype=np.uint8)
    else:
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

        # 检查索引是否在目标形状范围内，若不在则丢弃
        if 0 <= x_idx < new_voxels.shape[0] and 0 <= y_idx < new_voxels.shape[1] and 0 <= z_idx < new_voxels.shape[2]:
            if (x_idx, y_idx, z_idx) not in voxel_class_counts:
                voxel_class_counts[(x_idx, y_idx, z_idx)] = []
            voxel_class_counts[(x_idx, y_idx, z_idx)].append(combined_data[i, 3])

    for (x_idx, y_idx, z_idx), class_list in voxel_class_counts.items():
        counter = Counter(class_list)
        most_common_class = counter.most_common(1)[0][0]
        new_voxels[x_idx, y_idx, z_idx] = most_common_class

    return new_voxels


@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    cfg, _ = pre_build_callbacks(cfg)

    dls, meta_info = build_data_loaders(cfg.data)
    data_loader = dls[1]

    if cfg.get('ckpt_path'):
        model = LitModule.load_from_checkpoint(cfg.ckpt_path, **cfg, meta_info=meta_info)
    else:
        import warnings
        warnings.warn('\033[31;1m{}\033[0m'.format('No ckpt_path is provided'))
        model = LitModule(**cfg, meta_info=meta_info)

    test_evaluator = build_from_configs(evaluation, cfg.evaluator1).cuda()
    test_scene_evaluator = build_from_configs(evaluation, cfg.evaluator2).cuda()
    model.cuda()
    model.eval()
    total_steps = len(data_loader)
    total_time = 0.0

    scene_outputs = []  # 用于存储每个场景的输出
    scene_targets = []
    current_scene = None

    params = dict(
        img_size=(640, 480),
        f=480,
        voxel_size=0.08,
        d=0.5,
        colors=NYU_COLORS,
    )

    with torch.no_grad():
        for batch_inputs, targets in track(data_loader):
            # print(batch_inputs.keys())
            # print('batch_inputs.name: {}'.format(batch_inputs['name']))
            scenes = batch_inputs['scene']  # 假设 'name' 标识场景

            targets = {key: targets[key].cuda() for key in targets}
            # tar = targets['target']
            # mask = torch.where((tar != 0) & (tar != 255))
            # tar = tar[mask]

            # for i in range(len(class_names)):
            #     print('class: {}, occ: {}'.format(class_names[i], torch.sum(tar==i)/tar.size(0)))

            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()

            start_time = time.time()  # 开始计时
            outputs = model(batch_inputs)
            # print(f'outputs: {outputs.keys()}')
            # for key in outputs:
            #     if isinstance(outputs[key], torch.Tensor):
            #         print(f'outputs[{key}].shape: {outputs[key].shape}')
            #     if isinstance(outputs[key], list):
            #         print(f'outputs[{key}].len: {len(outputs[key])}')
            #         print(f'outputs[{key}][0].shape: {outputs[key][0].shape}')

            step_time = time.time() - start_time  # 计算每步所用的时间
            if test_evaluator:
                test_evaluator.update(outputs, targets)

            batch_size = len(scenes)
            for i in range(batch_size):
                scene = scenes[i]
                voxel_origin = batch_inputs['voxel_origin'][i]
                cam_pose = batch_inputs['cam_pose'][i]
                fov_mask = batch_inputs['fov_mask_1'][i]

                sample_outputs = {
                    'ssc_logits': outputs['ssc_logits'][i].unsqueeze(0),
                    'voxel_origin': voxel_origin,
                    'cam_pose': cam_pose,
                    'fov_mask': fov_mask
                }
                # 仅提取 target 作为 sample_targets
                sample_targets = {
                    'target': targets['target'][i].unsqueeze(0)
                }
                if current_scene is not None and scene != current_scene:
                    # 场景变化，对之前场景的输出进行拼接
                    scene_output = torch.stack([out['ssc_logits'] for out in scene_outputs], dim=0)
                    scene_voxel_origin = torch.stack([out['voxel_origin'] for out in scene_outputs], dim=0)
                    scene_cam_pose = torch.stack([out['cam_pose'] for out in scene_outputs], dim=0)
                    scene_fov_mask = torch.stack([out['fov_mask'] for out in scene_outputs], dim=0)
                    scene_target = torch.stack([out['target'] for out in scene_targets], dim=0)
                    scene_output = torch.squeeze(scene_output, dim=1)
                    scene_target = torch.squeeze(scene_target, dim=1)

                    # print(f"scene_output.shape: {scene_output.shape}")
                    # print(f"scene_target.shape: {scene_target.shape}")

                    scene_target_voxels = merge_voxels_to_world(scene_target, voxel_size=0.08,
                                                                vox_origins=scene_voxel_origin, label=True)
                    # print(f'scene_target_voxels.shape: {scene_target_voxels.shape}')
                    scene_output_voxels = merge_voxels_to_world(scene_output, target_shape=scene_target_voxels.shape,
                                                                voxel_size=0.08, vox_origins=scene_voxel_origin)
                    # print(f'scene_output_voxels.shape: {scene_output_voxels.shape}')

                    # 转换为 torch.Tensor 并放到指定设备
                    device = sample_outputs['ssc_logits'].device
                    scene_output_voxels = torch.tensor(scene_output_voxels, device=device).unsqueeze(0)
                    scene_target_voxels = torch.tensor(scene_target_voxels, device=device).unsqueeze(0)
                    if test_scene_evaluator:
                        test_scene_evaluator.update({'ssc_logits': scene_output_voxels},
                                                    {'target': scene_target_voxels})

                    pred_np = scene_output_voxels.squeeze(0).detach().cpu().numpy()
                    target_np = scene_target_voxels.squeeze(0).detach().cpu().numpy()
                    cam_pose_np = scene_cam_pose.detach().cpu().numpy()
                    vox_origin_np = scene_voxel_origin.detach().cpu().numpy()
                    fov_mask_np = scene_fov_mask.detach().cpu().numpy()

                    print(f'############## visual ##############')
                    for i, vol in enumerate((pred_np, target_np)):
                        if vol is None:
                            continue
                        print(f'{i} | vol: {vol.shape}')
                        # draw(vol, cam_pose, vox_origin, fov_mask, **params,
                        #      save_path=f'./outputs/visual', file_name=f'{id_name}_{i}.png')  #
                        for j in range(cam_pose_np.shape[0]):
                            draw_scene(vol, cam_pose_np[j], vox_origin_np[j], fov_mask_np[j], **params)  #

                    scene_outputs = []
                    scene_targets = []

                scene_outputs.append(sample_outputs)
                scene_targets.append(sample_targets)
                current_scene = scene

            fps = 1 / step_time  # 计算FPS
            total_time += step_time

            # preds = torch.softmax(outputs['ssc_logits'], dim=1).detach().cpu().numpy()
            # preds = np.argmax(preds, axis=1).astype(np.uint16)

            # print(f"FPS: {fps:.2f}")

        # 将每个场景的输出在新维度上拼接
        if len(scene_outputs) > 0 and len(scene_targets) > 0:
            scene_output = torch.stack([out['ssc_logits'] for out in scene_outputs], dim=0)
            scene_voxel_origin = torch.stack([out['voxel_origin'] for out in scene_outputs], dim=0)
            scene_target = torch.stack([out['target'] for out in scene_targets], dim=0)
            scene_output = torch.squeeze(scene_output, dim=1)
            scene_target = torch.squeeze(scene_target, dim=1)

            print(f"scene_output.shape: {scene_output.shape}")
            print(f"scene_target.shape: {scene_target.shape}")

            scene_target_voxels = merge_voxels_to_world(scene_target, voxel_size=0.08, vox_origins=scene_voxel_origin,
                                                        label=True)
            print(f'scene_target_voxels.shape: {scene_target_voxels.shape}')
            scene_output_voxels = merge_voxels_to_world(scene_output, target_shape=scene_target_voxels.shape,
                                                        voxel_size=0.08, vox_origins=scene_voxel_origin)
            print(f'scene_output_voxels.shape: {scene_output_voxels.shape}')

            # 转换为 torch.Tensor 并放到指定设备
            device = sample_outputs['ssc_logits'].device
            scene_output_voxels = torch.tensor(scene_output_voxels, device=device).unsqueeze(0)
            scene_target_voxels = torch.tensor(scene_target_voxels, device=device).unsqueeze(0)
            if test_scene_evaluator:
                test_scene_evaluator.update({'ssc_logits': scene_output_voxels}, {'target': scene_target_voxels})

        log_metrics(test_evaluator, 'val')
        log_metrics(test_scene_evaluator, 'val', scene=True)

        average_fps = total_steps / total_time  # 计算平均FPS
        print(f"Average FPS over {total_steps} steps: {average_fps:.2f}")


if __name__ == '__main__':
    main()
