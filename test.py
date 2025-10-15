import os
import os.path as osp
import pickle
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from rich.progress import track

from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks, build_from_configs, evaluation

class_names = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair',
                        'table', 'tvs', 'furn', 'objs', 'people']

def log_metrics(evaluator, prefix=None, file_names_list=None):
    metrics = evaluator.compute(file_names_list)
    iou_per_class = metrics.pop('iou_per_class')
    if prefix:
        metrics = {'/'.join((prefix, k)): v.item() for k, v in metrics.items()}
    print(f'metrics: {metrics}')
    print(f'iou_per_class: {iou_per_class}')
    evaluator.reset()


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

    test_evaluator = build_from_configs(evaluation, cfg.evaluator).cuda()
    model.cuda()
    model.eval()
    total_steps = len(data_loader)
    total_time = 0.0
    file_names_list = []

    with torch.no_grad():
        for batch_inputs, targets in track(data_loader):
            # print(batch_inputs.keys())
            # print('batch_inputs.name: {}'.format(batch_inputs['name']))
            targets = {key: targets[key].cuda() for key in targets}
            file_names = batch_inputs['filename'][0]
            scene_name = batch_inputs['scene'][0]
            file_names_list.append(f'{scene_name}_{file_names}')

            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs[key] = batch_inputs[key].cuda()

            start_time = time.time()  # 开始计时
            outputs = model(batch_inputs)
            step_time = time.time() - start_time  # 计算每步所用的时间
            if test_evaluator:
                test_evaluator.update(outputs, targets)

            fps = 1 / step_time  # 计算FPS
            total_time += step_time

        log_metrics(test_evaluator, 'val', file_names_list)

        average_fps = total_steps / total_time  # 计算平均FPS
        print(f"Average FPS over {total_steps} steps: {average_fps:.2f}")


if __name__ == '__main__':
    main()
