import os
os.environ['NCCL_P2P_DISABLE'] = '1'
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from ssc_pl import LitModule, build_data_loaders, pre_build_callbacks
from cfg_module import ConfigManager
import torch
@hydra.main(config_path='configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    # global_cfg, _ = pre_build_callbacks(global_cfg)

    # cfg.models = 'Symphonies'
    # import pdb;
    # pdb.set_trace()
    ckpt_path = '/share/lkl/Symphonies/outputs/11_19_dim64_sym/e25_miou0.2860.ckpt'
    meta_info = {}
    meta_info['class_weights'] = torch.tensor([0.0500, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000])
    meta_info['class_names'] = ('empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furn', 'objs')

    # 假设 LitModule 的 load_from_checkpoint 方法接受 meta_info 作为参数
    

    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, callbacks = pre_build_callbacks(cfg)


    # ConfigManager.set_global_cfg(cfg)
    # sym_model = LitModule.load_from_checkpoint(ckpt_path, **cfg, meta_info=meta_info)
    # ConfigManager.set_global_model(sym_model)
    # import pdb
    # pdb.set_trace()

    dls, meta_info = build_data_loaders(cfg.data)
    symphony_model = LitModule.load_from_checkpoint(ckpt_path, **cfg, meta_info=meta_info)

    # model = LitModule(**cfg, **meta_info)

    # import pdb;
    # pdb.set_trace()
    trainer = L.Trainer(strategy='ddp', **cfg.trainer, **callbacks)
    trainer.fit(symphony_model, *dls[:2])


if __name__ == '__main__':
    main()


