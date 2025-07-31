

# Symphonies使用说明

### [Symphonize 3D Semantic Scene Completion with Contextual Instance Queries](https://arxiv.org/abs/2306.15670)



## 环境配置

### 环境安装

- python 3.10
- cuda 11.8
- pytorch 2.1.0

```bash
pip install -U openmim

mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet

pip install -r requirements.txt
```

### 数据准备

#### 1. 下载数据

**NYUv2**: 

1. 从[NYUv2](https://www.rocq.inria.fr/rits_files/computer-vision/monoscene/nyu.zip)下载NYUv2的数据

2. 数据预处理

   ```
   python tools/preprocess_data_nyu.py # 注意修改数据位置路径
   ```

3. 最终处理好的数据路径如下

```
nyuv2
|-- depthbin
    |-- NYUtrain
    |-- NYUtest
|-- preprocess  # (4.8mx4.8mx2.88m)
    |-- base
    	|-- NYUtrain
        	|-- NYU0003_0000.pkl
        	|-- ...
    	|-- NYUtest
        	|-- NYU0001_0000.pkl
        	|-- ...   
```



#### 2. 深度估计图

方法一：

**DepthAnything:** 模型中已经内嵌DepthAnything推理模型，需下载两个权重文件，保存到`./checkpoints/`路径下。

- [depth_indoor](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth)
- [vitl14](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth)

方法二：

支持从DepthAnything模型预先估计得到所有NYUv2图像的深度图，然后在Symphonies中本地加载深度图。详细DepthAnything估计请查看`DepthAnything/metric_depth/README_NYUv2.md`



#### 3. 预训练权重

MaskDINO 预训练权重下载 [here](https://github.com/hustvl/Symphonies/releases/download/v1.0/maskdino_r50_50e_300q_panoptic_pq53.0.pth)，同样保存到`./checkpoints/`路径下。



## 训练

### 训练参数调整：

`configs/config.yaml`:

- trainer
  - device: 1-8  # 管理使用的显卡数

`configs/datasets/nyu_v2.yaml`：

- data_root：NYUv2数据集路径/depthbin
- depth_root（label）：NYUv2数据集路径/depthbin 
- label_root：NYUv2数据集路径/preprocess/base
- depth_eval：true / false    # true表示使用depthAnything深度估计，false表示使用深度label
- use_crop: true / false   # true表示使用crop，false表示使用scale
- voxel_size: 0.04 / 0.08   # dataloader中使用分辨率0.04或者0.08



`configs/models/symphonies.yaml`：

- model
  - embed_dims: 64   # 总体特征维度
- depth
  - depth_model: depthanything   # 使用的深度估计模型
  - depth_model_name: zoedepth   # depthanything中需要使用的模型
  - depth_pretrained_resource: local::./checkpoints/depth_anything_metric_depth_indoor.pt   # 加载微调后的权重文件路径（'local::'之后的才是相对路径）
  - depth_dataset: nyu   # 估计数据集名称
- optimizer
  - lr_mult: 0.05   # encoder的学习率缩小2倍（原先0.1）



`ssc_pl/models/decoders/symphonies_decoder.py`

- ```
  self.scene_pos = LearnableSqueezePositionalEncoding((20, 20, 50),	# 第130行
                                                      embed_dims,
                                                      squeeze_dims=(5, 5, 1))
  ```





#### 4m\*4m\*2m + 4cm + crop + label_depth + ddp1（单卡）

在上述训练参数调整的基础上修改以下参数：

`configs/config.yaml`:

- trainer
  - device: 1

`configs/datasets/nyu_v2.yaml`：

- depth_eval： false 
- use_crop: true
- voxel_size: 0.04 

`configs/models/symphonies.yaml`：

- depth
  - depth_model: None
- optimizer
  - lr_mult: 0.1  

`configs/schedules/adamw_lr_2e-4_30e.yaml`

- optimizer
  - lr: 2e-4



#### 4m\*4m\*2m + 4cm + crop + label_depth + ddp8（8卡）

在上述训练参数调整的基础上修改以下参数：

`configs/config.yaml`:

- trainer
  - device: 8

`configs/datasets/nyu_v2.yaml`：

- depth_eval： false 
- use_crop: true
- voxel_size: 0.04 

`configs/models/symphonies.yaml`：

- depth
  - depth_model: None
- optimizer
  - lr_mult: 0.05

`configs/schedules/adamw_lr_2e-4_30e.yaml`

- optimizer
  - lr: 8e-4



#### 4m\*4m\*2m + 4cm + crop + depthanything + ddp1（单卡）

在上述训练参数调整的基础上修改以下参数：

`configs/config.yaml`:

- trainer
  - device: 1

`configs/datasets/nyu_v2.yaml`：

- depth_eval： true
- use_crop: true
- voxel_size: 0.04 

`configs/models/symphonies.yaml`：

- depth
  - depth_model: depthanything
- optimizer
  - lr_mult: 0.1  

`configs/schedules/adamw_lr_2e-4_30e.yaml`

- optimizer
  - lr: 2e-4



#### 4m\*4m\*2m + 4cm + crop + depthanything + ddp8（8卡）

在上述训练参数调整的基础上修改以下参数：

`configs/config.yaml`:

- trainer
  - device: 8

`configs/datasets/nyu_v2.yaml`：

- depth_eval： true
- use_crop: true
- voxel_size: 0.04 

`configs/models/symphonies.yaml`：

- depth
  - depth_model: depthanything
- optimizer
  - lr_mult: 0.05

`configs/schedules/adamw_lr_2e-4_30e.yaml`

- optimizer
  - lr: 8e-4



#### 4m\*4m\*2m + 4cm + crop + depthanything + ddp8（8卡 本地加载深度图）

在上述训练参数调整的基础上修改以下参数：

`configs/config.yaml`:

- trainer
  - device: 8

`configs/datasets/nyu_v2.yaml`：

- depth_eval： true
- use_crop: true
- voxel_size: 0.04 

`configs/models/symphonies.yaml`：

- depth
  - depth_model: None
- optimizer
  - lr_mult: 0.05

`configs/schedules/adamw_lr_2e-4_30e.yaml`

- optimizer
  - lr: 8e-4

`ssc_pl/data/datasets/nyu_v2.py`

取消注释第146-148行，非本地加载请注释这三行代码。

```
depth_path = osp.join(self.depth_root, filename + '_pred.png')
depth = Image.open(depth_path)
data['depth'] = np.array(depth) / 8000.
```



#### 4m\*4m\*2m + 8cm + crop + label_depth + ddp1（单卡）

在上述训练参数调整的基础上修改以下参数：

`configs/config.yaml`:

- trainer
  - device: 1

`configs/datasets/nyu_v2.yaml`：

- data
  - dataset
    - depth_eval： false 
    - use_crop: true
    - voxel_size: 0.08
- model
  - scene_size：[50, 50, 25]
  - voxel_size：0.08

`configs/models/symphonies.yaml`：

- depth
  - depth_model: None
- optimizer
  - lr_mult: 0.1  

`configs/schedules/adamw_lr_2e-4_30e.yaml`

- optimizer
  - lr: 2e-4

`ssc_pl/models/decoders/symphonies_decoder.py`

- ```
  self.scene_pos = LearnableSqueezePositionalEncoding((10, 10, 25),	# 第130行
                                                       embed_dims,
                                                       squeeze_dims=(5, 5, 1))
  ```



### 运行

```
python train.py [--config-name config[.yaml]] [trainer.devices=4] \
    [+data_root=$DATA_ROOT] [+label_root=$LABEL_ROOT] [+depth_root=$DEPTH_ROOT]
```

例如：

4m\*4m\*2m + 4cm + crop + label_depth + ddp1（单卡）

如果已经根据上述训练参数调整好，直接以下输入命令即可。

```
python train.py
```

如果并未根据上述训练参数调整，则需要在之后添加上需要重写的参数，例如：

```
python train.py trainer.devices=1 +data_root=/data2/ylc/datasets/NYUv2/NYU_dataset/depthbin +label_root=/data2/ylc/datasets/NYUv2/NYU_dataset/preprocess/base +depth_root=/data2/ylc/datasets/NYUv2/NYU_dataset/depthbin +...
```



## 测试

```
python test.py [+ckpt_path=...]
```



## 可视化

1. 生成输出

        python generate_outputs.py [+ckpt_path=...]

2. 可视化

        xvfb-run python visualize.py [+path=...]  



## 结果

1. NYUv2 裁切到4m\*4m\*2m 4cm

|                    Method                    | Split | device | dim  |     depth      |    IoU    |   mIoU    |
| :------------------------------------------: | :---: | :----: | :--: | :------------: | :-------: | :-------: |
| [Symphonies](symphonies/configs/config.yaml) | test  |   1    |  64  |     label      |   40.74   |   24.10   |
| [Symphonies](symphonies/configs/config.yaml) | test  |   8    |  64  |     label      | **48.32** | **29.01** |
| [Symphonies](symphonies/configs/config.yaml) | test  |   1    |  64  | depthanything1 |   33.39   |   18.95   |
| [Symphonies](symphonies/configs/config.yaml) | test  |   8    |  64  | depthanything1 |   37.06   |   22.51   |
| [Symphonies](symphonies/configs/config.yaml) | test  |   8    |  64  | depthanything2 |  *32.45*  |  *19.99*  |

depthanything1: （相对深度 **需本地加载深度图**） 根据每个场景的label单独scale，需要用到每一个场景sample的深度真值

depthanything2：（相对深度 **直接在Symphonies估计**）取场景平均scale，不需要用到深度真值，只需要提前先验得到一个系数。



1. NYUv2 缩放到4m\*4m\*2m 4cm

|                    Method                    | Split | dim  |     depth      |  IoU  | mIoU  |
| :------------------------------------------: | :---: | :--: | :------------: | :---: | :---: |
| [Symphonies](symphonies/configs/config.yaml) | test  |  64  |     label      | 33.42 | 15.94 |
| [Symphonies](symphonies/configs/config.yaml) | test  |  64  | depthanything1 | 31.69 | 14.49 |

3. NYUv2 裁切到4m\*4m\*2m 8cm 

|                    Method                    | Split | dim  | depth |  IoU  | mIoU  |
| :------------------------------------------: | :---: | :--: | :---: | :---: | :---: |
| [Symphonies](symphonies/configs/config.yaml) | test  | 128  | label | 53.92 | 30.42 |

4. NYUv2 4.8m\*4.8m\*2.88m 8cm

|                    Method                    | Split | dim  | depth |  IoU  | mIoU  |
| :------------------------------------------: | :---: | :--: | :---: | :---: | :---: |
| [Symphonies](symphonies/configs/config.yaml) | test  | 128  | label | 50.76 | 30.88 |
