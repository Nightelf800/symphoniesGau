

# Symphonies使用说明

### [Symphonize 3D Semantic Scene Completion with Contextual Instance Queries](https://arxiv.org/abs/2306.15670)



## 环境配置

### 环境安装

- python 3.8
- cuda 11.3
- pytorch 1.12.0

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

**honor采集数据**
1. 解压honor_collection_data.zip

```
honor
|-- color
    |-- 000000.jpg
    |-- 000001.jpg
    ...
|-- depth_from_camera
    |-- 000000.png
    |-- 000001.png
    ...
|-- cleaned_preprocess_voxels
    |-- 000000.pkl
    |-- 000001.pkl
    ...
```
2. 软链接到代码`./data`路径下。 `ln -s 解压路径 ./data/honor_collection_data`
3. 预处理数据。执行`tools/preprocess_honor_scene_data.py`把数据分成train和test。
拆分规则: train: 0-650帧, test: 650帧之后

#### 2. 预训练权重

MaskDINO 预训练权重下载 [here](https://github.com/hustvl/Symphonies/releases/download/v1.0/maskdino_r50_50e_300q_panoptic_pq53.0.pth)，同样保存到`./checkpoints/`路径下。



## 训练

### 训练参数调整：

`configs/config_syndata_8cm.yaml`:

- trainer
  - device: 1-8  # 管理使用的显卡数

`configs/datasets/nyu_v2.yaml`：

- data_root：honor数据集采集路径
- depth_root（label）：honor数据集采集路径
- label_root：honor数据集采集路径
- depth_eval：true / false    # true表示使用depthAnything深度估计，false表示使用深度label
- use_crop: true / false   # true表示使用crop，false表示使用scale
- voxel_size: 0.04 / 0.08   # dataloader中使用分辨率0.04或者0.08

### 运行

```
[CUDA_VISIBLE_DEVICES=0,1,2...] python train.py --config-name config_syndata_8cm
```

## 测试

```
python test.py --config-name config_syndata_8cm [+ckpt_path=...]
```



## 可视化

1. 生成输出

```
python generate_outputs.py --config-name config_syndata_8cm [+ckpt_path=...]
```

2. 可视化

使用可视化mlab工具，推荐使用python3.8,推荐安装下列版本的第三方包：
```pip
pip install vtk==9.0.1
pip install mayavi==4.7.3
```

```
python visualize.py --config-name config_syndata_8cm [+path=...]
```



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



1. honor collection data

|                    Method                    |  train  |  test   | dim  |     depth      |  IoU  | mIoU  |
| :------------------------------------------: |  :----: |  :----: | :--: | :------------: | :---: | :---: |
| [Symphonies](symphonies/configs/config.yaml) |  0-650  | 650-817 |  128  |     label      | 24.16 | 7.05 |
