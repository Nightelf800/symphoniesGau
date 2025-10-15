# import torch
# from torchmetrics import Metric


# class SSCMetrics(Metric):

#     def __init__(self, num_classes, ignore_index=255):
#         super().__init__()
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index

#         for metric in ('tp_sc', 'fp_sc', 'fn_sc'):
#             self.add_state(metric, torch.tensor(0), dist_reduce_fx='sum')
#         for metric in ('tps_ssc', 'fps_ssc', 'fns_ssc'):
#             self.add_state(metric, torch.zeros(num_classes), dist_reduce_fx='sum')

#     def update(self, preds, target):
#         # print('preds.shape: {}'.format(preds['ssc_logits'].shape))
#         # density = preds['density']
#         dim = preds['ssc_logits'].dim()
#         if dim == 4:
#             preds = preds['ssc_logits']
#         elif dim == 5:
#             preds = torch.argmax(preds['ssc_logits'], dim=1)

#         # preds = preds['result']
#         # probs = preds['ssc_logits'].softmax(-1)
#         # import pdb;
#         # pdb.set_trace()

#         # preds = probs.argmax(-1)
#         # preds += 1
#         # preds = torch.where(density.squeeze(1) > 5e-2, preds, 0)
#         # print(f'preds.shape: {preds.shape}')
#         target = target['target']
#         mask = (target != self.ignore_index)

#         tp, fp, fn = self._calculate_sc_scores(preds, target, mask)
#         # print(f'tp: {tp}, fp: {fp}, fn: {fn}')
#         self.tp_sc += tp
#         self.fp_sc += fp
#         self.fn_sc += fn

#         tp, fp, fn = self._calculate_ssc_scores(preds, target, mask)
#         # print(f'tp: {tp}, fp: {fp}, fn: {fn}')
#         self.tps_ssc += tp
#         self.fps_ssc += fp
#         self.fns_ssc += fn

#     def compute(self):
#         if self.tp_sc != 0:
#             precision = self.tp_sc / (self.tp_sc + self.fp_sc)
#             recall = self.tp_sc / (self.tp_sc + self.fn_sc)
#             iou = self.tp_sc / (self.tp_sc + self.fp_sc + self.fn_sc)
#         else:
#             precision, recall, iou = 0, 0, 0
#         ious = self.tps_ssc / (self.tps_ssc + self.fps_ssc + self.fns_ssc + 1e-6)
#         return {
#             'Precision': precision,
#             'Recall': recall,
#             'IoU': iou,
#             'iou_per_class': ious,
#             'mIoU': ious[1:].mean()
#         }

#     def _calculate_sc_scores(self, preds, targets, nonempty=None):
#         preds = preds.clone()
#         targets = targets.clone()
#         bs = preds.shape[0]

#         mask = targets == self.ignore_index
#         preds[mask] = 0
#         targets[mask] = 0

#         preds = preds.flatten(1)
#         targets = targets.flatten(1)
#         preds = torch.where(preds > 0, 1, 0)
#         targets = torch.where(targets > 0, 1, 0)

#         tp, fp, fn = 0, 0, 0
#         for i in range(bs):
#             pred = preds[i]
#             target = targets[i]
#             if nonempty is not None:
#                 nonempty_ = nonempty[i].flatten()
#                 pred = pred[nonempty_]
#                 target = target[nonempty_]
#             pred = pred.bool()
#             target = target.bool()

#             tp += torch.logical_and(pred, target).sum()
#             fp += torch.logical_and(pred, ~target).sum()
#             fn += torch.logical_and(~pred, target).sum()
#         return tp, fp, fn

#     def _calculate_ssc_scores(self, preds, targets, nonempty=None):
#         preds = preds.clone()
#         targets = targets.clone()
#         bs = preds.shape[0]
#         C = self.num_classes

#         mask = targets == self.ignore_index
#         preds[mask] = 0
#         targets[mask] = 0

#         preds = preds.flatten(1)
#         targets = targets.flatten(1)

#         tp = torch.zeros(C, dtype=torch.int).to(preds.device)
#         fp = torch.zeros(C, dtype=torch.int).to(preds.device)
#         fn = torch.zeros(C, dtype=torch.int).to(preds.device)
#         for i in range(bs):
#             pred = preds[i]
#             target = targets[i]
#             if nonempty is not None:
#                 mask = nonempty[i].flatten() & (target != self.ignore_index)
#                 pred = pred[mask]
#                 target = target[mask]
#             for c in range(C):
#                 tp[c] += torch.logical_and(pred == c, target == c).sum()
#                 fp[c] += torch.logical_and(pred == c, ~(target == c)).sum()
#                 fn[c] += torch.logical_and(~(pred == c), target == c).sum()
#         return tp, fp, fn

import os, torch
from torchmetrics import Metric
from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SSCMetrics(Metric):

    def __init__(self, num_classes, ignore_index=255, voxel_size=(0.5, 0.5, 0.5)):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.voxel_size = voxel_size  # 体素物理尺寸，用于3D框计算
        self.last_class = num_classes - 1  # 最后一个类别的索引（person）

        # 原有状态变量
        for metric in ('tp_sc', 'fp_sc', 'fn_sc'):
            self.add_state(metric, torch.tensor(0), dist_reduce_fx='sum')
        for metric in ('tps_ssc', 'fps_ssc', 'fns_ssc'):
            self.add_state(metric, torch.zeros(num_classes), dist_reduce_fx='sum')

        # 存储最后一个类别的体素掩码（用于简化3D-IoU计算）
        # 注：为避免内存占用，仅存储当前batch的掩码，在compute中临时计算
        self.add_state('last_class_pred_masks', default=[], dist_reduce_fx='cat')  # 预测掩码列表
        self.add_state('last_class_gt_masks', default=[], dist_reduce_fx='cat')  # 真实掩码列表

    def update(self, preds, target):
        # 解析预测和目标数据
        dim = preds['ssc_logits'].dim()
        # print(f"preds_argmax shape: {preds['ssc_logits'].shape}")
        if dim == 4:
            ssc_logits = preds['ssc_logits']
            preds_argmax = torch.argmax(ssc_logits, dim=1)  # 计算预测类别
        elif dim == 5:
            ssc_logits = preds['ssc_logits']
            preds_argmax = torch.argmax(ssc_logits, dim=1)

        target = target['target']
        mask = (target != self.ignore_index)  # 有效区域掩码

        # 原有指标更新
        tp, fp, fn = self._calculate_sc_scores(preds_argmax, target, mask)
        self.tp_sc += tp
        self.fp_sc += fp
        self.fn_sc += fn

        tp, fp, fn = self._calculate_ssc_scores(preds_argmax, target, mask)
        self.tps_ssc += tp
        self.fps_ssc += fp
        self.fns_ssc += fn

        # 提取最后一个类别的体素掩码（用于简化3D-IoU）
        self._update_last_class_masks(preds_argmax, target, mask)

    def _update_last_class_masks(self, preds, target, mask):
        """提取最后一个类别的预测和真实体素掩码，存储为列表"""
        # 过滤忽略区域
        preds_valid = preds * mask
        target_valid = target * mask

        # 生成最后一个类别的二值掩码（1=该类别，0=其他）
        pred_mask = (preds_valid == self.last_class).float()  # 形状：(B, H, W, D)
        # print(f'pred_mask.shape: {pred_mask.shape}')
        gt_mask = (target_valid == self.last_class).float()  # 形状：(B, H, W, D)
        # print(f'gt_mask.shape: {gt_mask.shape}')

        # 存储当前batch的掩码（转为numpy便于后续处理，避免GPU内存占用）
        self.last_class_pred_masks.append(pred_mask.detach().clone())
        self.last_class_gt_masks.append(gt_mask.detach().clone())

    def compute(self, file_names_list=None):
        # 原有指标计算
        if self.tp_sc != 0:
            precision = self.tp_sc / (self.tp_sc + self.fp_sc)
            recall = self.tp_sc / (self.tp_sc + self.fn_sc)
            iou = self.tp_sc / (self.tp_sc + self.fp_sc + self.fn_sc)
        else:
            precision, recall, iou = 0, 0, 0
        ious = self.tps_ssc / (self.tps_ssc + self.fps_ssc + self.fns_ssc + 1e-6)

        # 基于物体边界框的准确率计算
        if file_names_list is not None:
            object_based_accuracy = self._compute_object_based_accuracy(file_names_list)
            return {
                'Precision': precision,
                'Recall': recall,
                'IoU': iou,
                'iou_per_class': ious,
                'mIoU': ious[1:].mean(),
                'last_class_object_based_accuracy': object_based_accuracy   # 基于物体边界框的准确率
            }
        else:
            return {
                'Precision': precision,
                'Recall': recall,
                'IoU': iou,
                'iou_per_class': ious,
                'mIoU': ious[1:].mean()
            }

    def _compute_object_based_accuracy(self, file_names_list=None):
        """
        计算基于物体边界框的准确率：
        1. 统计每个样本中GT物体的数量
        2. 为每个GT物体生成最大范围的3D边界框
        3. 检查每个预测的last_class体素是否在任何一个GT物体的边界框内
        4. 计算准确率 = 在边界框内的预测体素数 / 总预测体素数
        """
        total_correct_pred_voxels = 0  # 在GT物体边界框内的预测体素数
        total_pred_voxels = 0  # 总预测体素数
        count = 0

        if file_names_list is not None:
            print(f'file_names_list.len: {len(file_names_list)}')

        # 遍历每个样本的预测掩码和GT掩码
        for pred_mask, gt_mask in zip(self.last_class_pred_masks, self.last_class_gt_masks):
            pred_mask_np = pred_mask.cpu().numpy()[0]
            # print(f'pred_mask_np.shape: {pred_mask_np.shape}')
            gt_mask_np = gt_mask.cpu().numpy()[0]
            # print(f'gt_mask_np.shape: {gt_mask_np.shape}')
            # 获取GT中物体的边界框
            if file_names_list is not None:
                file_name = file_names_list[count]
            count += 1
            gt_boxes = self._get_gt_object_boxes(gt_mask_np)
            if not gt_boxes:  # 如果没有GT物体，跳过该样本
                continue

            # 获取预测的last_class体素坐标
            pred_voxel_coords = np.where(pred_mask_np > 0)  # (x, y, z) 坐标
            num_pred_voxels = len(pred_voxel_coords[0])
            if num_pred_voxels == 0:  # 如果没有预测体素，跳过
                continue

            # 统计在任何GT物体边界框内的预测体素数量
            correct_voxels = 0
            for i in range(num_pred_voxels):
                x, y, z = pred_voxel_coords[0][i], pred_voxel_coords[1][i], pred_voxel_coords[2][i]
                if self._is_voxel_in_any_box((x, y, z), gt_boxes):
                    correct_voxels += 1

            # 累加统计结果
            total_correct_pred_voxels += correct_voxels
            total_pred_voxels += num_pred_voxels

            if file_names_list is not None:
                visualize_path = os.path.join("./outputs/people_bbox", f"sample_{file_name}.png")
                self._visualize_3d_prediction(pred_mask_np, gt_boxes, visualize_path)

        # 计算准确率
        if total_pred_voxels > 0:
            accuracy = total_correct_pred_voxels / total_pred_voxels
        else:
            accuracy = 0.0

        return torch.tensor(accuracy, device=self.tps_ssc.device)

    def _get_gt_object_boxes(self, gt_mask):
        """获取GT中每个物体的3D边界框（以体素坐标表示）"""
        # 对GT掩码进行连通域分析，得到每个物体
        labeled = label(gt_mask, connectivity=3)
        if labeled.max() == 0:  # 没有物体
            return []

        boxes = []
        # 为每个物体计算边界框
        for obj_id in range(1, labeled.max() + 1):
            obj_mask = (labeled == obj_id)
            coords = np.where(obj_mask)
            if not coords or len(coords[0]) == 0:  # 物体没有体素
                continue

            # 计算边界框（体素坐标）
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            z_min, z_max = coords[2].min(), coords[2].max()

            # 保存边界框（x_min, y_min, z_min, x_max, y_max, z_max）
            boxes.append((x_min, y_min, z_min, x_max, y_max, z_max))

        return boxes

    def _is_voxel_in_any_box(self, voxel_coord, boxes):
        """检查体素是否在任何一个边界框内"""
        x, y, z = voxel_coord
        for box in boxes:
            x_min, y_min, z_min, x_max, y_max, z_max = box
            if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                return True
        return False

    # 动态物体可视化代码
    def _visualize_3d_prediction(self, pred_mask, gt_boxes, save_path=None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 获取预测体素坐标
        pred_voxels = np.where(pred_mask > 0)

        # 绘制真实物体边界框
        for box in gt_boxes:
            x_min, y_min, z_min, x_max, y_max, z_max = box
            # 绘制边界框的8个顶点和12条边
            edges = [
                [(x_min, y_min, z_min), (x_max, y_min, z_min)],
                [(x_min, y_max, z_min), (x_max, y_max, z_min)],
                [(x_min, y_min, z_max), (x_max, y_min, z_max)],
                [(x_min, y_max, z_max), (x_max, y_max, z_max)],
                [(x_min, y_min, z_min), (x_min, y_max, z_min)],
                [(x_max, y_min, z_min), (x_max, y_max, z_min)],
                [(x_min, y_min, z_max), (x_min, y_max, z_max)],
                [(x_max, y_min, z_max), (x_max, y_max, z_max)],
                [(x_min, y_min, z_min), (x_min, y_min, z_max)],
                [(x_max, y_min, z_min), (x_max, y_min, z_max)],
                [(x_min, y_max, z_min), (x_min, y_max, z_max)],
                [(x_max, y_max, z_min), (x_max, y_max, z_max)],
            ]
            for edge in edges:
                x_vals = [edge[0][0], edge[1][0]]
                y_vals = [edge[0][1], edge[1][1]]
                z_vals = [edge[0][2], edge[1][2]]
                ax.plot(x_vals, y_vals, z_vals, 'r-', linewidth=1)

        # 绘制预测体素，根据是否在框内着色
        correct_voxels = []
        incorrect_voxels = []
        for i in range(len(pred_voxels[0])):
            x, y, z = pred_voxels[0][i], pred_voxels[1][i], pred_voxels[2][i]
            if self._is_voxel_in_any_box((x, y, z), gt_boxes):
                correct_voxels.append((x, y, z))
            else:
                incorrect_voxels.append((x, y, z))

        # 绘制正确和错误的预测体素
        print(f"Total correct voxels: {len(correct_voxels)}")
        if correct_voxels:
            c_vox = np.array(correct_voxels).T
            ax.scatter(c_vox[0], c_vox[1], c_vox[2], c='g', marker='o', s=10, label='Correct Predictions')
        print(f"Total incorrect voxels: {len(incorrect_voxels)}")
        if incorrect_voxels:
            i_vox = np.array(incorrect_voxels).T
            ax.scatter(i_vox[0], i_vox[1], i_vox[2], c='r', marker='o', s=10, label='Incorrect Predictions')

        # 设置坐标轴和标题
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Object-based Predictions')
        ax.legend()
        if save_path:
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            # 保存图像
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
            plt.close()
        else:
            plt.show()

    # 原有方法保持不变
    def _calculate_sc_scores(self, preds, targets, nonempty=None):
        preds = preds.clone()
        targets = targets.clone()
        bs = preds.shape[0]

        mask = targets == self.ignore_index
        preds[mask] = 0
        targets[mask] = 0

        preds = preds.flatten(1)
        targets = targets.flatten(1)
        preds = torch.where(preds > 0, 1, 0)
        targets = torch.where(targets > 0, 1, 0)

        tp, fp, fn = 0, 0, 0
        for i in range(bs):
            pred = preds[i]
            target = targets[i]
            if nonempty is not None:
                nonempty_ = nonempty[i].flatten()
                pred = pred[nonempty_]
                target = target[nonempty_]
            pred = pred.bool()
            target = target.bool()

            tp += torch.logical_and(pred, target).sum()
            fp += torch.logical_and(pred, ~target).sum()
            fn += torch.logical_and(~pred, target).sum()
        return tp, fp, fn

    def _calculate_ssc_scores(self, preds, targets, nonempty=None):
        preds = preds.clone()
        targets = targets.clone()
        bs = preds.shape[0]
        C = self.num_classes

        mask = targets == self.ignore_index
        preds[mask] = 0
        targets[mask] = 0

        preds = preds.flatten(1)
        targets = targets.flatten(1)

        tp = torch.zeros(C, dtype=torch.int).to(preds.device)
        fp = torch.zeros(C, dtype=torch.int).to(preds.device)
        fn = torch.zeros(C, dtype=torch.int).to(preds.device)
        for i in range(bs):
            pred = preds[i]
            target = targets[i]
            if nonempty is not None:
                mask = nonempty[i].flatten() & (target != self.ignore_index)
                pred = pred[mask]
                target = target[mask]
            for c in range(C):
                tp[c] += torch.logical_and(pred == c, target == c).sum()
                fp[c] += torch.logical_and(pred == c, ~(target == c)).sum()
                fn[c] += torch.logical_and(~(pred == c), target == c).sum()
        return tp, fp, fn