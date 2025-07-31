import torch, torch.nn as nn
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid


@MODELS.register_module()
class GaussianLifter(BaseLifter):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        
        # gaussian generation number
        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        if phi_activation == 'sigmoid':
            xyz = safe_inverse_sigmoid(xyz)
        elif phi_activation == 'loop':
            xyz[:, :2] = safe_inverse_sigmoid(xyz[:, :2])
        else:
            raise NotImplementedError
        self.xyz = xyz
        scale = torch.rand_like(xyz)
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        self.semantic_dim = semantic_dim
        # xyz=(0,1), scale=(0,1), rots, opacity=0.1, semantic=17
        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)

        self.num_anchor = num_anchor
        # (Properties)
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        # (Queries)instance_feature.shape=[25600, 128]
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        # 51200,12 -> 51200,1 opacity    sigmoid
        # 51200,12 -> 51200,3 scale      softplus
        # 51200,12 -> 51200,4 rot        Norm
        self.opacity_head = nn.Linear(self.semantic_dim, 1)
        self.scale_head = nn.Linear(self.semantic_dim, 3)
        self.rot_head = nn.Linear(self.semantic_dim, 4)
        


    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def update_semantic(self, new_semantic):
        # 确保 new_semantic 是一个合适的 tensor
        # 这里假设 new_semantic 已经是正确的形状和类型
        if new_semantic is None:
            return

        assert new_semantic.shape == (self.num_anchor, self.semantic_dim), "new_semantic shape does not match."
        # import pdb;
        # pdb.set_trace()
        # 分离出 anchor 中的其他部分
        xyz_scale_rots_opacity = self.anchor[:, :-self.semantic_dim]
        
        # 创建一个新的 anchor，它是 xyz_scale_rots_opacity 和 new_semantic 的组合
        new_anchor = torch.cat([xyz_scale_rots_opacity, new_semantic], dim=-1)
        
        # 更新 self.anchor
        # 由于 self.anchor 是一个 nn.Parameter，我们需要用 nn.Parameter 包装更新后的值
        self.anchor = torch.nn.Parameter(new_anchor, requires_grad=self.anchor.requires_grad)

    def update_opacity(self, new_opacity):
        # 确保 new_opacity 是一个合适的 tensor
        # 这里假设 new_opacity 已经是正确的形状和类型
        if new_opacity is None:
            return

        # 假设 opacity 在 anchor 中的位置紧跟在 rots 后面
        # 首先，我们需要找到 opacity 的位置
        opacity_index = 7  # xyz(3) + scale(3) + rots(4) 的索引位置为 7
        # 分离出 anchor 中的其他部分
        before_opacity = self.anchor[:, :opacity_index]
        after_opacity = self.anchor[:, opacity_index+1:]  # 跳过原有的 opacity

        # 创建一个新的 anchor，它是 before_opacity, new_opacity 和 after_opacity 的组合
        new_anchor = torch.cat([before_opacity, new_opacity, after_opacity], dim=-1)
        
        # 更新 self.anchor
        self.anchor = torch.nn.Parameter(new_anchor, requires_grad=self.anchor.requires_grad)


    def update_rot(self, new_rot):
        # 确保 new_rot 是一个合适的 tensor
        if new_rot is None:
            return

        # 假设 rot 在 anchor 中的位置紧跟在 scale 后面
        rot_index_start = 3  # xyz(3) 的索引结束位置
        rot_index_end = 7  # xyz(3) + rot(4) 的索引结束位置为 7
        # 分离出 anchor 中的其他部分
        before_rot = self.anchor[:, :rot_index_start]
        after_rot = self.anchor[:, rot_index_end:]  # 跳过原有的 rot

        # 创建一个新的 anchor，它是 before_rot, new_rot 和 after_rot 的组合
        new_anchor = torch.cat([before_rot, new_rot, after_rot], dim=-1)
        
        # 更新 self.anchor
        self.anchor = torch.nn.Parameter(new_anchor, requires_grad=self.anchor.requires_grad)

    def update_scale(self, new_scale):
        # 确保 new_scale 是一个合适的 tensor
        if new_scale is None:
            return

        # 假设 scale 在 anchor 中的位置紧跟在 xyz 后面
        scale_index_start = 3  # xyz(3) 的索引结束位置
        scale_index_end = 6  # xyz(3) + scale(3) 的索引结束位置为 6
        # 分离出 anchor 中的其他部分
        before_scale = self.anchor[:, :scale_index_start]
        after_scale = self.anchor[:, scale_index_end:]  # 跳过原有的 scale

        # 创建一个新的 anchor，它是 before_scale, new_scale 和 after_scale 的组合
        new_anchor = torch.cat([before_scale, new_scale, after_scale], dim=-1)
        
        # 更新 self.anchor
        self.anchor = torch.nn.Parameter(new_anchor, requires_grad=self.anchor.requires_grad)

    def update_xyz(self, new_xyz):
        # 确保 new_xyz 是一个合适的 tensor
        # 这里假设 new_xyz 已经是正确的形状和类型
        if new_xyz is None:
            return

        assert new_xyz.shape == (self.num_anchor, 3), "new_xyz shape does not match."
        
        # 分离出 anchor 中的其他部分
        after_xyz = self.anchor[:, 3:]  # xyz 占据前三个位置
        
        # 创建一个新的 anchor，它是 new_xyz 和 after_xyz 的组合
        new_anchor = torch.cat([new_xyz, after_xyz], dim=-1)
        
        # 更新 self.anchor
        self.anchor = torch.nn.Parameter(new_anchor, requires_grad=self.anchor.requires_grad)


    def forward(self, ms_img_feats, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        # anchor.requires_grad_(True)
        # instance_feature.requires_grad_(True)
        # import pdb;
        # pdb.set_trace()

        return {
            'rep_features': instance_feature,
            'representation': anchor,
        }


        # 33 kuailong  55genggui 