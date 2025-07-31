import torch.nn as nn

from ssc_pl.models.utils import rasterize_gaussians, prepare_gs_attribute


class MLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=32, output_dim=12):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GaussianRenderHead(nn.Module):
    def __init__(self, dim, segment_head=False,):
        super().__init__()
        self.segment_head = MLP(input_dim=256, hidden_dim=128, output_dim=12) if segment_head else None

    def forward(self, representation, metas=None, **kwargs):
        pc = prepare_gs_attribute(representation[-1]['gaussian'])
        # viewpoint_camera = setup_opengl_proj(w = 640, h = 480, k = metas['cam_K'][0], w2c = metas['cam_pose'][0], near=0.0, far=80)
        means3d = pc['get_xyz'][0].float()
        features = pc['semantic'][0].float()
        # features = features @ pca_matrix.to(features)
        opacities = pc['get_opacity'].squeeze().float()
        scales = pc['get_scaling'][0].float()
        rotations = pc['get_rotation'][0].float()
        cam2img = metas['cam_K'][0].unsqueeze(0).float()
        # viewmat = viewpoint_camera['world_view_transform'].unsqueeze(0).float()
        viewmat = metas['cam_pose'][0].unsqueeze(0).float()

        rendered = rasterize_gaussians(
            means3d,
            features,
            opacities,
            scales,
            rotations,
            cam2img,
            viewmat,
            img_aug_mats=None,
            image_size=(480, 640),
            near_plane=0.0,
            far_plane=80,
            render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
            channel_chunk=32).flatten(0, 1)
        depth = rendered[:, :, -1:]
        rendered = rendered[:, :, :-1]
        depth = depth.clamp(min=0.0, max=80)

        # print(f'rendered.shape: {rendered.shape}')
        # print(f'depth.shape: {depth.shape}')


        if self.segment_head:
            seg_rendered = self.segment_head(rendered).unsqueeze(0)
        # print(f'seg_rendered.shape: {seg_rendered.shape}')

        return {
            'rendered_feature': rendered,
            'rendered_depth': depth,
            'rendered_seg': seg_rendered,
        }