import os
import random
from tqdm import tqdm
from datetime import datetime   
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from mmengine import Config
from torch.utils.tensorboard import SummaryWriter
import sys
import open_clip
from open_clip import tokenize
import configargparse

# Configuration
random.seed(0)

# Global writer initialization
writer = None

templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


classes = {
    'NuScenesDataset': [
        'empty', 'barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 
        'pedestrian', 'traffic cone', 'trailer', 'truck', 'drivable surface', 
        'other flat', 'sidewalk', 'terrain', 'man-made', 'vegetation', 'unknown'
    ],
    'NYUDataset':['ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furn', 'object']

}


class TextEncoder:
    def __init__(self, args):
        self.args = args
        self.prefix = self.name_prefix()

        self.model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        self.model.eval()
        self.model.cuda()

        self.zeroshot_weights = None

    def name_prefix(self):
        # Format the current timestamp. For example: "20240331-235959"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.args.feat_type}_{timestamp}"

    def encoder(self):
        self.zeroshot_weights = []
        
        with torch.no_grad():
            for classname in classes[self.args.dataset]:
                texts = [template.format(classname) for template in templates]
                texts = tokenize(texts).cuda()
                class_embeddings = self.model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                self.zeroshot_weights.append(class_embedding)
            self.zeroshot_weights = torch.stack(self.zeroshot_weights, dim=0).cuda()

    def save_prompt_embeddings(self):
        output_dir = self.args.output_embedding_dir
        torch.save(self.zeroshot_weights, os.path.join(output_dir, f'{self.prefix}_prompt_embeddings.pt'))


def parse_args():
    parser = configargparse.ArgParser(description="Training script parameters")

    parser.add_argument('--dataset', type=str, default='NYUDataset')
    parser.add_argument('--output_embedding_dir', type=str, default='checkpoints')
    parser.add_argument('--feat_type', type=str, default='clip')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    textencoder = TextEncoder(args)
    textencoder.encoder()
    textencoder.save_prompt_embeddings()