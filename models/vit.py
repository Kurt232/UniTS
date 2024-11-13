import clip
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from PIL import Image
from torchvision.transforms import transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class Model(nn.Module):
    def __init__(self, clip_model='ViT_L/14'):
        super().__init__()
        
        # 1. vision transformer
        self.visual = clip.load(clip_model)[0].visual

        feat_dim = self.visual.proj.shape[1]
        # 2. classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 7)
        )

        for para in self.parameters():
            para.data = para.data.float() # bf16 is not trainable
    
    def forward(self, imgs):
        visual_feat = self.visual(imgs)
        logits = self.classifier(visual_feat)
        return logits

if __name__ == "__main__":
    model = Model(clip_model="ViT-B/16")
    
    # 224x224 image
    imgs = torch.randn(1, 3, 224, 224)
    logits = model(imgs)
    print(logits.shape)  # torch.Size