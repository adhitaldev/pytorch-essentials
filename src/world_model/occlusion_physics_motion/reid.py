# =========================================================
# reid.py — ReID Embeddings (Improved Stability)
# =========================================================

import numpy as np
import torch
import torchreid
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ReIDModel:

    def __init__(self):
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.model.eval()
        print("[ReID] Model loaded.")

    def get_embedding(self, crop):
        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]
        if h < 32 or w < 16:
            return None

        try:
            t = transform(crop).unsqueeze(0)
            with torch.no_grad():
                emb = self.model(t)
                emb = emb.cpu().numpy().flatten()

            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                return None

            return emb / norm

        except Exception:
            return None

    def similarity(self, emb_a, emb_b):
        if emb_a is None or emb_b is None:
            return 0.0
        return float(np.dot(emb_a, emb_b))