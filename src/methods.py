import torch
import numpy as np

def psnr(img1, img2):
    if torch.is_tensor(img1):
        mse = ((img1 - img2)** 2).mean()
        if mse == 0:
            return 100
        if img1.max() > 1.0 or img2.max() > 1.0:
            ceiling = 255.0
        else:
            ceiling = 1.0
        return 20 * torch.log10(ceiling/ torch.sqrt(mse))
    else:
        mse = ((img1 - img2)** 2).mean()
        if mse == 0:
            return 100
        if img1.max() > 1.0 or img2.max() > 1.0:
            ceiling = 255.0
        else:
            ceiling = 1.0
        return 20 * np.log10(ceiling/ np.sqrt(mse))