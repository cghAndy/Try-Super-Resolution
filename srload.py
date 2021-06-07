import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import edsr, mdsr, rcan
import numpy as np
import torch
import skimage
import skimage.color as sc
import imageio
from skimage.transform import downscale_local_mean
from matplotlib.pyplot import imsave, imshow
from torch.utils.data import dataloader

class Args():
    def __init__(self, r, f, s, rgb_range=255, n_colors=3, res_scale=1):
        self.n_resblocks = r
        self.n_feats = f
        self.scale = [s]
        self.rgb_range = rgb_range
        self.n_colors = n_colors
        self.res_scale = res_scale
        self.range = 255
        self.n_resgroups = 1
        self.reduction = 16

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def quantize(img, rgb_range=255):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def max_pooling(img, G):

    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)
    out = np.zeros((Nh, Nw, C))

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[y, x, c] = np.max(img[G*y:G*(y+1), G*x:G*(x+1), c])  
    return out  

class SR():
    def __init__(self, model_type):
        if 'edsr' in model_type:
            if model_type == 'edsr_baseline':
                self.model_path = './pretrain/edsr_baseline_x2.pt'
                self.args = Args(16, 64, 2)
            elif model_type == 'edsr':
                self.model_path = './pretrain/edsr_x2.pt'
                self.args = Args(32, 256, 2)
            self.model = edsr.make_model(self.args)
        elif 'mdsr' in model_type:
            if model_type == 'mdsr_baseline':
                self.model_path = './pretrain/mdsr_baseline.pt'
                self.args = Args(16, 64, 2)
            if model_type == 'mdsr':
                self.model_path = './pretrain/mdsr.pt'
                self.args = Args(80, 64, 2)
            self.model = mdsr.make_model(self.args)
        elif 'rcan' in model_type:
            self.model_path = './pretrain/rcan_x2.pt'
            self.args = Args(20, 64, 2)
            self.args.res_group = 10
            self.model = rcan.make_model(self.args)
            
        state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.img_path = './denoised_img/img63_wd.jpg'
        self.save_path = './test.png'


    def process(self, i):
        if 'float' in str(i.dtype):
            i = (i * 255).astype(np.uint8)
        img = set_channel(i)[0]
        img = np2Tensor(img)
        testset = img
        d = dataloader.DataLoader(testset, batch_size=1, shuffle=False)
        for lr in d:
            img2 = self.model(lr)
        img2 = quantize(img2)
        img2 = img2.byte().squeeze().permute(1, 2, 0).numpy()
        ## img2 = skimage.img_as_float(img2)
        img2 = max_pooling(img2, 2) / 255
        return img2

if __name__ == '__main__':
    sr = SR('rcan')
    i = imageio.imread(sr.img_path)
    # i = skimage.img_as_float(i)
    a = sr.process(i)
    print(a.shape)
    plt.imshow(a)
    plt.show()
