import edsr
import numpy as np
import torch
from skimage.io import imread
from matplotlib.pyplot import imsave
import torchvision.transforms.functional as F
from torch.autograd import Variable

class Args():
    def __init__(self, r, f, s, rgb_range=255, n_colors=3, res_scale=1):
        self.n_resblocks = r
        self.n_feats = f
        self.scale = [s]
        self.rgb_range = rgb_range
        self.n_colors = n_colors
        self.res_scale = res_scale

model_path = './pretrain/edsr_baseline_x2.pt'
img_path = './denoised_img/img63_wd.jpg'
save_path = './test.png'
args = Args(16, 64, 2)
model = edsr.make_model(args)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()
img = F.to_tensor(imread(img_path))
img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
img2 = model(img)
img2 = img2.detach().squeeze(0).clamp(0,1).numpy()
img2 = np.transpose(img2, [1, 2, 0])
imsave(save_path, img2)