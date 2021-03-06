import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from torchvision import models

# Download and cache pretrained model from PyTorch model zoo
model = models.resnet18(pretrained=True)
import io

import numpy as np
import requests
import torch

from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.features import CNormalizerMeanStd

# Set random seed for pytorch and numpy
np.random.seed(0)
torch.manual_seed(0)

# imagenet normalization
normalizer = CNormalizerMeanStd(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))

# wrap the model, including the normalizer
clf = CClassifierPyTorch(model=model,
                         input_shape=(3, 224, 224),
                         softmax_outputs=False,
                         preprocess=normalizer,
                         random_state=0,
                         pretrained=True)

# load the imagenet labels
import json

imagenet_labels_path = "https://raw.githubusercontent.com/" \
                       "anishathalye/imagenet-simple-labels/" \
                       "master/imagenet-simple-labels.json"
proxy_servers = {'http': '127.0.0.1:7890', 'https': '127.0.0.1:7890'}
r = requests.get(imagenet_labels_path, proxies=proxy_servers)
labels = json.load(io.StringIO(r.text))

import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms


# create a function to show images
def plot_img(f, x, label):
    x = np.transpose(x.tondarray().reshape((3, 224, 224)), (1, 2, 0))
    f.sp.title(label)
    f.sp.imshow(x)
    return f


def define_lb_ub(image, x_low_b, x_up_b, y_low_b, y_up_b, low_b, up_b, n_channels=3):
    # reshape the img (it is stored as a flat vector)
    image = image.tondarray().reshape((3, 224, 224))

    # assign to the lower and upper bound the same values of the image pixels
    low_b_patch = deepcopy(image)
    up_b_patch = deepcopy(image)

    # for each image channel, set the lower bound of the pixels in the
    # region defined by x_low_b, x_up_b, y_low_b, y_up_b equal to lb and
    # the upper  bound equal to up in this way the attacker will be able
    # to modify only the pixels in this region.
    for ch in range(n_channels):
        low_b_patch[ch, x_low_b:x_up_b, y_low_b:y_up_b] = low_b
        up_b_patch[ch, x_low_b:x_up_b, y_low_b:y_up_b] = up_b

    return CArray(np.ravel(low_b_patch)), CArray(np.ravel(up_b_patch))


from copy import deepcopy
from secml.figure import CFigure
from secml.array import CArray

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
from torch.utils.data import DataLoader
import random
import time
dataset = torchvision.datasets.ImageFolder(root='./data/imagewang/train', transform=transform)
dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
seed = int(time.time() * 256)
torch.manual_seed(seed)

name_dict = dict()
cnt = 0
suc_cnt = 0
mod_cnt = 0

for img, label in dataset_loader:
    cnt += 1
    # transform the image into a vector
    img = torch.unsqueeze(img, 0).view(-1)
    img = CArray(img.numpy())

    # get the classifier prediction
    preds = clf.predict(img)
    pred_class = preds.item()
    pred_label = labels[pred_class]

    # show the original image

    # Only required for visualization in notebooks
    # %matplotlib inline

    #fig = CFigure(height=4, width=4, fontsize=14)
    #plot_img(fig, img, label=pred_label)
    #fig.show()

    attack_type = 'PGD'

    # attack_type = 'PGD'
    # attack_type = 'CW'
    # attack_type = 'None'

    from cleverhans.attacks import CarliniWagnerL2

    from secml.adv.attacks import CAttackEvasion
    from secml.explanation import CExplainerIntegratedGradients

    lb = 0
    ub = 1

    target_idx = random.randint(0, 999)  # random label

    attack_id = ''
    attack_params = {}

    if attack_type == "CW":
        attack_id = 'e-cleverhans'
        attack_params = {'max_iterations': 50, 'learning_rate': 0.005,
                         'binary_search_steps': 1, 'confidence': 1e6,
                         'abort_early': False, 'initial_const': 0.2,
                         'y_target': target_idx,
                         'clip_min': lb, 'clip_max': ub,
                         'clvh_attack_class': CarliniWagnerL2}

    if attack_type == 'PGD':
        attack_id = 'e-pgd'
        solver_params = {
            'eta': 1e-2,
            'max_iter': 50,
            'eps': 1e-6}
        attack_params = {'double_init': False,
                         'distance': 'l2',
                         'dmax': 0.875227,
                         'lb': lb,
                         'ub': ub,
                         'y_target': target_idx,
                         'solver_params': solver_params}

    if attack_type == 'PGD-patch':
        attack_id = 'e-pgd'
        # create the mask that we will use to allows the attack to modify only
        # a restricted region of the image
        x_lb = 140;
        x_ub = 160;
        y_lb = 10;
        y_ub = 80
        dmax_patch = 5000
        lb_patch, ub_patch = define_lb_ub(
            img, x_lb, x_ub, y_lb, y_ub, lb, ub, n_channels=3)
        solver_params = {
            'eta': 0.8,
            'max_iter': 50,
            'eps': 1e-6}

        attack_params = {'double_init': False,
                         'distance': 'l2',
                         'dmax': dmax_patch,
                         'lb': lb_patch,
                         'ub': ub_patch,
                         'y_target': target_idx,
                         'solver_params': solver_params}

    attack = CAttackEvasion.create(
        attack_id,
        clf,
        **attack_params)

    # run the attack
    eva_y_pred, _, eva_adv_ds, _ = attack.run(img, pred_class)
    adv_img = eva_adv_ds.X[0, :]

    # get the classifier prediction
    advx_pred = clf.predict(adv_img)
    advx_label_idx = advx_pred.item()
    adv_pred_label = labels[advx_label_idx]
    
    if adv_pred_label == pred_label:
        cnt -= 1
        continue

    # compute the explanations w.r.t. the target class
    #explainer = CExplainerIntegratedGradients(clf)
    #expl = explainer.explain(adv_img, y=target_idx, m=750)

    #fig = CFigure(height=8, width=20, fontsize=14)

    #fig.subplot(2, 4, 1)
    # plot the original image
    #fig = plot_img(fig, img, label=pred_label)

    # compute the adversarial perturbation
    #adv_noise = adv_img - img

    # normalize perturbation for visualization
    #diff_img = img - adv_img
    #diff_img -= diff_img.min()
    #diff_img /= diff_img.max()

    # plot the adversarial perturbation
    #fig.subplot(2, 4, 2)
    #fig = plot_img(fig, diff_img, label='adversarial perturbation')

    #fig.subplot(2, 4, 3)
    # plot the adversarial image
    #fig = plot_img(fig, adv_img, label=adv_pred_label)

    #fig.subplot(2, 4, 4)

    #expl = np.transpose(expl.tondarray().reshape((3, 224, 224)), (1, 2, 0))
    #r = np.fabs(expl[:, :, 0])
    #g = np.fabs(expl[:, :, 1])
    #b = np.fabs(expl[:, :, 2])

    # Calculate the maximum error for each pixel
    #expl = np.maximum(np.maximum(r, g), b)
    #fig.sp.title('explanations')
    #fig.sp.imshow(expl, cmap='seismic')

    # show wavelet-denoising result
    #fig.subplot(2, 4, 5)
    adv_img2 = np.transpose(adv_img.tondarray().reshape((3, 224, 224)), (1, 2, 0))

    import srload
    sr = srload.SR('edsr_baseline')
    transfer_sr = srload.SR('mdsr_baseline')

    T = 50
    alpha = 8/255
    loss_fun = torch.nn.MSELoss()
    lr_image0 = transfer_sr.half_process(adv_img2)
    lr_image = transfer_sr.half_process(adv_img2)
    sr_image0 = transfer_sr.model(lr_image)
    lr_rand = torch.Tensor(np.random.uniform(-0.2 / 255, 0.2 / 255, lr_image.shape)).type_as(lr_image)
    lr_image = lr_image + lr_rand
    lr_image.requires_grad = True

    for temT in range(T):
        # print(lr)
        sr_image = transfer_sr.model(lr_image).squeeze(0)
        transfer_sr.model.zero_grad()
        '''
        if (cfg.type == 'part'):
            sr = sr * outer_mask
        '''
        loss = torch.sqrt(loss_fun(sr_image, sr_image0) * sr_image.numel())  # L2??????
        if temT == 0:
            print('start loss: {:.3f}'.format(loss.data))
        loss.backward(retain_graph=True)
        data_grad = lr_image.grad.data
        # print(data_grad)
        lr_image = lr_image.detach()
        lr_imagen = torch.clamp(lr_image + torch.sign(data_grad) * alpha / T, 0, 1)
        # print(lrn)
        lr_image = (torch.clamp((lr_imagen - lr_image0), -alpha, alpha)) + lr_image0
        lr_image = torch.autograd.Variable(lr_image)
        lr_image.requires_grad = True
    
    print('final loss: {:.3f}'.format(loss.data))
    sr_image = transfer_sr.last_process(transfer_sr.model(lr_image))
    lr_image = transfer_sr.last_process(lr_image, pool=False)
    lr_image0 = transfer_sr.last_process(lr_image0, pool=False)
    sr_image0 = transfer_sr.last_process(sr_image0)
    
    '''
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.title('sr_image_modified')
    plt.imshow(sr_image)
    plt.subplot(1, 4, 2)
    plt.title('sr_image_original')
    plt.imshow(sr_image0)
    plt.subplot(1, 4, 3)
    plt.title('lr_image_adv')
    plt.imshow(lr_image)
    plt.subplot(1, 4, 4)
    plt.title('lr_image')
    plt.imshow(lr_image0)
    plt.show()
    '''

    from wavelet_denoise import WD
    denoiser = WD()
    denoised_img = denoiser.process(adv_img2)
    denoised_img = denoised_img.clip(0, 1)
    #fig.sp.title('wavelet_denoised')
    #fig.sp.imshow(denoised_img)

    # show SR result
    #fig.subplot(2, 4, 6)
    
    


    processed_img = sr.process(denoised_img)
    pc_img = np.transpose(processed_img, (2, 0, 1))
    pc_img = CArray(pc_img.reshape(1, -1))
    new_preds = clf.predict(pc_img)
    new_pred_class = new_preds.item()
    new_pred_label = labels[new_pred_class]
    #fig.sp.title('reconstructed: {}'.format(new_pred_label))
    #fig.sp.imshow(processed_img)

    # show noise after WD-SR
    #fig.subplot(2, 4, 7)
    #img2 = np.transpose(img.tondarray().reshape((3, 224, 224)), (1, 2, 0))
    #diff_img2 = processed_img - img2
    #diff_img2 -= diff_img2.min()
    #diff_img2 /= diff_img2.max()
    #fig.sp.title('new noise')
    #fig.sp.imshow(diff_img2)

    # show expl
    #fig.subplot(2, 4, 8)
    #expl2 = explainer.explain(pc_img, y=target_idx, m=750)
    #expl2 = np.transpose(expl2.tondarray().reshape((3, 224, 224)), (1, 2, 0))
    #r2 = np.fabs(expl2[:, :, 0])
    #g2 = np.fabs(expl2[:, :, 1])
    #b2 = np.fabs(expl2[:, :, 2])

    # Calculate the maximum error for each pixel
    #expl2 = np.maximum(np.maximum(r2, g2), b2)
    #fig.sp.title('explanations')
    #fig.sp.imshow(expl2, cmap='seismic')
    new2_pred_label = labels[clf.predict(CArray(np.transpose(sr.process(denoiser.process(lr_image).clip(0, 1)), (2, 0, 1)).reshape(1, -1))).item()]
    # show
    # fig.show()
    #defend_path = './defend/'
    if new_pred_label == pred_label:
        suc_cnt += 1
        if new2_pred_label != pred_label:
            mod_cnt += 1
        #fig.savefig(defend_path + 'success/{}.png'.format(cnt))
    #else:
        #fig.savefig(defend_path + 'failed/{}.png'.format(cnt))

    if cnt <= 500:
        print('modified success: {} / defend success: {} / all: {}'.format(mod_cnt, suc_cnt, cnt))
        continue
    else:
        break

    from secml.ml.classifiers.loss import CSoftmax
    from secml.ml.features.normalization import CNormalizerMinMax

    n_iter = attack.x_seq.shape[0]
    itrs = CArray.arange(n_iter)

    # create a plot that shows the loss and the confidence during the attack iterations
    # note that the loss is not available for all attacks
    fig = CFigure(width=10, height=4, fontsize=14, linewidth=2)

    # apply a linear scaling to have the loss in [0,1]
    loss = attack.f_seq
    
    if loss is not None:
        loss = CNormalizerMinMax().fit_transform(CArray(loss).T).ravel()
        fig.subplot(1, 2, 1)
        fig.sp.xlabel('iteration')
        fig.sp.ylabel('loss')
        fig.sp.plot(itrs, loss, c='black')
    

    # classify all the points in the attack path
    scores = clf.predict(attack.x_seq, return_decision_function=True)[1]

    # we apply the softmax to the score to have value in [0,1]
    scores = CSoftmax().softmax(scores)

    
    fig.subplot(1, 2, 2)
    fig.sp.xlabel('iteration')
    fig.sp.ylabel('confidence')
    fig.sp.plot(itrs, scores[:, pred_class], linestyle='--', c='black')
    fig.sp.plot(itrs, scores[:, target_idx], c='black')
    

    fig.show()

    import matplotlib.pyplot as plt

    adv_img = np.transpose(adv_img.tondarray().reshape((3, 224, 224)), (1, 2, 0))
    plt.imshow(adv_img)  # ????????????
    plt.axis('off')  # ??????????????????
    if pred_label != adv_pred_label:
        save_path = "./Adversarial Examples/" + pred_label + '_to_' + adv_pred_label
        if (pred_label,adv_pred_label) not in name_dict:
            name_dict[(pred_label,adv_pred_label)] = 1
            save_path += '.png'
        else:
            name_dict[(pred_label,adv_pred_label)] += 1
            save_path += '_{}_.png'.format(name_dict[(pred_label,adv_pred_label)])
        plt.savefig(save_path)
        break
