#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Author: Weiliang Luo
    Student No.: 1900011804
    Date changed: 2021.5.26
    Python Version: Anaconda3 (Python 3.8.8)
'''

# Imports
# std libs
import os

# 3rd Party libs
import numpy as np
from skimage.restoration import denoise_wavelet
from skimage.io import imread
from matplotlib.pyplot import imsave

# my modules



class WD():
    # 小波去噪
    def __init__(self):
        self.sigma = 0.01
        self.img_path = './original_img/'
        self.save_path = './denoised_img/'

    def process(self, img):
        return denoise_wavelet(img, multichannel=True, convert2ycbcr=True,
                               method='BayesShrink', mode='soft',sigma=self.sigma)

    def process_from_file(self, path=None):
        if path == None:
            path = self.img_path
        names = os.listdir(path)
        for name in names:
            ImgID = name.split('.')[0]
            img = imread(path + name)
            img2 = self.process(img)
            imsave(os.path.join(self.save_path, ImgID + '_wd.jpg'), img2)
