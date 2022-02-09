import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import torch.nn as nn
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

class Optical:
    
    # def __init__(self):
        # print ('Initializing RAFT network...')
        # ROOT_DIR = os.getcwd()
        # ROOT_DIR = "/mnt/RAFT"
        # print(ROOT_DIR)
        
        # MODEL_DIR = os.path.join(ROOT_DIR, "models/raft-things.pth")
        # PATH_DIR = os.path.join(ROOT_DIR, "TUM")   #/mnt/RAFT/TUM
        
        # dict = {'model': MODEL_DIR, 'path': PATH_DIR, 'small': 'store_true', 'mixed_precision': 'store_true', 'alternate_corr': 'store_true'}



    def load_image(self, imgfile):
        img = np.array(Image.open(imgfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def GetFlow(self, image1, image2):
        
        self.model = torch.nn.DataParallel(RAFT(**dict))
        
        # loaded_state = {k.replace('module.', ''): v for k, v in torch.load(dict['model']).items()}
        # self.model.load_state_dict(loaded_state, strict=False)
        
        # self.model.load_state_dict(torch.load(dict['model']), strict=False)
        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()
        # 求光流
        with torch.no_grad():

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            print('step0')
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            print('step1')
            
            img = image1[0].permute(1,2,0).cpu().numpy()
            flo = flow_up[0].permute(1,2,0).cpu().numpy()
            
            # map flow to rgb image
            flo = flow_viz.flow_to_image(flo)    # <class 'numpy.ndarray'>
            
            # print(type(flo))
            
            return flo

def main(**dict):
    
    flow = Optical()
    images = glob.glob(os.path.join(dict['path'], '*.png')) + \
             glob.glob(os.path.join(dict['path'], '*.jpg'))
    
    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        image1 = flow.load_image(imfile1)
        image2 = flow.load_image(imfile2)
        flo = flow.GetFlow(image1, image2)
        
    for i in range(4):
        cv2.imwrite("/mnt/RAFT/TUM/results/%s.png" % i, flo)
    
if __name__ == "__main__":
    
    print ('Initializing RAFT network...')
    ROOT_DIR = os.getcwd()
    ROOT_DIR = "/mnt/RAFT"
    print(ROOT_DIR)
    
    MODEL_DIR = os.path.join(ROOT_DIR, "models/raft-things.pth")
    PATH_DIR = os.path.join(ROOT_DIR, "TUM")   #/mnt/SceneFlow/src/RAFT/TUM
    
    
    dict = { 
            'model': MODEL_DIR, 
            'path': PATH_DIR, 
            'small': True, 
            'mixed_precision': True, 
            'alternate_corr': True
           }
    
    main(**dict)
