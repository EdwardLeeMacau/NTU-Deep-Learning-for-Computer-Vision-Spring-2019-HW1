"""
  FileName     [ pytorch_tutorial.py ]
  PackageName  [ DLCV ]
  Synopsis     [  ]

  Library:
  * np          version: 
  * torch       version: 1.0.0

  1. Finetuning the pretrained model
  2. Data augmentation
  3. Training with multiple GPU
  4. Exporting models to other platforms

  Caffe (UC Berkeley) -> Caffe2     (Facebook)
  Torch (NYU)         -> PyTorch    (Facebook)
  Theano              -> Tensorflow (Google) 

  CUDA 10.0 in linux 
"""

""" Numpy """
import numpy as np

""" pyTorch """
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
N, D = 3, 4