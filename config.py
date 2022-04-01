# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor_list = [
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0
]
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "Meta_RDN_baeline"

if mode == "train":
    # Dataset
    train_image_dir = f"data/DIV2K/Meta_RDN/train"
    valid_image_dir = f"data/DIV2K/Meta_RDN/valid"
    test_lr_image_dir = f"data/Set14/LRbicx2"
    test_hr_image_dir = f"data/Set14/GTmod12"

    image_size = 50
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""

    # Total num epochs
    epochs = 100

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    # StepLR scheduler parameter
    lr_scheduler_step_size = 20
    lr_scheduler_gamma = 0.5

    print_frequency = 1000

if mode == "valid":
    # Test data address
    lr_dir = f"data/Set14/LRbicx2"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set14/GTmod12"

    model_path = f"results/{exp_name}/best.pth.tar"
