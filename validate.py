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
"""File description: Realize the verification function after model training."""
import os

import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
from model import MetaRDN


def main() -> None:
    # Initialize the super-resolution model
    model = MetaRDN().to(config.device)
    print("Build MetaRDN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load MetaRDN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()

    # Initialize the image evaluation index.
    total_psnr = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.hr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(hr_image_path)}`...")
        # Read LR image and HR image
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        lr_image_height, lr_image_width, _ = lr_image.shape
        hr_image_height, hr_image_width, _ = hr_image.shape
        upscale_factor = hr_image_height // lr_image_height

        # Convert BGR image to Y image
        hr_y_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=True)

        # Convert BGR image to RGB image
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Convert Y image data convert to Y tensor data
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False).to(config.device).unsqueeze_(0)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False).to(config.device).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False).to(config.device).unsqueeze_(0)

        # Get the position matrix, mask
        batch_size, channels, lr_height, lr_width = lr_tensor.size()
        _, _, hr_height, hr_width = hr_tensor.size()
        pos_matrix, mask_matrix = imgproc.weight_prediction_matrix_from_lr(lr_height, lr_width, upscale_factor)
        pos_matrix = pos_matrix.to(config.device, non_blocking=True)
        mask_matrix = mask_matrix.to(config.device, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = model(lr_tensor, pos_matrix, upscale_factor)
            sr_tensor = torch.masked_select(sr_tensor, mask_matrix)
            sr_tensor = sr_tensor.contiguous().view(batch_size, channels, hr_height, hr_width)
            # sr_tensor = sr_tensor.clamp_(0, 1.0)

        # Save image
        sr_image = imgproc.tensor2image(sr_tensor, range_norm=False, half=False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal PSNR
        sr_image = sr_image.astype(np.float32) / 255.
        sr_y_image = imgproc.bgr2ycbcr(sr_image, use_y_channel=True)
        sr_y_tensor = imgproc.image2tensor(sr_y_image, range_norm=False, half=False).to(config.device).unsqueeze_(0)

        total_psnr += 10. * torch.log10(1. / torch.mean((sr_y_tensor - hr_y_tensor) ** 2))

    print(f"PSNR: {total_psnr / total_files:4.2f}dB.\n")


if __name__ == "__main__":
    main()
