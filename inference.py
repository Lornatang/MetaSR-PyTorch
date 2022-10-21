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
import argparse
import os

import cv2
import torch
from torch import nn

import model
from imgproc import weight_prediction_matrix_from_lr, preprocess_one_image, tensor_to_image
from utils import load_state_dict

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    sr_model = model.__dict__[model_arch_name](in_channels=3,
                                               out_channels=3,
                                               channels=64)
    sr_model = sr_model.to(device=device)

    return sr_model


def main(args):
    device = choice_device(args.device_type)

    # Initialize the model
    sr_model = build_model(args.model_arch_name, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    sr_model = load_state_dict(sr_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    sr_model.eval()

    lr_tensor = preprocess_one_image(args.inputs_path, device)

    # Get the position matrix, mask
    batch_size, channels, lr_height, lr_width = lr_tensor.size()
    pos_matrix, mask_matrix = weight_prediction_matrix_from_lr(lr_height, lr_width, args.upscale_factor)
    pos_matrix = pos_matrix.to(device, non_blocking=True)
    mask_matrix = mask_matrix.to(device, non_blocking=True)

    # Get GT image size
    gt_height = int(lr_height * args.upscale_factor)
    gt_width = int(lr_width * args.upscale_factor)

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = sr_model(lr_tensor, pos_matrix, args.upscale_factor)
        sr_tensor = torch.masked_select(sr_tensor, mask_matrix)
        sr_tensor = sr_tensor.contiguous().view(batch_size, channels, gt_height, gt_width)
    # Save image
    sr_image = tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_path, sr_image)

    print(f"SR image save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="metasr_rdn")
    parser.add_argument("--upscale_factor",
                        type=float,
                        default="4.0")
    parser.add_argument("--inputs_path",
                        type=str,
                        default="./figure/comic_lr.png",
                        help="Low-resolution image path.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./figure/comic_sr.png",
                        help="Super-resolution image path.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/pretrained_models/MetaSR_RDN-DIV2K-8daac205.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args)
