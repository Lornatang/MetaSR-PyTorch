# Meta_RDN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Meta-SR: A Magnification-Arbitrary Network for Super-Resolution](https://arxiv.org/abs/1903.00875).

### Table of contents

- [Meta_RDN-PyTorch](#meta_rdn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Meta-SR: A Magnification-Arbitrary Network for Super-Resolution](#about-meta-sr-a-magnification-arbitrary-network-for-super-resolution)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Meta-SR: A Magnification-Arbitrary Network for Super-Resolution](#enhanced-deep-residual-networks-for-single-image-super-resolution)

## About Meta-SR: A Magnification-Arbitrary Network for Super-Resolution

If you're new to Meta-RDN, here's an abstract straight from the paper:

Recent research on super-resolution has achieved great success due to the development of deep convolutional neural networks (DCNNs). However,
super-resolution of arbitrary scale factor has been ignored for a long time. Most previous researchers regard super-resolution of different scale
factors as independent tasks. They train a specific model for each scale factor which is inefficient in computing, and prior work only take the
super-resolution of several integer scale factors into consideration. In this work, we propose a novel method called Meta-SR to firstly solve
super-resolution of arbitrary scale factor (including non-integer scale factors) with a single model. In our Meta-SR, the Meta-Upscale Module is
proposed to replace the traditional upscale module. For arbitrary scale factor, the Meta-Upscale Module dynamically predicts the weights of the
upscale filters by taking the scale factor as input and use these weights to generate the HR image of arbitrary size. For any low-resolution image,
our Meta-SR can continuously zoom in it with arbitrary scale factor by only using a single model. We evaluated the proposed method through extensive
experiments on widely used benchmark datasets on single image super-resolution. The experimental results show the superiority of our Meta-Upscale.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the `config.py` file as follows.

- line 35: `upscale_factor` change to the magnification you need to enlarge.
- line 37: `mode` change Set to valid mode.
- line 73: `model_path` change weight address after training.

## Train

Modify the contents of the `config.py`file as follows.

- line 35: `upscale_factor` change to the magnification you need to enlarge.
- line 37: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the `config.py` file as follows.

- line 51: `start_epoch` change number of training iterations in the previous round.
- line 52: `resume` the weight address that needs to be loaded.

## Result

Source of original paper results: https://arxiv.org/pdf/1903.00875v4.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |     PSNR     |
|:-------:|:-----:|:------------:|
|  Set14  |  2.0  | 32.35(**-**) |
|  Set14  |  2.5  | 30.45(**-**) |
|  Set14  |  3.0  | 29.30(**-**) |
|  Set14  |  3.5  | 28.32(**-**) |
|  Set14  |  4.0  | 27.75(**-**) |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="assets/result.png"/></span>

### Credit

#### Meta-SR: A Magnification-Arbitrary Network for Super-Resolution

_Xuecai Hu, Haoyuan Mu, Xiangyu Zhang, Zilei Wang, Tieniu Tan, Jian Sun_ <br>

**Abstract** <br>
Recent research on super-resolution has achieved great success due to the development of deep convolutional neural networks (DCNNs). However,
super-resolution of arbitrary scale factor has been ignored for a long time. Most previous researchers regard super-resolution of different scale
factors as independent tasks. They train a specific model for each scale factor which is inefficient in computing, and prior work only take the
super-resolution of several integer scale factors into consideration. In this work, we propose a novel method called Meta-SR to firstly solve
super-resolution of arbitrary scale factor (including non-integer scale factors) with a single model. In our Meta-SR, the Meta-Upscale Module is
proposed to replace the traditional upscale module. For arbitrary scale factor, the Meta-Upscale Module dynamically predicts the weights of the
upscale filters by taking the scale factor as input and use these weights to generate the HR image of arbitrary size. For any low-resolution image,
our Meta-SR can continuously zoom in it with arbitrary scale factor by only using a single model. We evaluated the proposed method through extensive
experiments on widely used benchmark datasets on single image super-resolution. The experimental results show the superiority of our Meta-Upscale.

[[Code(PyTorch)]](https://github.com/XuecaiHu/Meta-SR-Pytorch) [[Paper]](https://arxiv.org/pdf/1903.00875v4.pdf)

```
@article{hu2019meta,
  title={Meta-SR: A Magnification-Arbitrary Network for Super-Resolution},
  author={Hu, Xuecai and Mu, Haoyuan and Zhang, Xiangyu and Wang, Zilei  and Tan, Tieniu and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
