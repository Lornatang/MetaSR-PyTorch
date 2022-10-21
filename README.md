# MetaSR-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Meta-SR: A Magnification-Arbitrary Network for Super-Resolution](https://arxiv.org/abs/1903.00875).

## Table of contents

- [MetaSR-PyTorch](#metasr-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
    - [Test](#test)
    - [Train MetaSR_RDN model](#train-metasr_rdn-model)
    - [Resume train MetaSR_RDN model](#resume-train-metasr_rdn-model)
    - [Result](#result)
    - [Credit](#credit)
        - [Meta-SR: A Magnification-Arbitrary Network for Super-Resolution](#meta-sr-a-magnification-arbitrary-network-for-super-resolution)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

## Test

Modify the `config.py` file.

- line 31: `arch_name` change to `metasr_rdn`.
- line 39: `upscale_factor` change to `4`.
- line 47: `mode` change to `test`.
- line 49: `exp_name` change to `test_MetaSR_RDN-DIV2K`.
- line 90: `model_weights_path` change to `./results/pretrained_models/MetaSR_RDN-DIV2K-8daac205.pth.tar`.

```bash
python3 test.py
```

## Train MetaSR_RDN model

Modify the `config.py` file.

- line 31: `arch_name` change to `metasr_rdn`.
- line 39: `upscale_factor` change to `4`.
- line 47: `mode` change to `train`.
- line 49: `exp_name` change to `MetaSR_RDN-DIV2K`.

```bash
python3 train.py
```

## Resume train MetaSR_RDN model

Modify the `config.py` file.

- line 31: `arch_name` change to `metasr_rdn`.
- line 39: `upscale_factor` change to `4`.
- line 47: `mode` change to `train`.
- line 49: `exp_name` change to `MetaSR_RDN-DIV2K`.
- line 67: `resume_model_weights_path` change to `./samples/MetaSR_RDN-DIV2K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: https://arxiv.org/pdf/1903.00875v4.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

|   Model    | Dataset | Scale |       PSNR       |
|:----------:|:-------:|:-----:|:----------------:|
| MetaSR_RDN |  Set14  |  2.0  | 32.35(**32.37**) |
| MetaSR_RDN |  Set14  |  2.5  | 30.45(**29.85**) |
| MetaSR_RDN |  Set14  |  3.0  | 29.30(**27.95**) |
| MetaSR_RDN |  Set14  |  3.5  | 28.32(**27.29**) |
| MetaSR_RDN |  Set14  |  4.0  | 27.75(**26.55**) |

```bash
# Download `MetaSR_RDN-DIV2K-8daac205.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input:

<span align="center"><img width="240" height="360" src="figure/comic_lr.png"/></span>

Output:

<span align="center"><img width="240" height="360" src="figure/comic_sr.png"/></span>

```text
Build `metasr_rdn` model successfully.
Load `metasr_rdn` model weights `./results/pretrained_models/MetaSR_RDN-DIV2K-8daac205.pth.tar` successfully.
SR image save to `./figure/comic_sr.png`
```

## Credit

### Meta-SR: A Magnification-Arbitrary Network for Super-Resolution

_Xuecai Hu, Haoyuan Mu, Xiangyu Zhang, Zilei Wang, Tieniu Tan, Jian Sun_ <br>

**Abstract** <br>
Recent research on super-resolution has achieved great success due to the development of deep convolutional neural
networks (DCNNs). However,
super-resolution of arbitrary scale factor has been ignored for a long time. Most previous researchers regard
super-resolution of different scale
factors as independent tasks. They train a specific model for each scale factor which is inefficient in computing, and
prior work only take the
super-resolution of several integer scale factors into consideration. In this work, we propose a novel method called
Meta-SR to firstly solve
super-resolution of arbitrary scale factor (including non-integer scale factors) with a single model. In our Meta-SR,
the Meta-Upscale Module is
proposed to replace the traditional upscale module. For arbitrary scale factor, the Meta-Upscale Module dynamically
predicts the weights of the
upscale filters by taking the scale factor as input and use these weights to generate the HR image of arbitrary size.
For any low-resolution image,
our Meta-SR can continuously zoom in it with arbitrary scale factor by only using a single model. We evaluated the
proposed method through extensive
experiments on widely used benchmark datasets on single image super-resolution. The experimental results show the
superiority of our Meta-Upscale.

[[Code(PyTorch)]](https://github.com/XuecaiHu/Meta-SR-Pytorch) [[Paper]](https://arxiv.org/pdf/1903.00875v4.pdf)

```
@article{hu2019meta,
  title={Meta-SR: A Magnification-Arbitrary Network for Super-Resolution},
  author={Hu, Xuecai and Mu, Haoyuan and Zhang, Xiangyu and Wang, Zilei  and Tan, Tieniu and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
