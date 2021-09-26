# SARNet_RS21

Code and result repository for our paper "Semantic-Guided Attention Refinement Network for Salient Object Detection in Optical Remote Sensing Images."

## 0. Prerequisites

Note that SARNet is only tested on Win_OS with the following environments. It may work on other operating systems as well, but we do not guarantee that it will.

○ Creating a virtual environment in terminal: `conda create -n SARNet python=3.6`.

○ Installing necessary packages: `pip install -r requirements.txt`.

## 1. Download training and test sets

Download the ORSSD and EORSSD datasets at the following link address:

○ [EORSSD dataset](https://github.com/rmcong/EORSSD-dataset)

○ [ORSSD dataset](https://pan.baidu.com/s/1k44UlTLCW17AS0VhPyP7JA)

## 2. Results Download

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single GeForce RTX 3090 GPU of 24 GB Memory.

Our SARNet test results on ORSSD and EORSSD datasets:

○ [BaiduYun](https://pan.baidu.com/s/15FafJpTlYzJT6h6x06Sb5A) (code: akyu)

○ [Google Drive](https://drive.google.com/file/d/1yFYXJLBraP4o1MPRMISDUp7VTf3N1bom/view?usp=sharing)

## 3. Evaluation

You can evaluate the result maps using the tool in [Matlab Version](http://dpfan.net/d3netbenchmark/) or [Python_GPU Version](https://github.com/zyjwuyan/SOD_Evaluation_Metrics).

## 4. Citation

Please cite our paper if you find the work useful:

```
@article{huang2021semantic,
  title={Semantic-Guided Attention Refinement Network for Salient Object Detection in Optical Remote Sensing Images},
  author={Huang, Zhou and Chen, Huaixin and Liu, Biyuan and Wang, Zhixi},
  journal={Remote Sensing},
  volume={13},
  number={11},
  pages={2163},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

## 5. Acknowledgement

Thanks to [Deng-Ping Fan](http://dpfan.net/), [Yu-Wei Jin](https://sciprofiles.com/profile/author/UHpJQzZuSWRUWlFxK1ZLaTdtY3U1bllJSVFYNk5YOGNtSjU4OExiL28vYz0=), and [Run-Min Cong](https://rmcong.github.io/) for their help in our work.

