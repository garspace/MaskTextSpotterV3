# Mask TextSpotter v3
This is a PyTorch implemntation of the ECCV 2020 paper [Mask TextSpotter v3](https://arxiv.org/abs/2007.09482). Mask TextSpotter v3 is an end-to-end trainable scene text spotter that adopts a Segmentation Proposal Network (SPN) instead of an RPN. Mask TextSpotter v3 significantly improves robustness to rotations, aspect ratios, and shapes.

## Relationship to Mask TextSpotter
Here we label the Mask TextSpotter series as Mask TextSpotter v1 ([ECCV 2018 paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Pengyuan_Lyu_Mask_TextSpotter_An_ECCV_2018_paper.pdf), [code](https://github.com/lvpengyuan/masktextspotter.caffe2)), Mask TextSpotter v2 ([TPAMI paper](https://ieeexplore.ieee.org/document/8812908), [code](https://github.com/MhLiao/MaskTextSpotter)), and Mask TextSpotter v3 (ECCV 2020 paper).

This project is under a lincense of Creative Commons Attribution-NonCommercial 4.0 International. Part of the code is inherited from [Mask TextSpotter v2](https://github.com/MhLiao/MaskTextSpotter), which is under an MIT license.


## Installation

### Requirements:
- Python3 (Python3.7 is recommended)
- PyTorch >= 1.4 (1.4 is recommended)
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9 (This is very important!)
- OpenCV
- CUDA >= 9.0 (10.0.130 is recommended)


```bash
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name masktextspotter -y
  conda activate masktextspotter

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX pyclipper Polygon3 editdistance 

  # install PyTorch
  conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

  export INSTALL_DIR=$PWD

  # install pycocotools
  cd $INSTALL_DIR
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py build_ext install

  # install apex
  cd $INSTALL_DIR
  git clone https://github.com/NVIDIA/apex.git
  cd apex
  python setup.py install --cuda_ext --cpp_ext

  # clone repo
  cd $INSTALL_DIR
  git clone https://github.com/MhLiao/MaskTextSpotterV3.git
  cd MaskTextSpotterV3

  # build 可能遇到的问题
   将maskrcnn_benchmark/csrc/cuda/deform_conv_cuda.cu和maskrcnn_benchmark/csrc/cuda/deform_pool_cuda.cu中的
   AT_CHECK 替换为 TORCH_CHECK

  # build
  python setup.py build develop


  unset INSTALL_DIR
```

## Models
Download the trained model [Google Drive](https://drive.google.com/file/d/1XQsikiNY7ILgZvmvOeUf9oPDG4fTp0zs/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1fV1RbyQ531IifdKxkScItQ) (downloading code: cnj2).

Option: Download the model pretrain with SynthText for your quick re-implementation. [Google Drive](https://drive.google.com/file/d/1vrG-EqiQWRpygh3uQB25NOiJu_jaRy4u/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1yR97s9EArTE2asv5rWOf4Q) (downloading code: c82l).


## Demo 
You can run a demo script for a single image inference by ```python tools/demo.py```.

## Datasets
The datasets are the same as Mask TextSpotter v2.

Download the ICDAR2013([Google Drive](https://drive.google.com/open?id=1sptDnAomQHFVZbjvnWt2uBvyeJ-gEl-A), [BaiduYun](https://pan.baidu.com/s/18W2aFe_qOH8YQUDg4OMZdw)) and ICDAR2015([Google Drive](https://drive.google.com/open?id=1HZ4Pbx6TM9cXO3gDyV04A4Gn9fTf2b5X), [BaiduYun](https://pan.baidu.com/s/16GzPPzC5kXpdgOB_76A3cA)) as examples.

The SCUT dataset used for training can be downloaded [here](https://drive.google.com/file/d/1BpE2GEFF7Ay7jPqgaeHxMmlXvM-1Es5_/view?usp=sharing).

The converted labels of Total-Text dataset can be downloaded [here](https://1drv.ms/u/s!ArsnjfK83FbXgcpti8Zq9jSzhoQrqw?e=99fukk).

The converted labels of SynthText can be downloaded [here](https://1drv.ms/u/s!ArsnjfK83FbXgb5vgOOVPYywgCWuQw?e=UPuNTa).

The root of the dataset directory should be ```MaskTextSpotterV3/datasets/```.

## Testing
### Prepar dataset
An example of the path of test images: ```MaskTextSpotterV3/datasets/icdar2015/test_iamges```

### Check the config file (configs/finetune.yaml) for some parameters.
test dataset: ```TEST.DATASETS```; 

input size: ```INPUT.MIN_SIZE_TEST''';

model path: ```MODEL.WEIGHT```;

output directory: ```OUTPUT_DIR```

### run ```sh test.sh```


## Training
Place all the training sets in ```MaskTextSpotterV3/datasets/``` and check ```DATASETS.TRAIN``` in the config file.
### Pretrain
Trained with SynthText

```python3 -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/pretrain/seg_rec_poly_fuse_feature.yaml ```
### Finetune
Trained with a mixure of SynthText, icdar2013, icdar2015, scut-eng-char, and total-text

check the initial weights in the config file.

```python3 -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/mixtrain/seg_rec_poly_fuse_feature.yaml ```

## 训练自己的中文数据集
在原作者代码的基础上为了训练自己的中文数据集做的改动

①在datasets目录下创建chinese数据集目录，类型格式参照ic13/15

train_gts
train_images
test_images
​ gt格式: 13, 338, 258, 320, 264, 408, 19, 426,耻辱不亚于,19, 388, 68, 387, 68, 422, 19, 425,耻,70, 336, 121, 335, 121, 379, 70, 381,辱,123, 330, 167, 328, 168, 363, 123, 364,不,170, 338, 215, 337, 216, 369, 170, 370,亚,222, 371, 262, 369, 262, 409, 222, 410,于

②修改configs目录下的yaml文件

ROI_MASK_HEAD下的PREDICTOR修改为“SeqMaskRCNNC4Predictor”
ROI_MASK_HEAD下的CHAR_NUM_CLASSES修改为加入汉字后的字典元素总个数
DATASETS中加入chinese_train
SOLVER中的BASE_LR设为0.002 
SEQUENCE中的NUM_CHAR修改为加入汉字后的字典元素总个数  
③修改maskrcnn_benchmark/config/paths_catalog.py 

在class DatasetCatalog 中的DATASETS中仿照ic13加入chinese dataset的目录设置

## 只训练检测模型，不需要识别部分，修改步骤如下
1.修改predictor
ROI_MASK_HEAD.PREDICTOR=MaskRCNN4Predictor
使用MaskRcnn作为预测器

2.修改POOLER_RESOLUTION
ROI_MASK_HEAD.POOLER_RESOLUTION_W=14

ROI_MASK_HEAD.POOLER_RESOLUTION_H=14
保证mask和输入的尺寸一致

3.CHAR_MASK_ON=False
该设置用来分割字符

4.SEQ_ON=False
该设置用来进行字符识别

仅训练检测模型：
python -m torch.distributed.launch  tools/train_net.py --config configs/mixtrain/only_detect_config.yaml

## Evaluation
### Download lexicons
[Google Drive](https://drive.google.com/file/d/15PAG-ok8KtJjNxP-pOp7kX_esjCpfzn5/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1kXGaF9jev1ysQhTOBbIDDg) (
downloading code: f3tk)

unzip and palce it like ```evaluation/lexicons/```.
### Evaluation for Total-Text dataset

```
cd evaluation/totaltext/e2e/
# edit "result_dir" in script.py
python script.py
```

### Evaluation for the Rotated ICDAR 2013 dataset
First, generate the Rotated ICDAR 2013 dataset
```
cd tools
# set the specific rotating angle in convert_dataset.py
python convert_dataset.py
```
Then, run testing (change test set in YAML) and evaluate by ```evaluation/rotated_icdar2013/e2e/script.py```

## Citing the related works

Please cite the related works in your publications if it helps your research:

    @inproceedings{liao2020mask,
      title={Mask TextSpotter v3: Segmentation Proposal Network for Robust Scene Text Spotting},
      author={Liao, Minghui and Pang, Guan and Huang, Jing and Hassner, Tal and Bai, Xiang},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      year={2020}
    }

    @article{liao2019mask,
      author={M. {Liao} and P. {Lyu} and M. {He} and C. {Yao} and W. {Wu} and X. {Bai}},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      title={Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes},
      volume={43},
      number={2},
      pages={532--548},
      year={2021}
    }
    
    @inproceedings{lyu2018mask,
      title={Mask textspotter: An end-to-end trainable neural network for spotting text with arbitrary shapes},
      author={Lyu, Pengyuan and Liao, Minghui and Yao, Cong and Wu, Wenhao and Bai, Xiang},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      pages={67--83},
      year={2018}
    }
    
