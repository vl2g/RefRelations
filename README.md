**Implementation of the Few-shot Referring Relatiohsip in Videos (CVPR 2022) paper**

**Complete Code will be updated soon**

[project page](https://vl2g.github.io/projects/refRelations/) | [paper]()

## Requirements
* Use **python >= 3.8.5**. Conda recommended : [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)

* Use **pytorch 1.7.0 CUDA 10.2 or higher**

* Other requirements from 'requirements.txt'

**To setup environment**
```
# create new env vrc
$ conda create -n fsrr python=3.8.5

# activate fsrr
$ conda activate fsrr

# install pytorch, torchvision
$ conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch

# install other dependencies
$ pip install -r requirements.txt
```

## Training

### Preparing dataset
- Download ViOR and ImageNet VidVRD dataset from [https://xdshang.github.io/docs/imagenet-vidvrd.html and https://xdshang.github.io/docs/vidor.html)


- Extract faster_rcnn features of video frames images using [data_preparation/fsrr_extract_frcnn_feats.py](data_preparation/fsrr_extract_frcnn_feats.py). Please follow instructions [here](data_preparation/README.md).
