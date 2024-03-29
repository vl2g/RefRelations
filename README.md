**Implementation of the Few-shot Referring Relatiohsip in Videos (CVPR 2023) paper**


[project page](https://vl2g.github.io/projects/refRelations/) | [paper](https://vl2g.github.io/projects/refRelations/docs/paper.pdf)

## Requirements
* Use **python >= 3.8.5**. Conda recommended : [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)

* Use **pytorch 1.7.0 CUDA 10.2 or higher**

* Other requirements from 'requirements.txt'

**To setup environment**
```
  # create new env fsrr
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

- Split Videos into Frames
``` 
$ python video_to_frame.py
```
- Extract faster_rcnn features: 
``` 
  $ sh data_preparation/vidor.sh
  # Please follow instructions [here](data_preparation/README.md).
```
- Extract I3d features:
```
  $ sh data_preparation/vidor_i3d.sh
```

### Traning RelationNet and VR_Encoder
```
  $ python model/relnet.py
  # Follow model/config.py for different model settings
```
### Inference
```
  $ python inference/FullModel_inf.py
  # Follow inference/config.py for inference settings
```

### Evaluation
```
  $ sh eval/eval.sh
```

## Cite
If you find this work useful for your research, please consider citing.
<pre><tt>@inproceedings{
fewshot_ref_rel,
title={Few-Shot Referring Relationships in Videos},
author={Yogesh Kumar, Anand Mishra},
booktitle={Conference on Computer Vision and Pattern Recognition 2023},
year={2023},
url={https://openreview.net/forum?id=dCbmHXhGtib}
}
}</tt></pre>

