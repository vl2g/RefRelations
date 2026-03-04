# Few-Shot Referring Relationships in Videos
**CVPR 2023** | [Paper](https://vl2g.github.io/projects/refRelations/) 

## Overview
Given a query visual relationship `<subject, predicate, object>` and a test video, this framework spatiotemporally localizes the subject and object using only a few support videos sharing the same predicate (which is **unseen during training**).

## Directory Structure
```
few_shot_refrel/
├── configs/
│   └── default.yaml            # All hyperparameters
├── datasets/
│   ├── base_dataset.py         # Abstract dataset class
│   ├── vidvrd_dataset.py        # ImageNet-VidVRD dataset loader
│   └── vidor_dataset.py        # VidOR dataset loader
├── models/
│   ├── feature_extractor.py    # FasterRCNN + I3D feature extraction
│   ├── relationship_embedding.py # Query-conditioned relationship embedding
│   ├── aggregation.py          # GSA and LLA modules
│   ├── relation_network.py     # Metric-based meta-learner
│   └── random_field.py         # T-partite random field + belief propagation
├── utils/
│   ├── metrics.py              # Asub, Aobj, Ar, mIoU computations
│   ├── episode_sampler.py      # Episodic training sampler
│   └── visualization.py        # Trajectory visualization
├── scripts/
│   ├── extract_features.py     # Pre-extract FasterRCNN/I3D features
│   ├── train.py                # Training script
│   ├── test.py                 # Evaluation script
├── train.py                    # Main training entry point
├── test.py                     # Main evaluation entry point
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Data Preparation
Download [ImageNet-VidVRD](https://xdshang.github.io/docs/imagenet-vidvrd.html) or [VidOR](https://xdshang.github.io/docs/vidor.html), then:
```bash
python scripts/extract_features.py --dataset vidvrd --data_root /path/to/data
```

## Training
```bash
python train.py --config configs/default.yaml --dataset vidvrd
```

## Evaluation
```bash
python test.py --config configs/default.yaml --dataset vidvrd --checkpoint checkpoints/best.pth
```


## Citation
```bibtex
@inproceedings{kumar2023fewshot,
  title={Few-Shot Referring Relationships in Videos},
  author={Kumar, Yogesh and Mishra, Anand},
  booktitle={CVPR},
  year={2023}
}
```
