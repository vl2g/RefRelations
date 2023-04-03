#!/usr/bin/env bash
python proposal_bbox_extract.py
python gt_bbox_extract.py
python eval.py