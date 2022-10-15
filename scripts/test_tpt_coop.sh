#!/bin/bash

data_root='/path/to/your/data/root'
coop_weight='/path/to/pretrained/coop/weight.pth'
testsets=$1
arch=RN50
# arch=ViT-B/16
bs=64

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --load ${coop_weight}