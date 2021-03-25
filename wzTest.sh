#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python wzTest.py --max_disp 192 \
                  --isDebug 0 \
                  --pretrained_net ./checkpoint/net_best.pth
