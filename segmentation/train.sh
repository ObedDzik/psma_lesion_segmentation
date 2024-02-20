# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

torchrun --standalone --nproc_per_node=1 trainddp.py --fold=0 --network-name='unet' --epochs=300 --input-patch-size=128 --train-bs=8 --num_workers=4 --lr=2e-4 --wd=1e-5 --val-interval=2 --sw-bs=8 --cache-rate=1