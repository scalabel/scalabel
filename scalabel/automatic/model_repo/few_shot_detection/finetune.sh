#!/bin/bash

shots=("1" "2" "3" "5" "10")

for i in "${!shots[@]}"; do
    # finetuning and evaluation
    # python3 -m tools.train_net --num-gpus 2 \
    #                            --config-file configs/BDD100K-detection/${shots[$i]}shot.yaml \
    #                            --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth

    if [ $i -eq 0 ]; then
        python3 -m tools.train_net --num-gpus 2 \
                                   --config-file configs/BDD100K-detection/${shots[$i]}shot.yaml \
                                   --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth
    else
        python3 -m tools.train_net --num-gpus 2 \
                                   --config-file configs/BDD100K-detection/${shots[$i]}shot.yaml \
                                   --opts MODEL.WEIGHTS checkpoints/bdd100k/bdd100k-${shots[${i}-1]}shot/model_reset_surgery.pth
    fi

    # inference
    python3 -m demo.demo --config-file checkpoints/bdd100k/bdd100k-${shots[$i]}shot/config.yaml \
                         --input datasets/scalabel/test/* \
                         --output datasets/scalabel/res/${shots[$i]}shot \
                         --opts MODEL.WEIGHTS checkpoints/bdd100k/bdd100k-${shots[$i]}shot/model_final.pth

    # randomly initialize the weights corresponding to the novel classes
    python3 -m tools.ckpt_surgery --src1 checkpoints/bdd100k/bdd100k-${shots[$i]}shot/model_final.pth \
                                  --method randinit \
                                  --save-dir checkpoints/bdd100k/bdd100k-${shots[$i]}shot
done
