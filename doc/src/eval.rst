Evaluation
===================

We currently support evaluation of three tasks: Object detection, instance segmentation, multi-object
tracking, and multi-object tracking and segmentation.
To evaluate your algorithms on each task, input your predictions and the
corresponding ground truth annotations in Scalabel format.

Detection
-----------------
The detection evaluation uses the AP metric and follows the protocol defined
in the COCO dataset. You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.detect \
        --gt scalabel/eval/testcases/box_track/track_sample_anns.json \
        --result scalabel/eval/testcases/det/bbox_predictions.json \
        --config scalabel/eval/testcases/det/det_configs.toml


Available arguments:

.. code-block:: bash

    --gt GT_PATH, -g GT_PATH
                            path to ground truth annotations.
    --result RESULT_PATH, -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for evaluation results.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for evaluation.


Instance Segmentation
-----------------------
The instance segmentation evaluation also uses the AP metric and follows the protocol defined
in the COCO dataset. You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.ins_seg \
        --gt scalabel/eval/testcases/ins_seg/ins_seg_rle_sample.json \
        --result scalabel/eval/testcases/ins_seg/ins_seg_preds.json \
        --config scalabel/eval/testcases/ins_seg/ins_seg_configs.toml


Available arguments:

.. code-block:: bash

    --gt GT_PATH, -g GT_PATH
                            path to ground truth annotations.
    --result RESULT_PATH, -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for evaluation results.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for evaluation.


Semantic Segmentation
-----------------------
The semantic segmentation evaluation uses the standard Jaccard Index, commonly known as mean-IoU. You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.sem_seg \
        --gt scalabel/eval/testcases/sem_seg/sem_seg_sample.json \
        --result scalabel/eval/testcases/sem_seg/sem_seg_preds.json \
        --config scalabel/eval/testcases/sem_seg/sem_seg_configs.toml


Available arguments:

.. code-block:: bash

    --gt GT_PATH, -g GT_PATH
                            path to ground truth annotations.
    --result RESULT_PATH, -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for evaluation results.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for evaluation.


Pose Estimation
-----------------
The pose estimation evaluation also uses the AP metric and follows the protocol defined
in the COCO dataset. You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.pose \
        --gt scalabel/eval/testcases/pose/pose_sample.json \
        --result scalabel/eval/testcases/pose/pose_preds.json \
        --config scalabel/eval/testcases/pose/pose_configs.toml


Available arguments:

.. code-block:: bash

    --gt GT_PATH, -g GT_PATH
                            path to ground truth annotations.
    --result RESULT_PATH, -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for evaluation results.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for evaluation.


Multi-object Tracking
----------------------
The MOT evaluation uses the CLEAR MOT metrics. You can start the evaluation
by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.mot \
        --gt scalabel/eval/testcases/box_track/track_sample_anns.json \
        --result scalabel/eval/testcases/box_track/track_predictions.json \
        --config scalabel/eval/testcases/box_track/box_track_configs.toml

Available arguments:

.. code-block:: bash

    --gt GT_PATH, -g GT_PATH
                            path to ground truth annotations.
    --result RESULT_PATH, -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for evaluation results.
    --iou-thr IOU_TRESH
                            IoU threshold for mot evaluation.
    --ignore-iof-thr IGNORE_IOF_THRESH
                            Ignore iof threshold for mot evaluation.
    --ignore-unknown-cats IGNORE_UNKNOWN_CATS
                            Ignore unknown categories for mot evaluation.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for mot evaluation.


Multi-object Tracking and Segmentation
----------------------------------------
The MOTS evaluation also uses the CLEAR MOT metrics, but uses mask IoU instead of box IoU. You can start the evaluation
by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.mots \
        --gt scalabel/eval/testcases/seg_track/seg_track_sample.json \
        --result scalabel/eval/testcases/seg_track/seg_track_preds.json \
        --config scalabel/eval/testcases/seg_track/seg_track_configs.toml

Available arguments:

.. code-block:: bash

    --gt GT_PATH, -g GT_PATH
                            path to ground truth annotations.
    --result RESULT_PATH, -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for evaluation results.
    --iou-thr IOU_TRESH
                            IoU threshold for mots evaluation.
    --ignore-iof-thr IGNORE_IOF_THRESH
                            Ignore iof threshold for mots evaluation.
    --ignore-unknown-cats IGNORE_UNKNOWN_CATS
                            Ignore unknown categories for mots evaluation.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for mots evaluation.
