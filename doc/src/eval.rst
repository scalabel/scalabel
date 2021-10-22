Evaluation
===================

We currently support evaluation of three tasks: Object detection, instance segmentation, and multi-object
tracking.
To evaluate your algorithms on each task, input your predictions and the
corresponding ground truth annotations in Scalabel format.

Detection
-----------------
The detection evaluation uses the AP metric and follows the protocol defined
in the COCO dataset. You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.detect \
        --gt scalabel/eval/testcases/track_sample_anns.json \
        --result scalabel/eval/testcases/bbox_predictions.json \
        --config scalabel/eval/testcases/det_configs.toml


Available arguments:

.. code-block:: bash

    --gt GT_PATH, -g GT_PATH
                            path to ground truth annotations.
    --result RESULT_PATH, -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for detection evaluation results.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for detection evaluation.


Instance Segmentation
-----------------------
The instance segmentation evaluation also uses the AP metric and follows the protocol defined
in the COCO dataset. You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.ins_seg \
        --gt scalabel/eval/testcases/ins_seg_rle_sample.json \
        --result scalabel/eval/testcases/ins_seg_preds.json \
        --config scalabel/eval/testcases/ins_seg_configs.toml


Available arguments:

.. code-block:: bash

    --gt GT_PATH, -g GT_PATH
                            path to ground truth annotations.
    --result RESULT_PATH, -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for instance segmentation evaluation results.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for detection evaluation.


Pose Estimation
-----------------
The pose estimation evaluation also uses the AP metric and follows the protocol defined
in the COCO dataset. You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.pose \
        --gt scalabel/eval/testcases/pose_sample.json \
        --result scalabel/eval/testcases/pose_preds.json \
        --config scalabel/eval/testcases/pose_configs.toml


Available arguments:

.. code-block:: bash

    --gt GT_PATH, -g GT_PATH
                            path to ground truth annotations.
    --result RESULT_PATH, -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for pose estimation evaluation results.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for detection evaluation.


Multi-object Tracking
----------------------
The MOT evaluation uses the CLEAR MOT metrics. You can start the evaluation
by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.mot \
        --gt scalabel/eval/testcases/track_sample_anns.json \
        --result scalabel/eval/testcases/track_predictions.json \
        --config scalabel/eval/testcases/box_track_configs.toml

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
