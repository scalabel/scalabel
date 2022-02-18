Evaluation
===================

We currently support evaluation of nine tasks: Image tagging, object detection,
instance segmentation, semantic segmentation, panoptic segmentation, pose
estimation, boundary detection, multi-object tracking, and multi-object tracking
and segmentation.
To evaluate your algorithms on each task, input your predictions and the
corresponding ground truth annotations in Scalabel format.

Image Tagging
-----------------
The tagging evaluation uses standard classification metrics.
You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.tagging \
        --gt scalabel/eval/testcases/tagging/tag_gts.json \
        --result scalabel/eval/testcases/tagging/tag_preds.json \
        --config scalabel/eval/testcases/tagging/tag_configs.toml


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


Panoptic Segmentation
-----------------------
The panoptic segmentation evaluation uses the Panoptic Quality (PQ) metric. You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.pan_seg \
        --gt scalabel/eval/testcases/pan_seg/pan_seg_sample.json \
        --result scalabel/eval/testcases/pan_seg/pan_seg_preds.json \
        --config scalabel/eval/testcases/pan_seg/pan_seg_configs.toml


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


Boundary Detection
--------------------
The boundary detection evaluation uses the F-measure for boundaries using
morphological operators.
You can start the evaluation by running, e.g.:

.. code-block:: bash

    python3 -m scalabel.eval.boundary \
        --gt scalabel/eval/testcases/boundary/boundary_gts.json \
        --result scalabel/eval/testcases/boundary/boundary_preds.json \
        --config scalabel/eval/testcases/boundary/boundary_configs.toml


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
