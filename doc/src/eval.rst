Evaluation
===================

We currently support evaluation of two tasks: Object detection and multi-object
tracking.
To evaluate your algorithms on each task, input your predictions and the
corresponding ground truth annotations in Scalabel format.

Detection
-----------------
The detection evaluation uses the AP metric and follows the protocol defined
in the COCO dataset. You can start the evaluation by running:

``python3 -m scalabel.eval.detect <args>``

Available arguments:

.. code-block:: bash

    --gt, GT_PATH -g GT_PATH
                            path to ground truth annotations.
    --result, RESULT_PATH -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for detection evaluation results.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for detection evaluation.


Multi-object Tracking
----------------------
The MOT evaluation uses the CLEAR MOT metrics. You can start the evaluation
by running:

``python3 -m scalabel.eval.mot <args>``

Available arguments:

.. code-block:: bash

    --gt, GT_PATH -g GT_PATH
                            path to ground truth annotations.
    --result, RESULT_PATH -r RESULT_PATH
                            path to results to be evaluated.
    --config CFG_PATH, -c CFG_PATH
                            Config path. Contains metadata like available categories.
    --out-dir OUT_DIR, -o OUT_DIR
                            Output path for evaluation results.
    --iou-thr IOU_TRESH
                            IoU threshold for mot evaluation.
    --ignore-iof-thr IGNORE_IOF_THRESH
                            Ignore iof threshold for mot evaluation.
    --nproc NUM_PROCS, -p NUM_PROCS
                            Number of processes for mot evaluation.
