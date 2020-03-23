""" Interface with the polyrnn model """
from typing import List
import os
import numpy as np
import skimage.transform as transform
import tensorflow as tf
from EvalNet import EvalNet  # type: ignore
from PolygonModel import PolygonModel  # type: ignore
from scalabel.polyrnn_base import CropData, PolyrnnBase


class PolyrnnInterface(PolyrnnBase):
    """Class to interface with the PolygonRNN model"""

    def __init__(self) -> None:
        # External PATHS
        model_dir = os.path.join('scalabel', 'polyrnn_pp', 'models')
        polyrnn_metagraph = os.path.join(
            model_dir, 'poly', 'polygonplusplus.ckpt.meta')
        polyrnn_checkpoint = os.path.join(
            model_dir, 'poly', 'polygonplusplus.ckpt')
        evalnet_checkpoint = os.path.join(model_dir, 'evalnet', 'evalnet.ckpt')

        # ggnn_metagraph = 'polyrnn_pp/models/ggnn/ggnn.ckpt.meta'
        # ggnn_checkpoint = 'polyrnn_pp/models/ggnn/ggnn.ckpt'

        # Const
        batch_size = 1
        self.first_top_k = 1
        self.img_side = 224
        self.context_expansion = 0.5

        # graphs
        eval_graph = tf.Graph()
        poly_graph = tf.Graph()

        # Initializing and restoring the evaluator net.
        with eval_graph.as_default():
            with tf.variable_scope("discriminator_network"):
                evaluator = EvalNet(batch_size)
                evaluator.build_graph()
            saver = tf.train.Saver()

            # Start session
            eval_sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True
            ), graph=eval_graph)
            saver.restore(eval_sess, evalnet_checkpoint)

        # Initializing and restoring PolyRNN++
        self.model = PolygonModel(polyrnn_metagraph, poly_graph)
        self.model.register_eval_fn(
            lambda input_: evaluator.do_test(eval_sess, input_))
        self.poly_sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True
        ), graph=poly_graph)
        self.model.saver.restore(self.poly_sess, polyrnn_checkpoint)

    def predict_from_rect(self, crop_img: np.ndarray) -> List[np.ndarray]:
        """ predict rect -> poly """
        preds = [self.model.do_test(self.poly_sess, np.expand_dims(
            crop_img, axis=0), top_k) for top_k in range(self.first_top_k)]
        preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)
        preds = np.array(preds[0]['polys'][0])
        return preds

    def bbox_to_crop(self, img: np.ndarray, bbox: List[float]) -> CropData:
        """
        Takes in image and bounding box
        Crops to an expanded square around the bounding box
        Source: https://github.com/fidler-lab/polyrnn-pp-pytorch
        """
        x0, y0, w, h = bbox

        x_center = x0 + (1+w)/2.
        y_center = y0 + (1+h)/2.

        widescreen = w > h

        if not widescreen:
            img = img.transpose((1, 0, 2))
            x_center, y_center, w, h = y_center, x_center, h, w

        x_min = int(np.floor(x_center - w*(1 + self.context_expansion)/2.))
        x_max = int(np.ceil(x_center + w*(1 + self.context_expansion)/2.))

        x_min = max(0, x_min)
        x_max = min(img.shape[1] - 1, x_max)

        patch_w = x_max - x_min
        # NOTE: Different from before

        y_min = int(np.floor(y_center - patch_w / 2.))
        y_max = y_min + patch_w

        top_margin = max(0, y_min) - y_min

        y_min = max(0, y_min)
        y_max = min(img.shape[0] - 1, y_max)

        scale_factor = float(self.img_side)/patch_w

        patch_img = img[y_min:y_max, x_min:x_max, :]

        new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
        new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img

        new_img = transform.rescale(new_img, scale_factor, order=1,
                                    preserve_range=True, multichannel=True)
        new_img = new_img.astype(np.float32)

        starting_point = [x_min, y_min-top_margin]

        if not widescreen:
            # Now that everything is in a square
            # bring things back to original mode
            new_img = new_img.transpose((1, 0, 2))
            starting_point = [y_min-top_margin, x_min]

        return_dict = CropData(img=new_img,
                               scale_factor=scale_factor,
                               start=np.array(starting_point))
        return return_dict

    def rescale_output(self, preds: np.ndarray,
                       crop_dict: CropData) -> List[List[float]]:
        """ undo the cropping transform to get original output coords """
        # translate back to image space
        preds = crop_dict.start + float(self.img_side) / \
            crop_dict.scale_factor * preds

        output: List[List[float]] = preds.tolist()
        return output
