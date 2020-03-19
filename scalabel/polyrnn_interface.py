import numpy as np
import tensorflow as tf
from polyrnn_pp.EvalNet import EvalNet
from polyrnn_pp.PolygonModel import PolygonModel


class PolyrnnInterface():
    """Class to interface with the PolygonRNN model"""

    def __init__(self):
        # External PATHS
        polyrnn_metagraph = 'polyrnn_pp/models/poly/polygonplusplus.ckpt.meta'
        polyrnn_checkpoint = 'polyrnn_pp/models/poly/polygonplusplus.ckpt'
        evalnet_checkpoint = 'polyrnn_pp/models/evalnet/evalnet.ckpt'
        ggnn_metagraph = 'polyrnn_pp/models/ggnn/ggnn.ckpt.meta'
        ggnn_checkpoint = 'polyrnn_pp/models/ggnn/ggnn.ckpt'

        # Const
        batch_size = 1
        self.first_top_k = 1

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

    def predict_from_rect(self, crop_img):
        """ predict rect -> poly """
        preds = [self.model.do_test(self.poly_sess, np.expand_dims(
            crop_img, axis=0), top_k) for top_k in range(self.first_top_k)]
        preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)
        preds = np.array(preds[0]['polys'][0])
        return preds
