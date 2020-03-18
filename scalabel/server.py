from os import environ as env
from polyrnn_pp.cityscapes import DataProvider
from polyrnn_pp.EvalNet import EvalNet
from polyrnn_pp.PolygonModel import PolygonModel
from PIL import Image
from flask import Flask, request, jsonify, make_response
import tensorflow as tf
import requests
import numpy as np
import json
import logging
import time
import io
import argparse

# External PATHS
PolyRNN_metagraph = 'polyrnn_pp/models/poly/polygonplusplus.ckpt.meta'
PolyRNN_checkpoint = 'polyrnn_pp/models/poly/polygonplusplus.ckpt'
EvalNet_checkpoint = 'polyrnn_pp/models/evalnet/evalnet.ckpt'
GGNN_metagraph = 'polyrnn_pp/models/ggnn/ggnn.ckpt.meta'
GGNN_checkpoint = 'polyrnn_pp/models/ggnn/ggnn.ckpt'

# Const
_BATCH_SIZE = 1
_FIRST_TOP_K = 1

# logging
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# graphs
evalGraph = tf.Graph()
polyGraph = tf.Graph()

# Initializing and restoring the evaluator net.
with evalGraph.as_default():
    with tf.variable_scope("discriminator_network"):
        evaluator = EvalNet(_BATCH_SIZE)
        evaluator.build_graph()
    saver = tf.train.Saver()

    # Start session
    evalSess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True
    ), graph=evalGraph)
    saver.restore(evalSess, EvalNet_checkpoint)

# Initializing and restoring PolyRNN++
model = PolygonModel(PolyRNN_metagraph, polyGraph)
model.register_eval_fn(lambda input_: evaluator.do_test(evalSess, input_))
polySess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True
), graph=polyGraph)
model.saver.restore(polySess, PolyRNN_checkpoint)


def homepage():
    """ hello world test """
    return 'Test server for PolygonRNN++\n'


def polyrnn_base():
    """ predict rect -> polygon """
    logger.info('Hitting prediction endpoint')
    start_time = time.time()
    data = request.get_json()
    url = data['url']
    labels = data['labels']
    label = labels[0]
    box = label['box2d']
    x1 = box['x1']
    x2 = box['x2']
    y1 = box['y1']
    y2 = box['y2']

    try:
        img_response = requests.get(url)
        img = Image.open(io.BytesIO(img_response.content))
    except requests.exceptions.ConnectionError:
        return 'Bad image url provided.'

    # crop using bbox, taken from pytorch version repo
    opts = {
        'img_side': 224
    }
    provider = DataProvider(opts, mode='tool')
    instance = {
        'bbox': [x1, y1, x2 - x1, y2-y1],
        'img': np.array(img)
    }
    context_expansion = 0.5
    crop_dict = provider.extract_crop(
        {}, instance, context_expansion)
    crop_img = crop_dict['img']

    preds = [model.do_test(polySess, np.expand_dims(
        crop_img, axis=0), top_k) for top_k in range(_FIRST_TOP_K)]
    preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)
    preds = np.array(preds[0]['polys'][0])
    start = np.array(crop_dict['starting_point'])

    # translate back to image space
    preds = [start + p * 224.0/crop_dict['scale_factor'] for p in preds]
    preds = np.array(preds).tolist()

    logger.info('Time for prediction: {}'.format(
        time.time() - start_time))
    return make_response(jsonify({'points': preds}))


def polyrnn_refine():
    """ predict poly -> poly """
    return 'This method is not yet implemented\n'


def create_app() -> Flask:
    """ set up the flask app """
    app = Flask(__name__)

    # url rules should match NodeJS endpoint names
    app.add_url_rule('/', view_func=homepage)
    app.add_url_rule('/polygonRNNBase',
                     view_func=polyrnn_base, methods=['POST'])
    app.add_url_rule('/polygonRNNRefine',
                     view_func=polyrnn_refine, methods=['POST'])

    return app


def launch() -> None:
    """ main process launcher """
    logger.info('Launching model server')
    parser = argparse.ArgumentParser(
        description='Launch the server on one machine.')
    parser.add_argument(
        '--host', dest='host',
        help='server hostname', default='0.0.0.0')
    parser.add_argument(
        '--port', dest='port',
        help='server port', default=8080)
    parser.add_argument(
        '--debugMode', dest='debugMode',
        help='server debug mode', default=1)
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, debug=args.debugMode,
            port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    launch()
