import io
import time
import logging
import json
import numpy as np
import config
import tensorflow as tf
from flask import Flask, request, jsonify, make_response
from PIL import Image
from polyrnn_pp.PolygonModel import PolygonModel
from polyrnn_pp.EvalNet import EvalNet

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

app = Flask(__name__)


@app.route('/')
def homepage():
    return 'Test server for PolygonRNN++\n'


@app.route('/polygonRNNBase', methods=["POST", "GET"])
def predictPolygonRNNBase():
    if request.method == 'POST':
        # we will get the file from the request
        logger.info('Hitting prediction endpoint')
        start_time = time.time()
        data = request.get_json()
        url = data['url']
        return url

        if not image_bytes:
            return

        parsed_image = Image.open(io.BytesIO(image_bytes))
        preds = [model.do_test(polySess, np.expand_dims(
            parsed_image, axis=0), top_k) for top_k in range(_FIRST_TOP_K)]
        preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)

        logger.info('Finish prediction time: {}'.format(
            time.time() - start_time))
        return make_response(jsonify({'pred': np.array(preds[0]['polys'][0]).tolist()}))


@app.route('/polygonRNNRefine', methods=["POST", "GET"])
def predictPolygonRNNRefine():
    return 'This method is not yet implemented\n'


if __name__ == "__main__":
    print("Starting web service")
    app.run(host='0.0.0.0', debug=config.DEBUG_MODE,
            port=config.PORT, threaded=True, use_reloader=False)
