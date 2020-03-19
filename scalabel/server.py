import argparse
import io
import logging
import time
import numpy as np
import requests
from polyrnn_pp.cityscapes import DataProvider
from polyrnn_interface import PolyrnnInterface
from PIL import Image
from flask import Flask, request, jsonify, make_response
import tensorflow as tf


def homepage():
    """ hello world test """
    return 'Test server for PolygonRNN++\n'


def polyrnn_base(interface):
    """ predict rect -> polygon """
    logger = logging.getLogger(__name__)
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

    preds = interface.predict_from_rect(crop_img)
    start = np.array(crop_dict['starting_point'])

    # translate back to image space
    preds = [start + p * 224.0/crop_dict['scale_factor'] for p in preds]
    preds = np.array(preds).tolist()

    logger.info('Time for prediction: %s',
                time.time() - start_time)
    return make_response(jsonify({'points': preds}))


def polyrnn_refine():
    """ predict poly -> poly """
    return 'This method is not yet implemented\n'


def create_app() -> Flask:
    """ set up the flask app """
    app = Flask(__name__)

    # pass to methods using closure
    interface = PolyrnnInterface()

    # url rules should match NodeJS endpoint names
    app.add_url_rule('/', view_func=homepage)
    app.add_url_rule('/polygonRNNBase',
                     view_func=lambda: polyrnn_base(interface), methods=['POST'])
    app.add_url_rule('/polygonRNNRefine',
                     view_func=polyrnn_refine, methods=['POST'])

    return app


def launch() -> None:
    """ main process launcher """
    # logging
    log_format = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=log_format)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

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
