""" Endpoints for model prediction """
import argparse
import io
import logging
import time
from typing import Dict
import os
import numpy as np
import requests
from flask import Flask, request, jsonify, make_response, Response
from PIL import Image
from .polyrnn_base import PolyrnnBase
try:
    from .polyrnn_interface import PolyrnnInterface as Polyrnn
except ImportError:
    from .polyrnn_dummy import PolyrnnDummy as Polyrnn  # type: ignore


def homepage() -> str:
    """ hello world test """
    return 'Test server for PolygonRNN++\n'


def polyrnn_base(polyrnn: PolyrnnBase) -> Response:
    """ predict rect -> polygon """
    logger = logging.getLogger(__name__)
    logger.info('Hitting prediction endpoint')
    start_time = time.time()
    data = request.get_json()
    url: str = data['url']
    box: Dict[str, float] = data['labels'][0]['box2d']
    x1 = box['x1']
    x2 = box['x2']
    y1 = box['y1']
    y2 = box['y2']
    bbox = [x1, y1, x2 - x1, y2 - y1]

    try:
        img_response = requests.get(url)
    except requests.exceptions.ConnectionError:
        response: Response = make_response('Bad image url provided.')
        return response

    img = np.array(Image.open(io.BytesIO(img_response.content)))
    preds = polyrnn.predict_rect_to_poly(np.array(img), bbox)

    logger.info('Time for prediction: %s', time.time() - start_time)
    response = make_response(jsonify({'points': preds}))
    return response


def polyrnn_refine() -> str:
    """ predict poly -> poly """
    return 'This method is not yet implemented\n'


def create_app() -> Flask:
    """ set up the flask app """
    app = Flask(__name__)
    home = os.path.join('scalabel', 'bot', 'models')

    # pass to methods using closure
    polyrnn = Polyrnn(home)

    # url rules should match NodeJS endpoint names
    app.add_url_rule('/', view_func=homepage)
    app.add_url_rule('/polygonRNNBase',
                     view_func=lambda: polyrnn_base(polyrnn),
                     methods=['POST'])
    app.add_url_rule('/polygonRNNRefine',
                     view_func=polyrnn_refine,
                     methods=['POST'])

    return app


def launch() -> None:
    """ main process launcher """
    log_f = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=log_f)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.info('Launching model server')
    parser = argparse.ArgumentParser(
        description='Launch the server on one machine.')
    parser.add_argument('--host',
                        dest='host',
                        help='server hostname',
                        default='0.0.0.0')
    parser.add_argument('--port',
                        dest='port',
                        help='server port',
                        default=8080)
    parser.add_argument('--debugMode',
                        dest='debugMode',
                        help='server debug mode',
                        default=1)
    args = parser.parse_args()

    app = create_app()

    app.run(host=args.host,
            debug=args.debugMode,
            port=args.port,
            threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    launch()
