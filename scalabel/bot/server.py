""" Endpoints for model prediction """
import argparse
import io
import logging
import time
from typing import Dict, List
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

def load_images(urls) -> Dict[str, np.ndarray]:
    """ load image for each url and cache results in a dictionary """
    url_to_img = {}
    for url in urls:
        if url not in url_to_img:
            img_response = requests.get(url)
            img = np.array(Image.open(io.BytesIO(img_response.content)))
            url_to_img[url] = img
    return url_to_img

def polyrnn_base(polyrnn: PolyrnnBase) -> Response:
    """ predict rect -> polygon """
    logger = logging.getLogger(__name__)
    logger.info('Hitting prediction endpoint')
    start_time = time.time()
    receive_data = request.get_json()

    try:
        url_to_img = load_images([data['url'] for data in receive_data])
    except requests.exceptions.ConnectionError:
        response: Response = make_response('Bad image url provided.')
        return response

    images: List[np.ndarray] = []
    boxes: List[List[float]] = []
    for data in receive_data:
        box: Dict[str, float] = data['labels'][0]['box2d']
        x1 = box['x1']
        x2 = box['x2']
        y1 = box['y1']
        y2 = box['y2']
        bbox = [x1, y1, x2 - x1, y2 - y1]
        boxes.append(bbox)
        images.append(url_to_img[data['url']])

    preds = polyrnn.predict_rect_to_poly(images, boxes)

    logger.info('Time for prediction: %s', time.time() - start_time)
    response = make_response(jsonify({'points': preds}))
    return response


def polyrnn_refine() -> str:
    """ predict poly -> poly """
    return 'This method is not yet implemented\n'


def create_app() -> Flask:
    """ set up the flask app """
    app = Flask(__name__)
    home = os.path.join('scalabel', 'bot')

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
