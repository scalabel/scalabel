"""Endpoints for model prediction."""
import argparse
import io
import logging
import os
import time
from typing import Dict, List

import numpy as np
import requests
from flask import Flask, Response, jsonify, make_response, request
from PIL import Image

from .seg_base import SegBase

try:
    from .polyrnn_adapter import PolyrnnAdapter as SegModel
except ImportError:
    from .seg_dummy import SegDummy as SegModel  # type: ignore


def homepage() -> str:
    """Test hello world."""
    return "Test server for segmentation\n"


def load_images(urls: List[str]) -> Dict[str, np.ndarray]:
    """Load image for each url and cache results in a dictionary."""
    url_to_img = {}
    for url in urls:
        if url not in url_to_img:
            img_response = requests.get(url)
            img = np.array(Image.open(io.BytesIO(img_response.content)))
            url_to_img[url] = img
    return url_to_img


def predict_poly(seg_model: SegBase) -> Response:
    """Predict rect -> polygon."""
    logger = logging.getLogger(__name__)
    logger.info("Hitting prediction endpoint")
    start_time = time.time()
    receive_data = request.get_json()

    try:
        url_to_img = load_images([data["url"] for data in receive_data])
    except requests.exceptions.ConnectionError:
        response: Response = make_response("Bad image url provided.")
        return response

    images: List[np.ndarray] = []
    boxes: List[List[float]] = []
    for data in receive_data:
        box: Dict[str, float] = data["labels"][0]["box2d"]
        x1 = box["x1"]
        x2 = box["x2"]
        y1 = box["y1"]
        y2 = box["y2"]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        boxes.append(bbox)
        images.append(url_to_img[data["url"]])

    preds = seg_model.convert_rect_to_poly(images, boxes)

    logger.info("Time for prediction: %s", time.time() - start_time)
    response = make_response(jsonify({"points": preds}))
    return response


def refine_poly() -> str:
    """Refine poly -> poly."""
    return "This method is not yet implemented\n"


def create_app() -> Flask:
    """Set up the flask app."""
    app = Flask(__name__)
    home = os.path.join("scalabel", "bot")

    # pass to methods using closure
    model = SegModel(home)

    # url rules should match NodeJS endpoint names
    app.add_url_rule("/", view_func=homepage)
    app.add_url_rule(
        "/predictPoly", view_func=lambda: predict_poly(model), methods=["POST"]
    )
    app.add_url_rule("/refinePoly", view_func=refine_poly, methods=["POST"])

    return app


def launch() -> None:
    """Launch processes."""
    log_f = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=log_f)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.info("Launching model server")
    parser = argparse.ArgumentParser(
        description="Launch the server on one machine."
    )
    parser.add_argument(
        "--host", dest="host", help="server hostname", default="0.0.0.0"
    )
    parser.add_argument(
        "--port", dest="port", help="server port", default=8080
    )
    parser.add_argument(
        "--debugMode", dest="debugMode", help="server debug mode", default=1
    )
    args = parser.parse_args()

    app = create_app()

    app.run(
        host=args.host,
        debug=args.debugMode,
        port=args.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    launch()
