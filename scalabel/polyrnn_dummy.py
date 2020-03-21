""" Dummy interface with the polyrnn model """
from typing import List
import numpy as np
from polyrnn_base import PolyrnnBase


class PolyrnnDummy(PolyrnnBase):
    """ Class for polyrnn dummy interface """

    def predict_from_rect(self, crop_img):
        """ predict rect -> dummy poly inscribed in rect """
        h, w, _ = crop_img.shape
        preds = [[w/2.0, 0], [w, h/2.0], [w/2.0, h], [0, h/2.0]]
        return [np.array(p) for p in preds]

    def bbox_to_crop(self, img, bbox):
        """ crop out rectangle of bbox """
        x0, y0, w, h = bbox
        crop_img = img[int(y0):int(y0+h), int(x0):int(x0+w)]
        crop_dict = {
            'img': crop_img,
            'start': [x0, y0]
        }
        return crop_dict

    def rescale_output(self, preds: List[np.ndarray],
                       crop_dict) -> List[List[float]]:
        """ undo the cropping to get original output coords """
        start = np.array(crop_dict['start'])
        preds = [start + p for p in preds]
        return np.array(preds).tolist()
