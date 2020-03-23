""" Dummy interface with the polyrnn model """
from typing import List
import numpy as np
from scalabel.polyrnn_base import CropData, PolyrnnBase


class PolyrnnDummy(PolyrnnBase):
    """ Class for polyrnn dummy interface """

    def predict_from_rect(self, crop_img: np.ndarray) -> np.ndarray:
        """ predict rect -> dummy poly inscribed in rect """
        h, w, _ = crop_img.shape
        preds = [[w/2.0, 0], [w, h/2.0], [w/2.0, h], [0, h/2.0]]
        return np.array(preds)

    def bbox_to_crop(self, img: np.ndarray, bbox: List[float]) -> CropData:
        """ crop out rectangle of bbox """
        x0, y0, w, h = bbox
        crop_img = img[int(y0):int(y0+h), int(x0):int(x0+w)]
        starting_point = np.array([x0, y0])
        crop_dict = CropData(
            img=crop_img, start=starting_point, scale_factor=1)
        return crop_dict

    def rescale_output(self, preds: np.ndarray,
                       crop_dict: CropData) -> List[List[float]]:
        """ undo the cropping to get original output coords """
        preds = crop_dict.start + preds

        output: List[List[float]] = preds.tolist()
        return output
