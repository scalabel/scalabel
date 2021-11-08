import torch
import numpy as np
import cv2

from .poly_point_isect import isect_polygon__naive_check


def check_self_intersection(poly):
    # The polygon MUST be in float
    return isect_polygon__naive_check(poly)


def count_self_intersection(polys, grid_size):
    """
    :param polys: Nx1 poly
    :return: number of polys that have self-intersection
    """
    new_polys = []
    isects = []
    for poly in polys:
        poly = get_masked_poly(poly, grid_size)
        poly = class_to_xy(poly, grid_size).astype(np.float32)
        isects.append(check_self_intersection(poly.tolist()))

    return np.array(isects, dtype=np.float32)


def xy_to_class(poly, grid_size):
    """
    NOTE: Torch function
    poly: [bs, time_steps, 2]

    Returns: [bs, time_steps] with class label
    for x,y location or EOS token
    """
    batch_size = poly.size(0)
    time_steps = poly.size(1)

    poly[:, :, 1] *= grid_size
    poly = torch.sum(poly, dim=-1)

    poly[poly < 0] = grid_size ** 2
    # EOS token

    return poly


def class_to_grid(poly, out_tensor, grid_size):
    """
    NOTE: Torch function
    accepts out_tensor to do it inplace

    poly: [batch, ]
    out_tensor: [batch, 1, grid_size, grid_size]
    """
    out_tensor.zero_()
    # Remove old state of out_tensor

    b = 0
    for i in poly:
        if i < grid_size * grid_size:
            x = (i % grid_size).long()
            y = (i / grid_size).long()
            out_tensor[b, 0, y, x] = 1
        b += 1

    return out_tensor


def class_to_xy(poly, grid_size):
    """
    NOTE: Numpy function
    poly: [bs, time_steps] or [time_steps]

    Returns: [bs, time_steps, 2] or [time_steps, 2]
    """
    x = (poly % grid_size).astype(np.int32)
    y = (poly / grid_size).astype(np.int32)

    out_poly = np.stack([x,y], axis=-1)

    return out_poly


def get_masked_poly(poly, grid_size):
    """
    NOTE: Numpy function

    Given a polygon of shape (N,), finds the first EOS token
    and masks the predicted polygon till that point
    """
    if np.max(poly) == grid_size**2:
        # If there is an EOS in the prediction
        length = np.argmax(poly)
        poly = poly[:length]
        # This automatically removes the EOS

    return poly


def get_vertices_mask(poly, mask):
    """
    Generate a vertex mask
    """
    mask[poly[:, 1], poly[:, 0]] = 1.

    return mask


def draw_poly(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    cv2.fillPoly(mask, [poly], 255)

    return mask


def poly0g_to_poly01(polygon, grid_side):
    """
    [0, grid_side] coordinates to [0, 1].
    Note: we add 0.5 to the vertices so that the points
    lie in the middle of the cell.
    """
    result = (polygon.astype(np.float32) + 0.5)/grid_side

    return result

def poly01_to_poly0g(poly, grid_size):
    """
    [0, 1] coordinates to [0, grid_size] coordinates

    Note: simplification is done at a reduced scale
    """
    poly = np.floor(poly * grid_size).astype(np.int32)
    poly = cv2.approxPolyDP(poly, 0, False)[:, 0, :]

    return poly