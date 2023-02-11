# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
#         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

import numpy as np

def lib_color_change(img, segment, value=[20, 0, 0]):
    # RGB order
    img = np.int32(img)

    mask = (segment == 12) | (segment == 13)
    img[..., 0][mask] += value[0]
    img[..., 1][mask] += value[1]
    img[..., 2][mask] += value[2]
    img[img > 255] = 255
    img[img < 0] = 0

    return np.uint8(img)

def eye_color_change(img, segment, value=[20, 0, 0]):
    # RGB order
    img = np.int32(img)

    mask = (segment == 4) | (segment == 5)
    img[..., 0][mask] += value[0]
    img[..., 1][mask] += value[1]
    img[..., 2][mask] += value[2]
    img[img > 255] = 255
    img[img < 0] = 0

    return np.uint8(img)

def nose_color_change(img, segment, value=[20, 0, 0]):
    # RGB order
    img = np.int32(img)

    mask = (segment == 10)
    img[..., 0][mask] += value[0]
    img[..., 1][mask] += value[1]
    img[..., 2][mask] += value[2]
    img[img > 255] = 255
    img[img < 0] = 0

    return np.uint8(img)

def brow_color_change(img, segment, value=[20, 0, 0]):
    # RGB order
    img = np.int32(img)

    mask = (segment == 2) | (segment == 3)
    img[..., 0][mask] += value[0]
    img[..., 1][mask] += value[1]
    img[..., 2][mask] += value[2]
    img[img > 255] = 255
    img[img < 0] = 0

    return np.uint8(img)

def skin_color_change(img, segment, value=[20, 0, 0]):
    # RGB order
    img = np.int32(img)

    mask = (segment == 1) | (segment == 2) | (segment == 3) | (segment == 8) | (segment == 9) | (segment == 10) | (segment == 14) | (segment == 15)
    img[..., 0][mask] += value[0]
    img[..., 1][mask] += value[1]
    img[..., 2][mask] += value[2]
    img[img > 255] = 255
    img[img < 0] = 0

    return np.uint8(img)