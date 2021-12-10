# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
#         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

import numpy as np
    
def lib_color_change(img, segment):
    # RGB order
    img = np.int32(img)
    
    mask = (segment == 12) | (segment == 13)
    img[..., 0][mask] += 20
    img[img > 255] = 255
    # img[..., 0][mask] = cv2.add(img[..., 0][mask], 100)

    return np.uint8(img)