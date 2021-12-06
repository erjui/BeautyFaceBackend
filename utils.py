import random
import numpy as np
import torch

def fix_seed(seed):
    random.seed(seed) # random
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # torch
    torch.cuda.manual_seed(seed) # cuda
    torch.cuda.manual_seed_all(seed) # cuda multi-GPU
    torch.backends.cudnn.deterministic = True # cudnn
    torch.backends.cudnn.benchmark = False # cudnn

# color dictionary
# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
#         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

color_dict = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (70, 70, 70),
    4: (102, 102, 156),
    5: (190, 153, 153),
    6: (153, 153, 153),
    7: (250, 170, 30),
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (152, 251, 152),
    11: (70, 130, 180),
    12: (220, 20, 60),
    13: (255, 0, 0),
    14: (0, 0, 142),
    15: (0, 0, 70),
    16: (0, 60, 100),
    17: (0, 80, 100),
    18: (0, 0, 230),
    19: (119, 11, 32),
}

def labelVisualize(img, num_class, color_dict=color_dict):
    import numpy as np
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255
