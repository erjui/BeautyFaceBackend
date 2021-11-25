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

def labelVisualize(num_class, color_dict, img):
    import numpy as np
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out


def inference(request):
    import cv2
    import json
    import base64
    import numpy as np
    from model import SegmentModel
    from torchvision.transforms import transforms

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows POST requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)
        
    request_json = request.get_json(silent=True)

    cat = request_json['data']
    cat = base64.b64decode(cat)
    img = np.frombuffer(cat, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize)
    ])

    shape = img.shape
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # model = SegmentModel.load_from_checkpoint('checkpoints/epoch=89-step=224999.ckpt')
    model = SegmentModel()
    print('pretrained model loaded')

    img = transform(img).unsqueeze(0)
    out = model(img).permute(1, 2, 0)
    out = out.detach().cpu().numpy()
    out = np.uint8(out)
    out = cv2.resize(out, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    out = labelVisualize(19, color_dict, out)

    _, img_base64 = cv2.imencode('.jpg', out)
    img_base64 = img_base64.tobytes()
    img_base64 = base64.b64encode(img_base64)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    return (json.dumps({"data": img_base64.decode('utf8')}), 200, headers)

if __name__ == '__main__':
    pass