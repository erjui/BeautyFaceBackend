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
    out = out * 10

    # TODO: add label colorization

    print(out.shape)

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