# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
#         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

def inference(request):
    import cv2
    import json
    import base64
    import numpy as np
    from model import SegmentModel
    from torchvision.transforms import transforms
    from utils import label_visualize
    import face_detection

    def lib_color_change(img, segment):
        # RGB order
        img = np.int32(img)
        
        mask = (segment == 12) | (segment == 13)
        img[..., 0][mask] += 50
        img[img > 255] = 255
        # img[..., 0][mask] = cv2.add(img[..., 0][mask], 100)

        return np.uint8(img)

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

    print(face_detection.available_detectors)
    detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)
    detections = detector.detect(img)
    detection = detections[0]
    xmin, ymin, xmax, ymax, conf = detection.astype(np.int32) 
    top, right, bottom, left = ymin, xmax, ymax, xmin


    org_img = img.copy()
    img = img[top:bottom, left:right]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

    model = SegmentModel.load_from_checkpoint('checkpoints/epoch=84-step=148749.ckpt')
    # model = SegmentModel()
    model.eval()
    model.cpu()
    print('pretrained model loaded')

    img_t = transform(img).unsqueeze(0)
    out_t = model(img_t)
    out = out_t[0].numpy()

    out_v = label_visualize(out, 19)
    out_v = np.uint8(out_v * 255.0)
    img_result = lib_color_change(img.copy(), out)

    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', np.hstack([img, out_v, img_result]))
    # cv2.waitKey(0)

    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    out_big = org_img.copy()
    result_patch = cv2.resize(img_result, dsize=(right-left, bottom-top), interpolation=cv2.INTER_CUBIC)
    out_big[top:bottom, left:right] = result_patch

    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', np.hstack([org_img, out_big]))
    # cv2.waitKey(0)

    out_big = cv2.cvtColor(out_big, cv2.COLOR_RGB2BGR)

    _, img_base64 = cv2.imencode('.jpg', out_big)
    img_base64 = img_base64.tobytes()
    img_base64 = base64.b64encode(img_base64)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    return (json.dumps({"data": img_base64.decode('utf8'), "org": request_json['data']}), 200, headers)

if __name__ == '__main__':
    pass