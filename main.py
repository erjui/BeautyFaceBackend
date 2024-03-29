import cv2
import json
import base64
import numpy as np
from model import SegmentModel
from torchvision.transforms import transforms
from utils import label_visualize
import face_detection

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from img_process import lib_color_change, skin_color_change, eye_color_change, nose_color_change

# STEP. Load pretrained model
print(face_detection.available_detectors)
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3, device='cpu')
detector.net.eval()
detector.net.cpu()
print(f"Face detection model loaded.")

# model = SegmentModel()
model = SegmentModel.load_from_checkpoint('checkpoints/epoch=85-step=64500.ckpt', map_location='cpu')
model.eval()
model.cpu()
print(f"Segmentation model loaded.")

# TODO: refactor enhance functions

def inference(request):
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

    if request_json['type'] == 'inference':
        img = request_json['data']
        img = base64.b64decode(img)
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*normalize)
        ])

        detections = detector.detect(img)
        detection = detections[0]
        xmin, ymin, xmax, ymax, conf = detection.astype(np.int32)
        top, right, bottom, left = ymin, xmax, ymax, xmin

        # Emprical resizing face size
        top = max(top - int((bottom - top) * 0.2), 0)
        bottom = min(bottom + int((bottom - top) * 0.2), img.shape[0])
        left = max(left - int((right - left) * 0.2), 0)
        right = min(right + int((right - left) * 0.2), img.shape[1])

        org_img = img.copy()
        img = img[top:bottom, left:right]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

        img_t = transform(img).unsqueeze(0)
        out_t = model(img_t)
        out = out_t[0].numpy()

        out_v = label_visualize(out, 19)
        out_v = np.uint8(out_v * 255.0)
        img_result = lib_color_change(img.copy(), out)

        cv2.imwrite('debug_inference/0_result.jpg', np.hstack([img, out_v, img_result])[:, :, ::-1])

        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        out_big = org_img.copy()
        result_patch = cv2.resize(img_result, dsize=(right-left, bottom-top), interpolation=cv2.INTER_CUBIC)
        out_big[top:bottom, left:right] = result_patch

        cv2.imwrite('debug_inference/1_result_big.jpg', np.hstack([org_img, out_big])[:, :, ::-1])

        out_big = cv2.cvtColor(out_big, cv2.COLOR_RGB2BGR)

        _, img_base64 = cv2.imencode('.png', out_big)
        img_base64 = img_base64.tobytes()
        img_base64 = base64.b64encode(img_base64)

        segment = np.ones(out_big.shape[:2], dtype=np.uint8) * 255
        out = cv2.resize(out, dsize=(right-left, bottom-top), interpolation=cv2.INTER_NEAREST)
        segment[top:bottom, left:right] = out

        cv2.imwrite('debug_inference/2_segment.jpg', np.uint8(label_visualize(segment, 19) * 255.0))

        # REMIND. jpg compression yeild bad results
        _, segment_base64 = cv2.imencode('.png', segment)
        segment_base64 = segment_base64.tobytes()
        segment_base64 = base64.b64encode(segment_base64)

        # Set CORS headers for the main request
        headers = {
            'Access-Control-Allow-Origin': '*'
        }

        return (json.dumps({"data": img_base64.decode('utf8'), "segment": segment_base64.decode('utf8') }), 200, headers)

    elif request_json['type'] == 'enhance_lib':
        img = request_json['data']
        img = base64.b64decode(img)
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        cv2.imwrite('debug_enhance_lib/0_org.jpg', img)

        segment = request_json['segment']
        segment = base64.b64decode(segment)
        segment = np.frombuffer(segment, dtype=np.uint8)
        segment = cv2.imdecode(segment, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite('debug_enhance_lib/1_segment.jpg', np.uint8(label_visualize(segment, 19) * 255.0))

        value = request_json['value']
        value.reverse()
        img_result = lib_color_change(img.copy(), segment, value)

        cv2.imwrite('debug_enhance_lib/2_result.jpg', img_result)

        _, img_base64 = cv2.imencode('.png', img_result)
        img_base64 = img_base64.tobytes()
        img_base64 = base64.b64encode(img_base64)

        # Set CORS headers for the main request
        headers = {
            'Access-Control-Allow-Origin': '*'
        }

        return (json.dumps({"data": img_base64.decode('utf8')}), 200, headers)

    elif request_json['type'] == 'enhance_skin':
        img = request_json['data']
        img = base64.b64decode(img)
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        cv2.imwrite('debug_enhance_skin/0_org.jpg', img)

        segment = request_json['segment']
        segment = base64.b64decode(segment)
        segment = np.frombuffer(segment, dtype=np.uint8)
        segment = cv2.imdecode(segment, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite('debug_enhance_skin/1_segment.jpg', np.uint8(label_visualize(segment, 19) * 255.0))

        value = request_json['value']
        value.reverse()
        img_result = skin_color_change(img.copy(), segment, value)

        cv2.imwrite('debug_enhance_skin/2_result.jpg', img_result)

        _, img_base64 = cv2.imencode('.png', img_result)
        img_base64 = img_base64.tobytes()
        img_base64 = base64.b64encode(img_base64)

        # Set CORS headers for the main request
        headers = {
            'Access-Control-Allow-Origin': '*'
        }

        return (json.dumps({"data": img_base64.decode('utf8')}), 200, headers)

    elif request_json['type'] == 'enhance_eye':
        img = request_json['data']
        img = base64.b64decode(img)
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        cv2.imwrite('debug_enhance_eye/0_org.jpg', img)

        segment = request_json['segment']
        segment = base64.b64decode(segment)
        segment = np.frombuffer(segment, dtype=np.uint8)
        segment = cv2.imdecode(segment, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite('debug_enhance_eye/1_segment.jpg', np.uint8(label_visualize(segment, 19) * 255.0))

        value = request_json['value']
        value.reverse()
        img_result = eye_color_change(img.copy(), segment, value)

        cv2.imwrite('debug_enhance_eye/2_result.jpg', img_result)

        _, img_base64 = cv2.imencode('.png', img_result)
        img_base64 = img_base64.tobytes()
        img_base64 = base64.b64encode(img_base64)

        # Set CORS headers for the main request
        headers = {
            'Access-Control-Allow-Origin': '*'
        }

        return (json.dumps({"data": img_base64.decode('utf8')}), 200, headers)

    elif request_json['type'] == 'enhance_nose':
        img = request_json['data']
        img = base64.b64decode(img)
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        cv2.imwrite('debug_enhance_nose/0_org.jpg', img)

        segment = request_json['segment']
        segment = base64.b64decode(segment)
        segment = np.frombuffer(segment, dtype=np.uint8)
        segment = cv2.imdecode(segment, cv2.IMREAD_GRAYSCALE)

        cv2.imwrite('debug_enhance_nose/1_segment.jpg', np.uint8(label_visualize(segment, 19) * 255.0))

        value = request_json['value']
        value.reverse()
        img_result = nose_color_change(img.copy(), segment, value)

        cv2.imwrite('debug_enhance_nose/2_result.jpg', img_result)

        _, img_base64 = cv2.imencode('.png', img_result)
        img_base64 = img_base64.tobytes()
        img_base64 = base64.b64encode(img_base64)

        # Set CORS headers for the main request
        headers = {
            'Access-Control-Allow-Origin': '*'
        }

        return (json.dumps({"data": img_base64.decode('utf8')}), 200, headers)

if __name__ == '__main__':
    pass