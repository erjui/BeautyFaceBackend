import requests

with open('base64_sample/1.txt', ) as f:
    value = f.read()

# with open('base64_sample/segment.txt', ) as f:
#     segment = f.read()

# STEP 1. Inference Test
headers = {'Content-Type': 'application/json'}
result = requests.post('http://127.0.0.1:8080/', json={'type': 'inference', 'data': value}, headers=headers)
output = result.json()

import cv2
import base64
import numpy as np
img = base64.b64decode(output['data'])
img = np.frombuffer(img, dtype=np.uint8)
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

org = base64.b64decode(output['org'])
org = np.frombuffer(org, dtype=np.uint8)
org = cv2.imdecode(org, cv2.IMREAD_COLOR)

segment = base64.b64decode(output['segment'])
segment = np.frombuffer(segment, dtype=np.uint8)
segment = cv2.imdecode(segment, cv2.IMREAD_GRAYSCALE)

from utils import label_visualize

cv2.imwrite('debug_function_test/0_image.jpg', np.hstack([org, img]))
cv2.imwrite('debug_function_test/1_segment.jpg', np.uint8(label_visualize(segment, 19) * 255.0))

# STEP 2. Enhance Test
segment = output['segment']
result = requests.post('http://127.0.0.1:8080/', json={'type': 'enhance', 'segment': segment, 'lib': [255, 255, 255], 'data': value}, headers=headers)
output = result.json()