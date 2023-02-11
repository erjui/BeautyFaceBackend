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

org = base64.b64decode(value)
org = np.frombuffer(org, dtype=np.uint8)
org = cv2.imdecode(org, cv2.IMREAD_COLOR)

segment = base64.b64decode(output['segment'])
segment = np.frombuffer(segment, dtype=np.uint8)
segment = cv2.imdecode(segment, cv2.IMREAD_GRAYSCALE)

from utils import label_visualize

cv2.imwrite('debug_function_test/0_image.jpg', np.hstack([org, img]))
cv2.imwrite('debug_function_test/1_segment.jpg', np.uint8(label_visualize(segment, 19) * 255.0))

# STEP 2. Enhance Lib Test
segment = output['segment']
result = requests.post('http://127.0.0.1:8080/', json={'type': 'enhance_lib', 'segment': segment, 'value': [10, 0, 0], 'data': value}, headers=headers)
# output = result.json()

# STEP 3.Enhance Skin Test
segment = output['segment']
result = requests.post('http://127.0.0.1:8080/', json={'type': 'enhance_skin', 'segment': segment, 'value': [10, 0, 0], 'data': value}, headers=headers)
# output = result.json()

# STEP 4.Enhance Eye Test
segment = output['segment']
result = requests.post('http://127.0.0.1:8080/', json={'type': 'enhance_eye', 'segment': segment, 'value': [10, 0, 30], 'data': value}, headers=headers)
# output = result.json()

# STEP 5. Enhance Nose Test
segment = output['segment']
result = requests.post('http://127.0.0.1:8080/', json={'type': 'enhance_nose', 'segment': segment, 'value': [10, 10, 10], 'data': value}, headers=headers)
# output = result.json()

# STEP 6. Enhance Brow Test
segment = output['segment']
result = requests.post('http://127.0.0.1:8080/', json={'type': 'enhance_brow', 'segment': segment, 'value': [0, 30, 0], 'data': value}, headers=headers)
# output = result.json()
