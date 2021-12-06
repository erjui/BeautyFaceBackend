import requests

with open('base64_sample/cat.txt', ) as f:
    value = f.read()
    value = value[1:]

headers = {'Content-Type': 'application/json'}
result = requests.post('http://192.168.0.81:8080/', json={'data': value}, headers=headers)
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

from utils import label_visualize

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', np.hstack([org, np.uint8(label_visualize(img) * 255.0)]))
cv2.waitKey(0)