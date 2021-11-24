def inference(request):
    import base64
    import PIL.Image
    import numpy as np
    from io import BytesIO
    import json
    import cv2

    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
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
        

    print('request', request)

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request_args and 'name' in request_args:
        name = request_args['name']
    else:
        name = 'World'

    print('name', name)
    print('json', request_json)

    cat = request_json['cat']
    print('bf decoding', cat)
    cat = base64.b64decode(cat)
    # cat = BytesIO(cat)
    # img = PIL.Image.open(cat)
    # img = np.array(img)
    img = np.frombuffer(cat, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    print('img shape', img.shape)

    # img_base64 = base64.b64encode(img)
    _, img_base64 = cv2.imencode('.jpg', img)
    img_base64 = img_base64.tobytes()
    img_base64 = base64.b64encode(img_base64)

    print('af encoding', img_base64)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    # return img_base64, 
    return (json.dumps({"cat": img_base64.decode('utf8')}), 200, headers)

if __name__ == '__main__':
    pass