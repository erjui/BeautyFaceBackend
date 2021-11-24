def inference(request):
    import cv2
    import json
    import base64
    import numpy as np

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
        
    request_json = request.get_json(silent=True)

    cat = request_json['data']
    cat = base64.b64decode(cat)
    img = np.frombuffer(cat, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    _, img_base64 = cv2.imencode('.jpg', img)
    img_base64 = img_base64.tobytes()
    img_base64 = base64.b64encode(img_base64)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    return (json.dumps({"data": img_base64.decode('utf8')}), 200, headers)

if __name__ == '__main__':
    pass