def inference(request):
    import base64
    import PIL.Image
    import numpy as np
    from io import BytesIO

    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    print(request)

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request_args and 'name' in request_args:
        name = request_args['name']
    else:
        name = 'World'

    cat = request_json['cat']
    cat = BytesIO(base64.b64decode(cat))
    img = PIL.Image.open(cat)
    img = np.array(img)

    print(img.shape)
    print(name)

    img_base64 = base64.b64encode(img)

    return img_base64

if __name__ == '__main__':
    pass