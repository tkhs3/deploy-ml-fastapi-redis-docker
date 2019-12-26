import sys 
import base64
import numpy as np


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")

def base64_decode_image(a, dtype, shape):

    # If this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # Convert the string to a NumPy array using the supplied data 
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
    # restore type and target shape
    if len(shape) == 4:
        a = a.reshape(shape)
    ## HWC shape is supplied
    elif len(shape) == 3 and shape[-1] in [1,3] :
        a = a.reshape((1,)+shape)
    ## HW or 1HW shape is supplied
    elif ( len(shape) == 2 ) or ( len(shape) == 3 and shape[0] == 1 ):
        size_array = len(a)
        size_shape = shape[0] * shape[1]
        if len(shape) == 3 :
            size_shape = size_shape * shape[2]
        n_c = size_array / size_shape
        _shape = shape + (int(n_c),)
        a = a.reshape(_shape)

    # Return the decoded image
    return a