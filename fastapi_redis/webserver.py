
import base64
import io
import json
import os
import time
import uuid

import numpy as np
from PIL import Image
import redis

from fastapi import FastAPI, File, HTTPException
from starlette.requests import Request


class processor():

    # self reference to processor class
    me = None
    # FastAPI instance
    app = None
    # Redis instance
    db = None

    @staticmethod
    def init(app, db, me=None):

        # set attribute for base class
        __class__.me = me or __class__
        __class__.db = db
        __class__.app = app

    @staticmethod
    def _preprocess_image(image, target):
        # If the image mode is not RGB, convert it
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize the input image and preprocess it
        image = image.resize(target)
        image = np.array(image)
        # image = img_to_array(image)
        # image = np.expand_dims(image, axis=0)
        # image = imagenet_utils.preprocess_input(image)

        # Ensure our NumPy array is C-contiguous as well, otherwise we won't be able to serialize it
        image = image.copy(order="C")

        # Return the processed image
        return image

    @staticmethod
    def _decode_image(byte_image):
        image = Image.open(io.BytesIO(byte_image))
        return image

    @staticmethod
    def _prepare_image(byte_image):
        # me = __class__.me or __class__.me
        image = __class__.me._decode_image(byte_image) 
        image = __class__.me._preprocess_image(image,
                            (int(os.environ.get("N_WIDTH_IMAGE")),
                            int(os.environ.get("N_HEIGHT_IMAGE")))
                            )
        return image

    @staticmethod
    def _encode_contents(val):
        val_encoded = base64.b64encode(val).decode("utf-8")
        return val_encoded

    @staticmethod
    def _make_queue(key=None, val=None):
        assert val is not None

        k = str(uuid.uuid4())
        k = key or k

        val_encoded = __class__.me._encode_contents(val)

        d = {"id": k, "image": val_encoded}
        return d

    @staticmethod
    def _decode_contents(val_json):
        val = json.loads(val_json)
        return val

# init self reference
processor.me = processor



class_processor = processor
if class_processor.app is not None:

    # @staticmethod
    @class_processor.app.get("/")
    def index():
        return "Hello World!"

    # @staticmethod
    @class_processor.app.post("/predict")
    def predict(request: Request, byte_image: bytes=File(...)):

        data = {"success": False}

        if request.method == "POST":
            
            image = class_processor._prepare_image(byte_image)

            # Generate an ID for the classification then add the classification ID + image to the queue
            key_queue = str(uuid.uuid4())
            dict_queue = class_processor._make_queue(id=key_queue, val=image)

            class_processor.db.rpush(os.environ.get("NAME_QUEUE"), json.dumps(dict_queue))

            # Keep looping for CLIENT_MAX_TRIES times
            num_tries = 0
            while num_tries < CLIENT_MAX_TRIES:
                num_tries += 1

                # Attempt to grab the output predictions
                output = class_processor.db.get(key_queue)

                # Check to see if our model has classified the input image
                if output is not None:

                    # Add the output predictions to our data dictionary so we can return it to the client
                    data["predictions"] = class_processor._decode_contents(output)

                    # Delete the result from the database and break from the polling loop
                    class_processor.db.delete(key_queue)
                    break

                # Sleep for a small amount to give the model a chance to classify the input image
                time.sleep(float(os.environ.get("CLIENT_SLEEP")))

                # Indicate that the request was a success
                data["success"] = True
            else:
                raise HTTPException(status_code=400, detail="Request failed after {} tries".format(CLIENT_MAX_TRIES))

        # Return the data dictionary as a JSON response
        return data

