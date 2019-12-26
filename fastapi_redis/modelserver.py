import base64
import json
import os
import sys
import time
import itertools

import numpy as np
import redis


# Connect to Redis server
# db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

# Load the pre-trained Keras model (here we are using a model
# pre-trained on ImageNet and provided by Keras, but you can
# substitute in your own networks just as easily)
# model = ResNet50(weights="imagenet")



class processor():

    # self reference to processor class
    me = None
    model = None
    db = None

    @staticmethod
    def init(model, db, me=None):

        # set attribute for base class
        __class__.me = me or __class__
        __class__.db = db
        __class__.model = model


    @staticmethod
    def _decode_image(a, dtype=np.uint8, shape=(256,256)):
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

    @staticmethod
    def _decode_queue(queue):
        contents = json.loads(queue.decode("utf-8"))
        image = __class__.me._decode_image(
            contents["image"],
            os.environ.get("NAME_TYPE_IMAGE"),
            (   
                1, 
                int(os.environ.get("N_HEIGHT_IMAGE")),
                int(os.environ.get("N_WIDTH_IMAGE")),
                int(os.environ.get("N_CHANNEL_IMAGE"))
            )
        )
        contents["image"] = image
        return contents

    @staticmethod
    def _decode_queues(queues):

        list_id = []
        array_images = None
        for q in queues:

            # Deserialize the object and obtain the input image
            contents = __class__.me._decode_queue(q)
            image = contents["image"]

            # Check to see if the batch list is None
            if array_images is None:
                array_images = image

            # Otherwise, stack the data
            else:
                array_images = np.vstack([array_images, image])

            # Update the list of image IDs
            list_id.append(contents["id"])

        return list_id, array_images

    @staticmethod
    def _fetch_queues():
        # Pop off multiple images from Redis queue atomically
        with __class__.me.db.pipeline() as pipe:
            pipe.lrange(os.environ.get("NAME_QUEUE"), 0, int(os.environ.get("BATCH_SIZE")) - 1)
            pipe.ltrim(os.environ.get("NAME_QUEUE"), int(os.environ.get("BATCH_SIZE")), -1)
            queues, _ = pipe.execute()

        return queues

    @staticmethod
    def classify_process():

        # Continually poll for new images to classify
        while True:

            queues = __class__.me._fetch_queues()
            if len(queues) == 0:
                continue

            list_id, samples = __class__.me._decode_queues(queues)

            # Check to see if we need to process the batch
            if len(list_id) > 0:

                # Classify the batch
                if isinstance(samples, np.ndarray):
                    print("* Batch size: {}".format(samples.shape))
                else:
                    print("* Batch size: {}".format(len(list_id)))
                preds = __class__.me._predict_on_samples(samples)

                results = __class__.me._postprocess_samples(preds)

                __class__.me._return_queues(list_id, results)

            # Sleep for a small amount
            time.sleep(float(os.environ.get("SERVER_SLEEP")))

    @staticmethod
    def _predict_on_samples(samples):
        preds = __class__.me.model.predict(samples)

        return preds

    @staticmethod
    def _postprocess_samples(preds):
        results = imagenet_utils.decode_predictions(preds)
        return results

    @staticmethod
    def _return_queues(list_id, results):

        # Loop over the image IDs and their corresponding set of results from our model
        for (imageID, resultSet) in zip(list_id, results):
            # Initialize the list of output predictions
            output = []

            # Loop over the results and add them to the list of output predictions
            for (imagenetID, label, prob) in resultSet:
                r = {"label": label, "probability": float(prob)}
                output.append(r)

            # Store the output predictions in the database, using image ID as the key so we can fetch the results
            db.set(imageID, json.dumps(output))

# init self reference
processor.me = processor

if __name__ == "__main__":
    processor.classify_process()
