import pickle

import numpy as np
import cv2 as cv
import tensorflow as tf

from keras import backend
from tensorflow import keras


class HandwritingRecognizer:
    """
    Main class that does OCR on handwritten small notes
    """
    def __init__(self, model_path: str = "models/ocr_model_10_epoch.h5") -> None:
        self.model_path = model_path
        self.__load_encoders()
        self.__max_length = 80

    def __load_encoders(self):
        self.ocr = keras.models.load_model(self.model_path)

        with open("models/char_to_num.pkl", "rb") as f:
            self.char_to_num = pickle.load(f)

        with open("models/num_to_char.pkl", "rb") as f:
            self.num_to_char = pickle.load(f)

    @staticmethod
    def __read_image(im_source):
        if isinstance(im_source, str):
            try:
                im_source = cv.imread(im_source)
            except Exception as E:
                raise ValueError("Problem reading the image source")

        if isinstance(im_source, np.ndarray):
            im = im_source
            if not isinstance(im, np.ndarray):
                raise ValueError("Problem reading the image source")

            return im

        else:
            raise ValueError("Image source is neither image path nor numpy array")

    def inference(self, im_source: str | np.ndarray) -> str:
        """
        This method does OCR of the given image, image can be passed as image path in string or as numpy array
        :param im_source: (str or np.ndarray) image source
        :return: (str) OCR output
        """
        im = HandwritingRecognizer.__read_image(im_source)

        im = HandwritingRecognizer.__prepare(im)
        yp = self.ocr.predict(im)
        input_len = np.ones(yp.shape[0]) * yp.shape[1]
        results = backend.ctc_decode(yp, input_length=input_len, greedy=True)[0][0][:, :self.__max_length]

        out = ""
        for res in results:
            for token in res:
                out += self.num_to_char[token]

        return out

    @staticmethod
    def __prepare(im):
        w, h = 128, 32
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        im = np.expand_dims(im, axis=-1)

        image = tf.image.resize(im, size=(h, w), preserve_aspect_ratio=True)

        # Check tha amount of padding needed to be done.
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]

        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )

        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)

        image = tf.cast(image, tf.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        return image
