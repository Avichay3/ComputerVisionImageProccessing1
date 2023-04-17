"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import sys
from typing import List
from matplotlib import pyplot as plt

import cv2 as cv2
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
RGB2YIQ_mat = np.array([0.258, 0.347, 0.133, 0.583, -0.333, -0.456, 0.211, 0.233, 0.521]).reshape(3, 3)


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 211780267


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    if img is None:
        sys.exit("cannot read the image")
    if representation == 1:   # means that the image is in gray scale
        img = img.convert('L')  # make it gray scale
    if representation == 2:
        img = img.convert('RGB') # convert it to RGB colors
    img = np.asarray(img) # after convert the image to gray/RGB scale, need to make it as numpy array
    img = img.astype(np.float) # convert it to floating point array
    normalized_img = normalizeData(img) # normalize to the same scale of values the numpy array with the floating point
    return normalized_img



def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation) # step 1: convert the image into numpy array
    cv2.imshow(filename, image)  # display the image
    cv2.waitKey(0) # wait for user to press something in keybord
    cv2.destoryAllWindows()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    matrix = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    imgYIQ = np.dot(imgRGB, matrix) # apply the conversion matrix to the image by multiply the two arrays
    return imgYIQ # return the YIQ image



def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """

    """
    the next formulas will be the formula that we gonna use for convert between YIQ to RGB color spaces
    --> R = Y + 0.956I + 0.621Q
    --> G = Y - 0.272I - 0.647Q
    --> B = Y - 1.106I + 1.703Q
    [ R ]     [ 1.000  0.956  0.621 ]   [ Y ]
    [ G ]  =  [ 1.000 -0.272 -0.647 ] * [ I ]
    [ B ]     [ 1.000 -1.106  1.703 ]   [ Q ]
    """
    matrix = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]])
    imageRGB = np.dot(imgYIQ, matrix) # function in numpy that multiply two matrix
    imageRGB = np.clip(imageRGB, 0.0, 1.0) # we want to clip the pixels in the RGB image to values between [0,1]
    return imageRGB



def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))