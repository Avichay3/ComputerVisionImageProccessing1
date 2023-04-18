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
from matplotlib import pyplot as plt
from typing import List

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
        img = img.convert('RGB')  # convert it to RGB colors
    img = np.asarray(img)  # after convert the image to gray/RGB scale, need to make it as numpy array
    img = img.astype(np.float)  # convert it to floating point array
    normalized_img = normalizeData(img)  # normalize to the same scale of values the numpy array with the floating point
    return normalized_img



def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation)  # step 1: convert the image into numpy array
    cv2.imshow(filename, image)  # display the image
    cv2.waitKey(0)  # wait for user to press something in keybord
    cv2.destoryAllWindows()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    matrix = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    imgYIQ = np.dot(imgRGB, matrix)  # apply the conversion matrix to the image by multiply the two arrays
    return imgYIQ  # return the YIQ image



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
    imageRGB = np.dot(imgYIQ, matrix)  # function in numpy that multiply two matrix
    imageRGB = np.clip(imageRGB, 0.0, 1.0)  # we want to clip the pixels in the RGB image to values between [0,1]
    return imageRGB



def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    im = (imgOrig * 255).astype(np.uint8)  # convert image to uint8 and scale it to [0, 255]
    histOrg, _ = np.histogram(im.flatten(), bins=256, range=(0, 255))  # calculate histogram of original image
    cumSum = histOrg.cumsum()  # calculate cumulative sum of histogram
    cumSumNorm = cumSum / cumSum[-1]  # normalize

    LookUpTable = np.round(cumSumNorm * 255).astype(np.uint8)  # Create LookUpTable for histogram equalization
    imEq = cv2.LUT(im, LookUpTable)  # Apply LookUpTable to input image
    imEq = imEq.astype(np.float32) / 255.0  # Convert equalized image to float and scale to [0, 1]

    histEQ, _ = np.histogram(imEq.flatten(), bins=256, range=(0, 1))  # Calculate histogram of equalized image

    # Display input and equalized images
    cv2.imshow("Input Image", imgOrig)
    cv2.imshow("Equalized Image", imEq)
    cv2.waitKey(0)

    return imEq, histOrg, histEQ



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    """
    #  the next function compute the histogram of the input image
    def compute_histogram(im, n_bins=256):
        hist, _ = np.histogram(im, bins=n_bins, range=(0, 1))
        return hist

    #  this next function compute the cumulative distribution  of an input histogram
    # returns the normalized cumulative histogram
    def compute_cumulative_histogram(hist):
        cumsum = np.cumsum(hist)
        return cumsum / cumsum[-1]

    #  this next function gets an image and 2 arrays of quantization values,
    #  and computes the mean-squared error between the original image and the quantized image
    def compute_error(im, z, q):
        quantized_im = np.zeros_like(im)  # create new numpy array with the same shape as im
        for i in range(len(z) - 1):  # iterate all over quantization intervals defined by the z values.
            mask = (im >= z[i]) & (im < z[i + 1])
            quantized_im[mask] = q[i]  # assigns the value of q[i] to the corresponding elements in quantized_im where the mask is true.
        quantized_im[im >= z[-1]] = q[-1]  # sets all pixel values in quantized_im that are greater than or equal to the last quantization level in z,
                                           # to the last quantization value in q.
        error = np.sum(np.square(im - quantized_im))
        return error

    # this next function gets an image array and also another two arrays z and q
    # and returns a new set of optimized quantization levels z_new and q_new, and also the corresponding error.
    def find_optimal_zq(im, z, q):
        hist = compute_histogram(im)
        cumsum = compute_cumulative_histogram(hist)
        z_new = np.zeros(nQuant + 1)
        q_new = np.zeros(nQuant)
        for i in range(1, nQuant):
            s = np.argmax(cumsum >= i / nQuant)
            z_new[i] = (z[s] + z[s - 1]) / 2
            q_new[i - 1] = np.mean(im[(im >= z[s - 1]) & (im <= z[s])])   # q_new is set to the average pixel value within each interval
        z_new[nQuant] = 1
        return z_new, q_new, compute_error(im, z_new, q_new)


    def quantize(im):
        if len(im.shape) == 3:
            yiq = transformRGB2YIQ(im)
            y = yiq[:, :, 0]
            z = np.linspace(0, 1, nQuant + 1)
            q = np.linspace(0, 1, nQuant)
            for _ in range(nIter):
                z, q, error = find_optimal_zq(y, z, q)
                y = np.interp(y, z, q)
            yiq[:, :, 0] = y
            quantized_image = transformYIQ2RGB(yiq)
        else:
            z = np.linspace(0, 1, nQuant + 1)
            q = np.linspace(0, 1, nQuant)
            for _ in range(nIter):
                z, q, error = find_optimal_zq(im, z, q)
                im = np.interp(im, z, q)
            quantized_image = im
        return quantized_image, error

    ims = [imOrig]
    errors = []
    for i in range(nIter):
        quantized_im, error = quantize(ims[-1])
        ims.append(quantized_im)
        errors.append(error)
    plt.plot(errors)
    plt.xlabel("Iteration")
    plt.ylabel("Total Error")
    plt.show()
    return ims[1:], errors
"""

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))