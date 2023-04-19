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
from typing import List, Tuple

from typing import List
import cv2
import numpy as np
from sklearn.cluster import KMeans

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
RGB2YIQ_mat = np.array([0.258, 0.347, 0.133, 0.583, -0.333, -0.456, 0.211, 0.233, 0.521]).reshape(3, 3)


def myID() -> np.int32:
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
        sys.exit("could not read the image")
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    normalized_img = normalizeData(img)
    return normalized_img






def imDisplay(filename: str, representation: int) :
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    if representation == LOAD_GRAY_SCALE:
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        plt.imshow(img)
        plt.show()


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
    isColored = False
    YIQimg = 0
    tmpMat = imgOrig
    if len(imgOrig.shape) == 3:  # it's RGB convert to YIQ and take the Y dimension
        YIQimg = transformRGB2YIQ(imgOrig)
        tmpMat = YIQimg[:, :, 0]
        isColored = True
    tmpMat = cv2.normalize(tmpMat, None, 0, 255, cv2.NORM_MINMAX)
    tmpMat = tmpMat.astype('uint8')
    histOrg = np.histogram(tmpMat.flatten(), bins=256)[0]  # original image histogram
    cumSum = np.cumsum(histOrg)  # image cumSum

    LUT = np.ceil((cumSum / cumSum.max()) * 255)  # calculate the LUT table
    imEqualized = tmpMat.copy()
    for i in range(256):  # give the right value for each pixel according to the LUT table
        imEqualized[tmpMat == i] = int(LUT[i])

    histEq = np.histogram(imEqualized.flatten().astype('uint8'), bins=256)[0]  # equalized image histogram

    imEqualized = imEqualized / 255
    if isColored:  # RGB img -> convert back to RGB color space
        YIQimg[:, :, 0] = imEqualized
        imEqualized = transformYIQ2RGB(YIQimg)

    return imEqualized, histOrg, histEq



def quantize_image(im_orig: np.ndarray, n_quant: int, n_iter: int) -> Tuple[List[np.ndarray], List[float]]:
    """
    Quantizes an image into **n_quant** colors
    :param im_orig: The original image (RGB or Grayscale)
    :param n_quant: Number of colors to quantize the image to
    :param n_iter: Number of optimization loops
    :return: (List[q_image_i], List[error_i])
    """
    isColored = False
    YIQimg = 0
    tmpImg = im_orig
    if len(im_orig.shape) == 3:  # this is rgb convert to yiq color space and take the Y dimension
        YIQimg = transformRGB2YIQ(im_orig)
        tmpImg = YIQimg[:, :, 0]
        isColored = True
    tmpImg = cv2.normalize(tmpImg, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    Orig_copy = tmpImg.copy()

    # part 1 of the algorithm :  create the first division of borders according to the histogram
    histOrg = np.histogram(tmpImg.flatten(), bins=256)[0]
    cumSum = np.cumsum(histOrg)
    each_slice = cumSum.max() / n_quant  # ultimate size for each slice
    slices = [0]
    curr_sum = 0
    curr_ind = 0
    for i in range(1, n_quant + 1):  # divide it to slices for the first time.
        while curr_sum < each_slice and curr_ind < 256:
            curr_sum += histOrg[curr_ind]
            curr_ind = curr_ind + 1
        if slices[-1] != curr_ind - 1:
            curr_ind = curr_ind - 1
        slices.append(curr_ind)
        curr_sum = 0

    slices.pop()
    slices.insert(n_quant, 255)


    # part 3 of the algorithm : quantize the image
    images_list = []  # the images list for each iteration
    MSE_list = []  # the MSE(min squared error) list for each iteration.
    for i in range(n_iter):
        quantizeImg = np.zeros(tmpImg.shape)
        Qi = []
        for j in range(1, n_quant + 1):
            try:
                Si = np.array(
                    range(slices[j - 1], slices[j]))  # which of the intensities levels is within the range of this slice
                Pi = histOrg[slices[j - 1]:slices[j]]
                intensity_avg = int((Si * Pi).sum() / Pi.sum())  # the intensity level that is the average of this slice
                Qi.append(intensity_avg)
            except RuntimeWarning:
                Qi.append(0)
            except ValueError:
                Qi.append(0)

        # part 3.2 of the algorithm:  update the @quantizeImg according to the @Qi average values.
        for k in range(n_quant):
            quantizeImg[tmpImg > slices[k]] = Qi[k]

        slices.clear()
        # part 3.3 : update the slices according to the @Qi values -> slices[k] = average of the Qi[left] and Qi[right]
        for k in range(1, n_quant):
            slices.append(int((Qi[k - 1] + Qi[k]) / 2))

        slices.insert(0, 0)
        slices.insert(n_quant, 255)

        # part 3.4 : add MSE and check if done.
        MSE_list.append((np.sqrt((Orig_copy * 255 - quantizeImg) ** 2)).mean())
        tmpImg = quantizeImg
        images_list.append(quantizeImg / 255)
        if checkMSE(MSE_list, n_iter):  # check whether the last 5 MSE values were not changed if so -> break.
            break

    # part 4 : if @imOrig was in RGB color space convert it back.
    if isColored:
        for i in range(len(MSE_list)):
            YIQimg[:, :, 0] = images_list[i]
            images_list[i] = transformYIQ2RGB(YIQimg)

    return images_list, MSE_list




# This function checks if the last 5 values of the @MSE_list is the same -> if so returns true.
def checkMSE(MSE_list: List[float], nIter: int) -> bool:
    if len(MSE_list) > nIter / 10:
        for i in range(2, int(nIter / 10) + 1):
            if MSE_list[-1] != MSE_list[-i]:
                return False
    else:
        return False
    return True
def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))