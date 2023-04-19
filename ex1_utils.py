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
    norm_img = normalizeData(img)
    return norm_img






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



def quantize_image(im_orig: np.ndarray, n_quant: int, n_iter: int) -> Tuple[List[np.ndarray], List[float]]:
    """
    Quantizes an image into **n_quant** colors
    :param im_orig: The original image (RGB or Grayscale)
    :param n_quant: Number of colors to quantize the image to
    :param n_iter: Number of optimization loops
    :return: (List[q_image_i], List[error_i])
    """
    is_colored = len(im_orig.shape) == 3
    tmp_img = cv2.normalize(im_orig, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    hist_org, _ = np.histogram(tmp_img.flatten(), bins=256)
    cum_sum = np.cumsum(hist_org)
    each_slice = cum_sum.max() / n_quant
    slices = np.zeros(n_quant + 1, dtype=int)
    curr_sum = 0
    curr_ind = 0
    for i in range(1, n_quant + 1):
        while curr_sum < each_slice and curr_ind < 256:
            curr_sum += hist_org[curr_ind]
            curr_ind += 1
        if slices[i - 1] != curr_ind - 1:
            curr_ind -= 1
        slices[i] = curr_ind
        curr_sum = 0

    q_images = []
    mse_list = []
    for i in range(n_iter):
        quantize_img = np.zeros(tmp_img.shape)
        qi = []
        for j in range(1, n_quant + 1):
            si = np.arange(slices[j-1], slices[j])
            pi = hist_org[slices[j-1]:slices[j]]
            avg = int((si * pi).sum() / pi.sum())
            qi.append(avg)

        for k in range(n_quant):
            quantize_img[tmp_img > slices[k]] = qi[k]

        slices = [(qi[i] + qi[i + 1]) // 2 for i in range(len(qi) - 1)]

        slices = np.insert(slices, 0, 0)
        slices = np.append(slices, 255)

        mse = np.sqrt((tmp_img - quantize_img) ** 2).mean()
        mse_list.append(mse)
        q_images.append(quantize_img / 255)

        if len(mse_list) > n_iter / 10 and all(mse_list[-1] == mse for mse in mse_list[-int(n_iter / 10):]):
            break

        tmp_img = quantize_img

    if is_colored:
        yiq_img = transformRGB2YIQ(im_orig)
        for i, q_img in enumerate(q_images):
            q_img_resized = cv2.resize(q_img, dsize=(im_orig.shape[1], im_orig.shape[0]), interpolation=cv2.INTER_LINEAR)
            q_img_gray = np.mean(q_img_resized, axis=2)
            yiq_img[:, :, 0] = q_img_gray
            q_images[i] = transformYIQ2RGB(yiq_img)

    return q_images, mse_list


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))