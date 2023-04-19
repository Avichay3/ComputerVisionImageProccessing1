from  ex1_utils import *
from gamma import *
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

def histEqDemo(img_path: str, rep: int) :
    img = imReadAndConvert(img_path, rep)
    imgeq, histOrg, histEq = hsitogramEqualize(img)

    # Display the cum sum
    cum_sum = np.cumsum(histOrg)
    cum_sumEq = np.cumsum(histEq)
    plt.gray()
    plt.plot(range(256), cum_sum, 'r')
    plt.plot(range(256), cum_sumEq, 'g')

    # Display the images
    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(imgeq)
    plt.show()


def quantDemo(img_path: str, rep: int):
    img = imReadAndConvert(img_path, rep)
    st = time.time()

    img_lst, err_lst = quantize_image(img, 3, 20)

    print("Time:%.2f" % (time.time() - st))
    print("Error 0:\t %f" % err_lst[0])
    print("Error last:\t %f" % err_lst[-1])

    plt.gray()
    plt.imshow(img_lst[0])
    plt.figure()
    plt.imshow(img_lst[-1])

    plt.figure()
    plt.plot(err_lst, 'r')
    plt.show()


def main():
    print("ID:", myID())
    img_path = 'dark.jpg'

    # basic read and display to the screen image in gray scale and in rgb
    #  imDisplay(img_path, LOAD_GRAY_SCALE)
    #  imDisplay(img_path, LOAD_RGB)

    """
    rgb_img = imReadAndConvert(img_path, LOAD_RGB)
    yiq_img = transformRGB2YIQ(rgb_img)  # transform the rgb image to yiq image with numpy arrays
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb_img)  # put in the left of the plot the rgb image
    ax[1].imshow(yiq_img)  # as above, but in the right put the yiq image
    plt.show()  # will show 2 photos side by side. the left will be in rgb and the right will be in yiq
    """

    
    
    #  Image histEq
    #histEqDemo(img_path, LOAD_GRAY_SCALE)
    #histEqDemo(img_path, LOAD_RGB)


    # Image Quantization
    # quantDemo(img_path, LOAD_GRAY_SCALE)
    quantDemo(img_path, LOAD_RGB *32)

    # Gamma
    # gammaDisplay(img_path, LOAD_RGB)


if __name__ == '__main__':
     main()
