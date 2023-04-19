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
import numpy as np
import sys
import cv2
from ex1_utils import *

title_window = 'Gamma Correction'
trackbar_name = 'Gamma:'
gamma_slider_max_val = 200
max_pix = 255
isColor = False

slider_max_val = 200  # this is for making 200 "jumps" between 0-2 with 0.01 in each step
window_name = "the gamma correction gui"

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    img = imReadAndConvert(img_path, rep)
    def on_trackbar(val):
        gamma = val / (slider_max_val/2)
        gamma_corrected = np.power(img, gamma)
        cv2.imshow(window_name, gamma_corrected)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    trackbar_name = f"Gamma "
    cv2.createTrackbar(trackbar_name, window_name, 50, slider_max_val, on_trackbar)
    cv2.waitKey()
    cv2.destroyAllWindows()
    sys.exit()


def main():
    gammaDisplay('dark.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
