# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


# run line detection pipeline
def run_pipeline(img_name):


    # 1.- Read image
    img_colour = cv2.imread(img_name)

    # verify that image `img` exist
    if img_colour is None:
        print('ERROR: image ', img_name, 'could not be read')
        exit()

    # 2. Convert from BGR to RGB then from RGB to greyscale
    img_colour_rgb = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_colour_rgb, cv2.COLOR_RGB2GRAY)

    # 3.- Apply Gaussuan smoothing
    kernel_size = (21,21)
    blur_gray = cv2.GaussianBlur(gray, kernel_size, sigmaX=0, sigmaY=0)

    #blur = cv2.bilateralFilter(grey, 21, 0, 0)

    # 4.- Apply Canny edge detector
    low_threshold = 60
    high_threshold = 80
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold, apertureSize=3)


    plt.figure(1)
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.figure(2)
    plt.imshow(blur_gray, cmap='gray')
    plt.axis('off')

    plt.figure(3)
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.show()
    return None

# Run pipeline
img_name = 'pista.jpg'
run_pipeline(img_name)
