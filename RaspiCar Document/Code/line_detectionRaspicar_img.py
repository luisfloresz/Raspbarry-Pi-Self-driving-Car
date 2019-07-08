"""
    Line detection in an img using Hough lines

    Luis Felipe Flores Zertuche
    Universidad de Monterrey
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# select a region of interest
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



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
    grey = cv2.cvtColor(img_colour_rgb, cv2.COLOR_RGB2GRAY)

    # 3.- Apply Gaussuan smoothing
    kernel_size = (21,21)
    blur_grey = cv2.GaussianBlur(grey, kernel_size, sigmaX=0, sigmaY=0)

    # 4.- Apply Canny edge detector
    low_threshold = 60
    high_threshold = 80
    edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize=3)

    # 5.- Define a polygon-shape like region of interest
    img_shape = grey.shape


    # Region of interest
    bottom_left = (0, 350)
    top_left = (0, 80)
    top_right = (1280, 80)
    bottom_right = (1280, 350)

    # create a vertices array that will be used for the roi
    vertices = np.array([[bottom_left,top_left, top_right, bottom_right]], dtype=np.int32)

    # 6.- Get a region of interest using the just created polygon. This will be
    #     used together with the Hough transform to obtain the estimated Hough lines
    masked_edges = region_of_interest(edges, vertices)

    # 7.- Apply Hough transform for lane lines detection
    rho = 1                       # distance resolution in pixels of the Hough grid
    theta = np.pi/180             # angular resolution in radians of the Hough grid
    threshold = 10                # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 5              # minimum number of pixels making up a line
    max_line_gap = 10              # maximum gap in pixels between connectable line segments
    line_image = np.copy(img_colour)*0   # creating a blank to draw lines on
    hough_lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    # 8.- Visualise input and output images
    img_colour_with_lines = img_colour_rgb.copy()
##    if(hough_lines !=None):
##        for line in hough_lines:
##            for x1, y1, x2, y2 in line:
##                cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,0,255), 3)
##                slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
##                print(slope)

    # 9.- Define left and right lane
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if(hough_lines != []):
        for line in hough_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,0,255), 3)
                slope = float(y2 - y1) / (x2 - x1) # <-- Calculating the slope.
                #print(slope)
                # if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                #     continue
                if slope <= 0: # <-- If the slope is negative, left group.
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else: # <-- Otherwise, right group.
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

    min_y = 0 # <-- Just below the horizon
    max_y = 230 # <-- The bottom of the image


    poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
    ))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    fullline = [[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
                ]]
    
    #print(fullline)

    img_colour_with_fulllines = img_colour_rgb.copy()    
    for line in fullline:
    	for x1, y1, x2, y2 in line:
    		cv2.line(img_colour_with_fulllines, (x1, y1), (x2, y2), (0,255,0), 3)

    pf_bottom_left = (left_x_end , min_y)
    pf_top_left = (left_x_start, max_y)
    pf_top_right = (right_x_start, max_y)
    pf_bottom_right = (right_x_end, min_y)

    pf_vertices = np.array([[pf_bottom_left,pf_top_left, pf_top_right, pf_bottom_right]], dtype=np.int32)
    cv2.fillPoly(img_colour_with_fulllines, pf_vertices, (255,0,0))


    # visualise input and output images
##    plt.figure(1)
##    plt.imshow(masked_edges)
##    plt.axis('off')
##
##    plt.figure(2)
##    plt.imshow(blur_grey, cmap='gray')
##    plt.axis('off')
##
##    plt.figure(3)
##    plt.imshow(edges, cmap='gray')  
##    plt.axis('off')

##    plt.figure(4)
##    plt.imshow(img_colour_with_lines)
##    plt.axis('off')

    plt.figure(5)
    plt.imshow(img_colour_with_fulllines, cmap='gray')
    plt.axis('off')

    #print (hough_lines)
    print(max(left_line_y))
    print(max(right_line_y))


    plt.show()
    return None

# Run pipeline
img_name = 'pista.jpg'
run_pipeline(img_name)


