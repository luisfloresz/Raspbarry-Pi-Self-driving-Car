"""
	Line detection using Hough lines

	Luis Felipe Flores
	Universidad de Monterrey
"""

# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time

start = time.time()

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
def run_pipeline():

    # initialise a video capture object
    cap = cv2.VideoCapture("Video_Pista2.h264")

    # check that the videocapture object was successfully created
    if(not cap.isOpened()):
        print("Error opening video source")
        exit()
    
    # create new windows for visualisation purposes
    cv2.namedWindow('input video', cv2.WINDOW_AUTOSIZE)

    while (cap.isOpened):
        
        flagstart = time.time()
        
        # grab current frame
        ret, frame = cap.read()        

        # verify that frame was properly captured
        if ret == False:
            print("ERROR: current frame could not be read")            
            break

        # 1.- Read image
        #frame_rs = cv2.resize(frame,(640,360))
        img_colour = frame


        # verify that image `img` exist
        if img_colour is None:
            print('ERROR: image ', img_name, 'could not be read')
            exit()

        # 2. Convert from BGR to RGB then from RGB to greyscale
        img_colour_rgb = img_colour
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
        bottom_left = (0, 175)
        top_left = (0, 40)
        top_right = (640, 40)
        bottom_right = (640, 175)
        

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
        hough_lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

        # 8.- Visualise input and output images
        img_colour_with_lines = img_colour_rgb.copy()

        # 9.- Define left and right lane
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        if(hough_lines !=None):
            for line in hough_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (0,0,255), 3)
                    slope = float(y2 - y1) / (x2 - x1) # <-- Calculating the slope.
                    #print(slope)
                    if math.fabs(slope) < 0.3: # <-- Only consider extreme slope
                        continue
                    if slope <= 0: # <-- If the slope is negative, left group.
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else: # <-- Otherwise, right group.
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])

        min_y = 0 # <-- Just below the horizon
        #max_y = 350 # <-- The bottom of the image
        max_y = 175
        

        if(left_line_y != [] and left_line_x != []):
            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
                ))

        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        if(right_line_y != [] and right_line_x != []):
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


        flagend = time.time()
        #print(' Run time: ')
        runTime = flagend - flagstart
        fps = 1/runTime
        #print(runTime)
        #print(' Fps: ')
        #print(fps)
        Fps = 'Fps = ' + str(fps)

        # Add FPS to img
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_colour_with_fulllines,Fps,(0,20), font, 1,(255,255,255),2,cv2.LINE_AA)

        # Calculate min left and right point in y
        if(left_line_y !=[] and right_line_y !=[]):
            left_min = max(left_line_y)
            right_min = max(right_line_y)
##          print(left_min)
##          print(right_min)
            direction = 'left = ' + str(left_min) + ' Right = ' + str(right_min)
            #print(direction)

            if (left_min > (right_min + 20)):
                motor = 'right'
                #rotateRight()
                #time.sleep(0.1)
                #moveFoward()
            elif(right_min > (left_min + 20)):
                motor = 'left'
                #rotateLeft()
                #time.sleep(0.1)
                #moveFoward()
            else:
                #moveFoward()
                motor = 'foward'
            #print(motor)

            cv2.putText(img_colour_with_fulllines,motor,(0,60), font, 1,(255,255,255),2,cv2.LINE_AA)
        

        #visualise current frame
        cv2.imshow('input video',img_colour_with_fulllines)

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('carriles.png', img_colour_with_fulllines)
            cv2.imwrite('carriles_segmentados.png', img_colour_with_fulllines)
            break

           
    return None

# Run pipeline
run_pipeline()


