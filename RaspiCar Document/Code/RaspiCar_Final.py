"""
	Self driving car using Raspberry Pi

	Luis Felipe Flores
	Universidad de Monterrey
"""

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
import math
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 360)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(640, 360))
 
# allow the camera to warmup
time.sleep(0.1)

GPIO.setmode(GPIO.BCM)

motorLeftF = 5          # GPIO 5, move left wheels foward, IN1
motorLeftR = 6          # GPIO 6, move left wheels backward, IN2

motorRightF = 19        # GPIO 13, move right wheels foward, IN3
motorRightR = 13        # GPIO 19, move right wheels backward, IN4

GPIO.setwarnings(False)

GPIO.setup(motorLeftF, GPIO.OUT)
GPIO.setup(motorLeftR, GPIO.OUT)
GPIO.setup(motorRightF, GPIO.OUT)
GPIO.setup(motorRightR, GPIO.OUT)

def moveFoward():
    GPIO.output(motorLeftF, GPIO.LOW)
    GPIO.output(motorLeftR, GPIO.HIGH)
    GPIO.output(motorRightF, GPIO.LOW)
    GPIO.output(motorRightR, GPIO.HIGH)

def moveBackward():
    GPIO.output(motorLeftF, GPIO.HIGH)
    GPIO.output(motorLeftR, GPIO.LOW)
    GPIO.output(motorRightF, GPIO.HIGH)
    GPIO.output(motorRightR, GPIO.LOW)

def rotateRight():
    GPIO.output(motorLeftF, GPIO.HIGH)
    GPIO.output(motorLeftR, GPIO.HIGH)
    GPIO.output(motorRightF, GPIO.LOW)
    GPIO.output(motorRightR, GPIO.HIGH)

def rotateLeft():
    GPIO.output(motorLeftF, GPIO.LOW)
    GPIO.output(motorLeftR, GPIO.HIGH)
    GPIO.output(motorRightF, GPIO.HIGH)
    GPIO.output(motorRightR, GPIO.HIGH)

def rotateBLeft():
    GPIO.output(motorLeftF, GPIO.LOW)
    GPIO.output(motorLeftR, GPIO.LOW)
    GPIO.output(motorRightF, GPIO.HIGH)
    GPIO.output(motorRightR, GPIO.LOW)

def rotateBRight():
    GPIO.output(motorLeftF, GPIO.HIGH)
    GPIO.output(motorLeftR, GPIO.LOW)
    GPIO.output(motorRightF, GPIO.LOW)
    GPIO.output(motorRightR, GPIO.LOW)

def carStop():
    GPIO.output(motorLeftF, GPIO.LOW)
    GPIO.output(motorLeftR, GPIO.LOW)
    GPIO.output(motorRightF, GPIO.LOW)
    GPIO.output(motorRightR, GPIO.LOW)
    
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
        flagstart = time.time()
	
        image = frame.array
	
	# 1.- Read image
        #frame_rs = cv2.resize(frame,(640,360))
        img_colour = image


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

        # 6.- Apply Hough transform for lane lines detection
        rho = 1                       # distance resolution in pixels of the Hough grid
        theta = np.pi/180             # angular resolution in radians of the Hough grid
        threshold = 10                # minimum number of votes (intersections in Hough grid cell)
        min_line_len = 5              # minimum number of pixels making up a line
        max_line_gap = 10              # maximum gap in pixels between connectable line segments
        line_image = np.copy(img_colour)*0   # creating a blank to draw lines on
        hough_lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

        # 7.- Visualise input and output images
        img_colour_with_lines = img_colour_rgb.copy()

        # 8.- Define left and right lane
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
                    if math.fabs(slope) < 0.2: # <-- Only consider extreme slope
                        continue
                    if slope <= 0: # <-- If the slope is negative, left group.
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else: # <-- Otherwise, right group.
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])

        min_y = 0
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
            
        img_colour_with_fulllines = img_colour_rgb.copy()
        if (left_line_y != [] and left_line_x != [] and right_line_y != [] and right_line_x != []):
            fullline = [[
                        [left_x_start, max_y, left_x_end, min_y],
                        [right_x_start, max_y, right_x_end, min_y],
                        ]]
            
            #print(fullline)
    
            for line in fullline:
                for x1, y1, x2, y2 in line:
                        cv2.line(img_colour_with_fulllines, (x1, y1), (x2, y2), (0,255,0), 3)

        

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
                rotateRight()
                time.sleep(0.1)
                moveFoward()
            elif(right_min > (left_min + 20)):
                motor = 'left'
                rotateLeft()
                time.sleep(0.1)
                moveFoward()
            else:
                moveFoward()
                motor = 'foward'
            #print(motor)

            #cv2.putText(img_colour_with_fulllines,motor,(0,60), font, 1,(255,255,255),2,cv2.LINE_AA)
        

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
        

        #visualise current frame
        cv2.imshow('input video',img_colour_with_fulllines)

        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #cv2.imwrite('carriles.png', img_colour_with_fulllines)
            #cv2.imwrite('carriles_segmentados.png', img_colour_with_fulllines)
            carStop()
            break
 
	# clear the stream in preparation for the next frame
        rawCapture.truncate(0)
 
