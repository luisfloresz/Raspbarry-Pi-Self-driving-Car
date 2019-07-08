"""
    Raspberry Controled car

    Control the car with arrow keys,
    Enter = Stop
    f = Take photo
    v = Start recording
    s = Stop recording

    Luis Felipe Flores Zertuche
    Universidad de Monterrey
"""

import RPi.GPIO as GPIO
import time
import curses
import picamera
from datetime import datetime

print('Raspberry Controled car ','\n','Control the car with arrow keys','\n')
print('Enter = Stop','\n','f = Take photo','\n','v = Start recording','\n','s = Stop recording')

# Create a time object, datetime.now() = date & time with microseconds
#time = datetime.now()

# Create a camera object from picamera
camera = picamera.PiCamera()
camera.resolution = (640,360)      # 1/3 of max resolution
camera.framerate = 20

# Get the curses window, turn off echoing of keyboard to screen, turn on
# instant (no waiting) key response, and use special values for cursor keys
screen = curses.initscr()
curses.noecho() 
curses.cbreak()
screen.keypad(True)

GPIO.setmode(GPIO.BCM)

motorLeftF = 5          # GPIO 5, move left wheels foward, IN1
motorLeftR = 6          # GPIO 6, move left wheels backward, IN2

motorRightF = 19        # GPIO 13, move right wheels foward, IN3
motorRightR = 13        # GPIO 19, move right wheels backward, IN4


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

try:
        while True:   
            char = screen.getch()
            if char == ord('q'):
                break
            elif char == curses.KEY_UP:
                moveFoward()
            elif char == curses.KEY_DOWN:
                moveBackward()
            elif char == curses.KEY_RIGHT:
                rotateRight()
            elif char == curses.KEY_LEFT:
                rotateLeft()
            elif char == 10:
                carStop()
            elif char == ord('k'):
                rotateBRight()
            elif char == ord('l'):
                rotateBLeft()
            elif char == ord('f'):
                camera.capture(str(datetime.now())+'.jpg')
                print('Snapshot saved')
            elif char == ord('v'):
                camera.start_recording(str(datetime.now()) +'.h264')
                print('Start recording, press S to stop')
            elif char == ord('s'):
                camera.stop_recording()
                print('Video saved')
             
finally:
    #Close down curses properly, inc turn echo back on!
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()

GPIO.cleanup()
