# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:26:46 2019

@author: Mayank
"""
import cv2
class PredictFire(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.cap = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.cap.release()
       
    def getCameraFrames(self):

# Create a VideoCapture object and read from input file 
        
   
# Check if camera opened successfully 
        if (self.cap.isOpened()== False):  
            print("Error opening video  file") 
   
# Read until video is completed 
          
      # Capture frame-by-frame 
        ret, frame = self.cap.read() 
        if ret == True: 
       
        # Display the resulting frame 
            cv2.imshow('Frame', frame) 
       
        # Press Q on keyboard to  exit 
        
           
        # When everything done, release  
        # the video capture object 
        
p = PredictFire();
while(True):
    p.getCameraFrames();
    if cv2.waitKey(25) & 0xFF == ord('q'): 
          break
cv2.destroyAllWindows() 