# USAGE
# python predict_fire.py

# import the necessary packages
from tensorflow.keras.models import load_model
from pyimagesearch import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

# load the trained model from disk
class PredictFire(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        print("[INFO] loading model...")
        self.model = load_model(config.MODEL_PATH)
        self.cap = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.cap.release()
       
    def getCameraFrames(self,count,ap):

# Create a VideoCapture object and read from input file 
        
   
# Check if camera opened successfully 
        if (self.cap.isOpened()== False):  
            print("Error opening video  file") 
   
# Read until video is completed 
          
      # Capture frame-by-frame 
        ret, frame = self.cap.read() 
        if ret == True: 
       
        # Display the resulting frame 
            ret, jpeg = cv2.imencode('.jpg',self.obj_detection(self.predict(frame),ap)); 
            return jpeg.tobytes();
        # Press Q on keyboard to  exit 
        
           
        # When everything done, release  
        # the video capture object 

# loop over the sampled image paths
    def predict(self,image):
        # load the image and clone it
        
        
        # grab the paths to the fire and non-fire images, respectively
        print("[INFO] predicting...")
        output = image.copy()
        
            # resize the input image to be a fixed 128x128 pixels, ignoring
            # aspect ratio
        image = cv2.resize(image, (128, 128))
        image = image.astype("float32") / 255.0
                
            # make predictions on the image
        preds = self.model.predict(np.expand_dims(image, axis=0))[0]
        j = np.argmax(preds)
        label = config.CLASSES[j]
        
        # draw the activity on the output frame
        text = label if label == "Non-Fire" else "WARNING! Fire!"
        output = imutils.resize(output, width=500)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.25, (0, 255, 0), 5)
        # write the output image to disk     
       
        return output
    def obj_detection(self,image1,ap):
        args = vars(ap.parse_args())
        args["image"] = image1
        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor","fan"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        # (note: normalization is done via the authors of the MobileNet SSD
        # implementation)
        image = image1
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    
        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
    
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
    
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
    
                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
        # show the output image
        return image
