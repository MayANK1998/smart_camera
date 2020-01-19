import cv2
import os
import shutil
import time
from flask import Flask, render_template, Response,request
import requests
from bs4 import BeautifulSoup
from deep_learning_object_detection import VideoCamera
from cvtoi import extractImages
import sys
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,help="minimum probability to filter weak detections")
app = Flask(__name__)
mcnt = -1
mcnt2 = -1
etcount = 0;
@app.route('/')
def index():
    
    return render_template('index.html')

def gen(camera):
    
    count =0
    while True:
        frame,etcount = camera.get_frame(count,ap)
        #     # save frame as JPEG filea
        #atime.sleep(0.1)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                
@app.route('/video_feed')
def video_feed():
     #os.system('python cameratest.py')
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
     #os.system('python cameratest.py')
    return Response(gen2(VideoCamera2()),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':

    app.run(debug=True)
