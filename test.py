from __future__ import division
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json 
import random 
import cv2
import numpy as np
from keras.preprocessing import image
import os
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

m=0

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h, c = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36,48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    print (A,B,C)
	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
    return ear

#load saved model
model = load_model('vgg16_2_model.h5')

# For naming the classes
l_ = []
for f in os.listdir('dataset2/train/'):
    l_.append(f.upper())

l_ = sorted(l_)
people = {}
for i,person in enumerate(l_):
    people[i] = person.title()


#testing
# Loading the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# def face_extractor(img):
#     # Function detects faces and returns the cropped face
#     # If no face detected, it returns the input image
    
#     faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
#     if faces is ():
#         return None
    
#     # Crop all faces found
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
#         cropped_face = img[y:y+h, x:x+w]

#     return cropped_face


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
#camera = cv2.VideoCapture(1)

predictor_path = 'shape_predictor_68_face_landmarks.dat_2'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total=0


while True:
    _, frame = video_capture.read()
    #frame1 = camera.read()
    
    cropped_face=None
#     face=face_extractor(frame)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)  
    if faces is ():
        face= None  
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,200,0),2)
        cropped_face = frame[y:y+h, x:x+w]
    face=cropped_face

    #frame_grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame, width=120)
        
        
    cv2.putText(frame,"Press 'q' to quit", (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)
    
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        pid = np.argmax(pred,axis=1)[0]
        name="None matching"
        name = people[pid]
            
        cv2.putText(frame,name, (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,220),2)
            cropped_face = frame[y:y+h, x:x+w]
        face=cropped_face
        name='Not-Recognized'
        cv2.putText(frame,name, (x, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,220), 2)

    dets = detector(frame_resized, 1)
    for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            leftEye= shape[lStart:lEnd]
            rightEye= shape[rStart:rEnd]
            leftEAR= eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
	       
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear>.29:
                print (ear)
                m=1
                print ('o')
                cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if m==1:
                    total+=1
                    m=0
                    cv2.putText(frame, "blink" ,(250, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
                print (ear)
                print ('c')
                
                cv2.putText(frame, "Eyes close".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Total Count: {}".format(total), (410, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            for (x, y) in shape:
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)

    cv2.imshow('Video', frame)
    #cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

