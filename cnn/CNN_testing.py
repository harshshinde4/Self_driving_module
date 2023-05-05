import tflearn 
import imutils
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
import cv2
import numpy as np
import tensorflow as tf
import time 
IMG_SIZE =100
object_count=0
LR = 1e-3#0.001
tf.reset_default_graph() 
convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 

convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.5) 

convnet = fully_connected(convnet,3, activation ='softmax')
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
	loss ='categorical_crossentropy', name ='targets') 

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
model = tflearn.DNN(convnet, tensorboard_dir ='log') 
model.load('self driving .model')
label=['Left turn','Right Turn','Go Straight']
CLASSES = ["", "", "", "", "",
    "", "bus", "car", "", "", "", "",
    "", "", "", "", "", "",
    "", "", ""] 
def cNN_test(frame):
    # frame=cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newimg = cv2.resize(gray,(int(IMG_SIZE),int(IMG_SIZE)))
    data = newimg.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    model_out=list(model_out)
    print(model_out)
    val=max(model_out)
    idx=model_out.index(max(model_out))
    print(idx)
    return idx
cam = cv2.VideoCapture('test01.mp4')
count=0
st=time.time()
while(count<250):
    ret,img=cam.read()
    count=count+1
    if ret == True:
        frame=cv2.resize(img,(600,400))
        frame=frame[200:,:]
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0,0,i,2]
            if(confidence > 0.5):
                
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if CLASSES[idx]=='car'or CLASSES[idx]=='bus':
                    object_count+=1

                cv2.putText(frame,CLASSES[idx], (startX,startY),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                # cv2.rectangle(frame, (startX, startY), (endX, endY),(0,255,0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0,0,255), 2)
                    
        id=cNN_test(frame)
        cv2.putText(frame, label[id], (350,25),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("live",frame)
        cv2.waitKey(10)
et=time.time()

print("total Time Elapse=",et-st)
print("total object detcted=",object_count)
