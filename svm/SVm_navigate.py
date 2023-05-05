from sklearn.svm import SVC
import pickle
from skimage.transform import resize
import cv2
import numpy as np
import time 
model=pickle.load(open('finalized_model.sav', 'rb'))
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
cap = cv2.VideoCapture('InShot_20220403_121921259.mp4')
label=['Left turn','Right Turn','Go Stright']
CLASSES = ["", "", "", "", "",
    "", "bus", "car", "", "", "", "",
    "", "", "", "", "", "",
    "", "", ""]  
count=0
object_count=0
st=time.time()
while(count<250):
    # Capture frame-by-frame
    ret, frame = cap.read()
    count=count+1
    if ret == True:
        frame=cv2.resize(frame,(600,400))
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
        img_array=resize(frame,(150,150,3))
        img_array = (img_array.flatten())
        pred=model.predict([img_array])
        
        cv2.putText(frame, label[pred[0]], (350,25),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("live",frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break  
et=time.time()

print("Total time Elapse=",et-st)
print("total object detcted=",object_count)
