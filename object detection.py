import cv2
import numpy as np
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
cap = cv2.VideoCapture('test01.mp4')
CLASSES = ["", "", "", "", "",
    "", "bus", "car", "", "", "", "",
    "", "", "", "", "", "",
    "", "", ""]  
count=0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
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
            cv2.putText(frame,CLASSES[idx], (startX,startY),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                # cv2.rectangle(frame, (startX, startY), (endX, endY),(0,255,0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0,0,255), 2)
        
    cv2.imshow("live",frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break  
cap.release()
cv2.destroyAllWindows()