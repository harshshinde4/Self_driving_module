import pickle
from skimage.transform import resize
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
IMG_DIR = 'test'
model=pickle.load(open('finalized_model.sav', 'rb'))
features_list = []
class_list=[] 
predicted_list=[]  
count_of_total_class=0
count_of_total_predricted=0 
for img in os.listdir(IMG_DIR):
    count_of_total_class+=1
    frame = cv2.imread(os.path.join(IMG_DIR,img))
    img_array=resize(frame,(150,150,3))
    img_array = (img_array.flatten())
    pred=model.predict([img_array])
    predicted_list.append(pred[0])
    if(img[0]=='l'):
        class_list.append(0)
        if pred[0]==0:
            count_of_total_predricted+=1
    elif(img[0]=='R'):
        class_list.append(1)
        if pred[0]==1:
            count_of_total_predricted+=1
    elif(img[0]=='t'):
        class_list.append(2) 
        if pred[0]==2:
            count_of_total_predricted+=1
        
    
print("Actual Class",class_list)
print("Actual class_count",count_of_total_class)
print("predicted Class",predicted_list)
print("predicted Class count",count_of_total_predricted)
print("total accuracy=",(count_of_total_predricted/count_of_total_class)*100)

