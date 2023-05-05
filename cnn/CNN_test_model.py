# Importing the required libraries 
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
'''Setting up the env'''
TEST_DIR ='test'
IMG_SIZE =100
LR = 1e-3#0.001

count = 0
images = []     # LIST CONTAINING ALL THE IMAGES 
classNo = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS ID OF IMAGES
predicted_class=[] 
numOfSamples= []
actual_class=[0,0,0]
object_count=0
#'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'Kidneyst-{}-{}.model'.format(LR, '6conv-basic') 

'''Labelling the dataset'''
def label_img(img): 
	word_label = img.split('.')[-2]
	print('label',word_label)
	# DIY One hot encoder 
	if word_label[0] == 'l': return 0
	elif word_label[0] == 'R': return 1
	elif word_label[0] == 't': return 2  
class_0=[]
class_1=[]
class_2=[]
for img in tqdm(os.listdir(TEST_DIR)):
    label = label_img(img)
    if label==0:
            class_0.append(label)
    elif label==1:
            class_1.append(label)	
    else:
            class_2.append(label)
    
actual_class=[class_0,class_1,class_2]
numOfSamples=[len(class_0),len(class_1),len(class_2)]


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

convnet = fully_connected(convnet, 3, activation ='softmax')
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
	loss ='categorical_crossentropy', name ='targets') 
model = tflearn.DNN(convnet, tensorboard_dir ='log') 
model.load('self driving .model')

def cNN_test(image):
    frame=cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newimg = cv2.resize(gray,(int(IMG_SIZE),int(IMG_SIZE)))
    data = newimg.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([ data])[0]
    model_out=list(model_out)
    print(model_out)
    val=max(model_out)
    idx=model_out.index(max(model_out))
    return idx
pred_class_0=[]
pred_class_1=[]
pred_class_2=[]    
for img in tqdm(os.listdir(TEST_DIR)):
    word_label = img.split('.')[-2]
    path = os.path.join(TEST_DIR, img) 
    if word_label[0] == 'l': 
        pred_class_0.append(cNN_test(path))
    elif word_label[0] == 'R':
        pred_class_1.append(cNN_test(path))
    elif word_label[0] == 't':
        pred_class_2.append(cNN_test(path))
predicted_class=[pred_class_0,pred_class_1,pred_class_2]

print("Actual Class",actual_class)
print("predicted Class",predicted_class) 
actual_score=0
score=0
class_wise_score=[]
for x in range(0,3):
    class_wise_score.append(predicted_class[x].count(x))

acc=sum(class_wise_score)/sum(numOfSamples)*100


print('sample per class',numOfSamples)
print('class wise score',class_wise_score)
print("total Accuracy of testing=",acc)


##plt.show()
