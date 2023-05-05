# Python program to create 
# Image Classifier using CNN 

# Importing the required libraries 
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 

'''Setting up the env'''
TRAIN_DIR ='train'
IMG_SIZE =100
LR = 1e-3#0.001


#'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'selfdrivng-{}-{}.model'.format(LR, '6conv-basic') 

'''Labelling the dataset'''
def label_img(img): 
	word_label = img.split('.')[-2]
	print('label',word_label)
	# DIY One hot encoder 
	if word_label[0] == 'l': return   [1,0,0]
	elif word_label[0] == 'R': return   [0,1,0]
	elif word_label[0] == 't': return   [0,0,1]
	
	
	
'''Creating the training data'''
def create_train_data(): 
	# Creating an empty list where we should store the training data 
	# after a little preprocessing of the data 
	training_data = [] 
        
	# tqdm is only used for interactive loading 
	# loading the training data
	count=0
	for img in tqdm(os.listdir(TRAIN_DIR)):
		#count=count+1
		label = label_img(img)
		print(label)
		path = os.path.join(TRAIN_DIR, img) 
		# loading the image from the path and then converting them into 
		# greyscale for easier covnet prob 
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
		# resizing the image for processing them in the covnet 
		img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
		# final step-forming the training data list with numpy array of the images 
		training_data.append([np.array(img), np.array(label)]) 
		
	# shuffling of the training data to preserve the random state of our data 
	shuffle(training_data) 

	# saving our trained data for further uses if required 
	#np.save('train_data.npy', training_data) 
	return training_data 

'''Processing the given test data'''
# Almost same as processing the training data but 
# we dont have to label it. 

train_data = create_train_data() 
print(train_data[0][0][0])
print(train_data[0][1])

'''Creating the neural network using tensorflow'''
# Importing the required libraries 
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
import tensorflow as tf 

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

model = tflearn.DNN(convnet, tensorboard_dir ='log') 

# Splitting the testing data and training data 
train = train_data 


'''Setting up the features and lables'''
# X-Features & Y-Labels 

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
Y = [i[1] for i in train]
model.fit({'input': X}, {'targets': Y}, n_epoch =70,
         
           snapshot_step =500, show_metric = True, run_id = MODEL_NAME) 
model.save('self driving .model')
