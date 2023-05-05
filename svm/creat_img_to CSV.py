import numpy as np
import cv2
import os
from skimage.feature import hog
from skimage.color import rgb2grey
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
IMG_DIR = 'train'
def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    return flat_features
features_list = []
class_list=[]    
for img in os.listdir(IMG_DIR):
        # img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)
        img_array = cv2.imread(os.path.join(IMG_DIR,img))

        img_array=resize(img_array,(150,150,3))

        img_array = (img_array.flatten())

        feat  = img_array
        # feat=create_features(img_array)
        features_list.append(feat)
        print(img[0],feat)
        if(img[0]=='l'):
            class_list.append(0)
        elif(img[0]=='R'):
            class_list.append(1)
        elif(img[0]=='t'):
            class_list.append(2) 
        # with open('output.csv', 'ab') as f:
        #     f.write(img) 
        #     np.savetxt(f,img_array, delimiter=",")
# feature_matrix = np.array(features_list)
class_matrix = np.array(class_list)
print(class_list)
# feature_matrix=feature_matrix.reshape(-1, 1).T
# class_matrix=class_matrix.reshape(-1, 1).T
print(class_matrix.shape)
print(features_list)

X_train, X_test, y_train, y_test = train_test_split(features_list,class_matrix,test_size=.3)
# define support vector classifier
svm = SVC(kernel='linear', probability=True, random_state=42)

# fit model
svm.fit(X_train, y_train)
# generate predictions
y_pred = svm.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(svm, open(filename, 'wb'))