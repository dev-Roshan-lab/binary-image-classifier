#imports 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
from imutils import paths
import joblib
import urllib.request

#so called brain where we extract histogram from the picture 
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
    [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
   
#the folder dataset contains all the images used for training the images are name in a specific fashion for ease of use
imagePaths = list(paths.list_images('dataset'))

#create two list where the numpy array is stored and to be converted as binary files
data = []
labels = []

#here run to all the images in dataset folder. 
#the naming conention is (class).(a number).jpg for example cat.1.jpg 
for (i, imagePaths) in enumerate(imagePaths):
    image = cv2.imread(imagePaths)
    label = imagePaths.split(os.path.sep)[-1].split(".")[0] #now we OS to split the directory to image file to get the class name for example if "E:/dataset/cat.1.jpg" is the location of the image on running through this piece of code it returns "cat" so easy to classify
    
    hist = extract_color_histogram(image)
    data.append(hist)
    labels.append(label)
 
 #encoding the labels for ease and less memory usage if deploying on cloud
le = LabelEncoder()
labels = le.fit_transform(labels)
 
 #creating our machine Learning Model
 #As SVM Calssifiers are good at classifying we use the same
(trainData, testData, trainLabels, testLabels) = train_test_split(
np.array(data), labels, test_size=0.25, random_state=42)
    
model = LinearSVC()
model.fit(trainData, trainLabels)

predictions = model.predict(testData)
#print(classification_report(testLabels, predictions,
# target_names=le.classes_))

#save the model created with joblib to classify it easily basically it again uses less memory in cloud
joblib.dump(model, 'model.sav')

model = joblib.load('model.sav')

singleImage = cv2.imread(image file)
hist_pic = extract_color_histogram(img)
hist_reshape = hist_pic.reshape(1, -1)
prediction = model.predict(hist_reshape)
if prediction == [0]:
  print("Dog")
elif prediction == [1]:
  print("Cat")

