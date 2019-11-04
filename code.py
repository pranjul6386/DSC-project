if smoke==1 and flame==1:
  !unzip image_data1.zip
  
  !pip install --upgrade Keras
  !pip3 list | grep -i keras
  !python -c "import keras"
  
  from tensorflow.keras.models import Sequential
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  
  
  from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Convolution2D, MaxPooling2D
  classifier=Sequential()
 
  classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = "relu"))
  classifier.add(MaxPooling2D(pool_size=(2,2)))
  
  classifier.add(Convolution2D(32,3,3,activation='relu'))
  classifier.add(MaxPooling2D(pool_size=(2,2)))
  
  classifier.add(Flatten())
  classifier.add(Dense(128,activation='relu'))
  classifier.add(Dense(1,activation='sigmoid'))
  
  
  classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
  
  from keras.preprocessing.image import ImageDataGenerator
  train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
  test_datagen=ImageDataGenerator(rescale=1./255)
  training_set=train_datagen.flow_from_directory('image_data/training_set',target_size=(64,64),batch_size=32,class_mode='binary')
  test_set=test_datagen.flow_from_directory('image_data/test_set',target_size=(64,64),batch_size=32,class_mode='binary')
  classifier.fit_generator(training_set,steps_per_epoch=600,epochs=25,validation_data=test_set,validation_steps=94)
  
  import numpy as np
  from keras.preprocessing.image import load_img
  from keras.preprocessing.image import img_to_array
  from keras.preprocessing.image import array_to_img
  # loading the image
  test_image = load_img('random3.jpg',target_size=(64,64))
   
  test_image = img_to_array(test_image)
  print(test_image.dtype)
  print(test_image.shape)
  
  test_image=np.expand_dims(test_image,axis=0)
  result=classifier.predict(test_image)
  training_set.class_indices
  if result[0][0] >=0.7:
    prediction='fire'
  else:
    prediction='safe'
  print(prediction)


!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

downloaded = drive.CreateFile({'id':'1-jsrUfAt06Gk94EQrid1qNJq474xlPBW'}) # replace the id with id of file you want to access
downloaded.GetContentFile('data1.csv')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("data1.csv",error_bad_lines=False)
x=dataset.iloc[:,:3].values
y=dataset.iloc[:,3].values

dataset.head(5)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x[:, 0] = labelencoder.fit_transform(x[:, 0])

print(x)
print(y)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

print(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm4=confusion_matrix(y_test, y_pred)

print(cm4)

y_prob=classifier.predict_proba(x_test)

