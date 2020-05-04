import tensorflow as tf
import pandas as  pd
import numpy as np 
import matplotlib.pyplot as plt
from  matplotlib.image import imread
import os,sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping


tr_path = os.getcwd()+'/'+'intel-image-classification/seg_train/seg_train/'
tst_path = os.getcwd()+'/'+'intel-image-classification/seg_test/seg_test/'
img_shape = (150,150,3)

imgdata  = ImageDataGenerator(rotation_range = 20,width_shift_range = 0.1,height_shift_range=0.1,
                                            shear_range=0.1,zoom_range=0.1,horizontal_flip=True)

model  = Sequential()
model.add(Conv2D(filters = 32,kernel_size = (3,3),input_shape = img_shape,activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(filters = 64,kernel_size = (3,3),input_shape = img_shape,activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(filters = 64,kernel_size = (3,3),input_shape = img_shape,activation ='relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6,activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#(model.summary())

earlystop =  EarlyStopping(monitor = 'val_loss',patience = 5)

batch_size = 32

tr_img_gen =  imgdata.flow_from_directory(tr_path,target_size =img_shape[:2],color_mode = 'rgb',
                            batch_size = batch_size,class_mode = 'categorical' )

tst_img_gen =  imgdata.flow_from_directory(tst_path,target_size =img_shape[:2],color_mode = 'rgb',
batch_size = batch_size,class_mode = 'categorical', shuffle = False)

print(tr_img_gen.class_indices)

result = model.fit_generator(tr_img_gen, epochs = 50,validation_data =tst_img_gen,callbacks = [earlystop])

metrics = pd.DataFrame(model.history.history)
#plot the loss vs accuracy
metrics[['accuracy','val_accuracy']].plot()
#evaluate model with 
model.evaluate('xtest,ytest')
#predict classes
model.predict_classes('xtest')