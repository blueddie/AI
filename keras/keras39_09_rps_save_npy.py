import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import datetime


path_train = 'C:\\_data\\image\\rps\\'

train_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(150, 150)
    , batch_size=32
    , class_mode='categorical'
    , shuffle=True
    # Found 2520 images belonging to 3 classes.
)

x = []
y = []

for i in range(len(xy_train)) :
    images, labels = xy_train.next()
    x.append(images)
    y.append(labels)
    
x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)

date = datetime.datetime.now().strftime("%m%d_%H%M")

np_path = 'c:\\_data\\_save_npy\\'
np.save(np_path + 'keras39_' + date +'09_rps_x_np.npy', arr=x)
np.save(np_path + 'keras39_'+ date + '09_rps_y_np.npy', arr=y)

