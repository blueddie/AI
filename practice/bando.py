import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255

)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = 'C:\\_data\\image\\bando\\train\\'
path_test = 'C:\\_data\\image\\bando\\test\\'

xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(300,300)
    , batch_size=20
    , class_mode='binary'
    , color_mode='rgb'
    , shuffle='True'
    # Found 213 images belonging to 1 classes.
)
test = test_datagen.flow_from_directory(
    path_test
    , target_size=(300,300)
    , batch_size=20
    , class_mode=None
    , color_mode='rgb' # default
)

x = []
y = []

for i in range(len(xy_train)):
    images, labels = xy_train.next()
    x.append(images)
    y.append(labels)

x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)

submit = []
for i in range(len(test)):
    images = test.next()
    submit.append(images)
   
submit = np.concatenate(submit, axis=0)

np_path = 'c:\\_data\\_save_npy\\'
np.save(np_path + 'x_bando.npy', arr=x)
np.save(np_path + 'y_bando,npy', arr=y)
np.save(np_path + 'submit_bando.npy', arr=submit)

   