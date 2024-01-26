import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import datetime
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import datetime

# target 300 300 3

path_train = 'C:\\_data\\image\\horse_human\\'

train_datagen = ImageDataGenerator(
    rescale=1./255
    
)

xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(300, 300)
    , batch_size=32
    , class_mode='binary'
    , shuffle=True
    # Found 1027 images belonging to 2 classes.
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
np.save(np_path + 'keras39_' + date +'07_horse_2진_x_np.npy', arr=x)
np.save(np_path + 'keras39_'+ date + '07_horse_2진_y_np.npy', arr=y)