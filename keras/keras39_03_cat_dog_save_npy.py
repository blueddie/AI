from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time

train_datagen = ImageDataGenerator(
    rescale=1./255

)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = 'c:\\_data\\image\\cat_and_dog\\Train\\'
path_test = 'c:\\_data\\image\\cat_and_dog\\Test\\'

st = time.time()


Xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(300, 300)
    , batch_size= 32
    , class_mode='binary'
    , color_mode='rgb' # default
    , shuffle='True'
    #Found 20000 images belonging to 2 classes.
)
print('train data ok')

test = test_datagen.flow_from_directory(
    path_test
    , target_size=(300, 300)
    , batch_size= 32
    , class_mode=None
    , color_mode='rgb' # default
    
)
print('submit data ok')


X = []
y = []

for i in range(len(Xy_train)):
    images, labels = Xy_train.next()
    X.append(images)
    y.append(labels)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

submit = []
for i in range(len(test)):
    images = test.next()
    submit.append(images)

submit = np.concatenate(submit, axis=0)

print(submit.shape)

# print(X.shape)  # (19997, 120, 120, 3)
# print(y.shape)  # (19997,)

np_path = 'c:\\_data\\_save_npy\\'
np.save(np_path + 'keras39_3_cat_dog_x_np.npy', arr=X)
np.save(np_path + 'keras39_3_cat_dog_y_np.npy', arr=y)
np.save(np_path + 'kaggle_cat_dog_submission_np.npy', arr=submit)

et = time.time()

print(f"걸린 시간 : {(et - st)} 초")