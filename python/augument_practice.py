from keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen = ImageDataGenerator(
    rescale=1./255

)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = 'c:\\_data\\image\\cat_and_dog\\Train\\'
path_test = 'c:\\_data\\image\\cat_and_dog\\Test\\'

xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(120, 120)
    , batch_size= 20000
    , class_mode='binary'
    , color_mode='rgb' # default
    , shuffle='True'
    #Found 20000 images belonging to 2 classes.
)
print('train data ok')

# test = test_datagen.flow_from_directory(
#     path_test
#     , target_size=(120, 120)
#     , batch_size= 20000
#     , class_mode=None
#     , color_mode='rgb' # default
    
# )
# print('submit data ok')

print(xy_train)