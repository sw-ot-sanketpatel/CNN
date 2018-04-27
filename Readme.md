
# Images-Multi Class Classification Using Keras-tensorflow(CNN)


```python
import keras # set keras backend to tesnorflow
```

    C:\Users\sanket.patel\Anaconda3\envs\py3.6\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


```python
import os
import pandas as pd
import numpy as np
import scipy.misc
from skimage import io
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras.models import Sequential
from keras import backend as K
from sklearn.model_selection import train_test_split
```

### **1. Read images from folders and give folder path**


```python
black_grass=os.listdir("plant_seed_ds/train/Black-grass")
charlock=os.listdir("plant_seed_ds/train/Charlock")
cleavers=os.listdir("plant_seed_ds/train/Cleavers")

filepath0="plant_seed_ds/train/Black-grass/"
filepath1="plant_seed_ds/train/Charlock/"
filepath2="plant_seed_ds/train/Cleavers/"
```

### **2. Store images in list and assign label according to that**


```python
images=[]
label=[]

for i in black_grass:
    image = scipy.misc.imread(filepath0+i)
    images.append(image)
    label.append(0) #for black_grass images
    
for i in charlock:
    image = scipy.misc.imread(filepath1+i)
    images.append(image)
    label.append(1) #for charlock images
    
for i in cleavers:
    image = scipy.misc.imread(filepath2+i)
    images.append(image)
    label.append(2) #for cleavers images
```

    C:\Users\sanket.patel\Anaconda3\envs\py3.6\lib\site-packages\ipykernel_launcher.py:5: DeprecationWarning: `imread` is deprecated!
    `imread` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    Use ``imageio.imread`` instead.
      """
    C:\Users\sanket.patel\Anaconda3\envs\py3.6\lib\site-packages\ipykernel_launcher.py:10: DeprecationWarning: `imread` is deprecated!
    `imread` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    Use ``imageio.imread`` instead.
      # Remove the CWD from sys.path while we load stuff.
    C:\Users\sanket.patel\Anaconda3\envs\py3.6\lib\site-packages\ipykernel_launcher.py:15: DeprecationWarning: `imread` is deprecated!
    `imread` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    Use ``imageio.imread`` instead.
      from ipykernel import kernelapp as app
    


```python
for i in range(0,len(images)):# resize each image in 100*100 pixels
    images[i]=cv2.resize(images[i],(100,100))
```


```python
X_train,X_val,Y_train,Y_val=train_test_split(images,label,test_size=0.2,random_state=1) #split data set to train and validation sets

```


```python
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16

print(nb_train_samples)
```

    158
    

### ** 3. Formulate Model **


* Number of filters=32
* stride=(3,3)
* input shape=(100,100,3) 3 is for color channel which is RGB if grey image then 1 insted of 3
* Max Pool window size= (2,2)
* Activation function = relu which is right now mostly used in deep learning
* last activation function= as we have multiclass classification we used softmax/ if two class or binary use sigmoid/tanh
* dropout =0.5
* dense : number of classes
* loss: if predicted class is two then it is binary_crossentropy / if more than two classes categorical_crossentropy






```python
img_width=100
img_height=100

model = Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3))
model.add(layers.Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 98, 98, 32)        896       
    _________________________________________________________________
    activation_1 (Activation)    (None, 98, 98, 32)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 49, 49, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 47, 47, 32)        9248      
    _________________________________________________________________
    activation_2 (Activation)    (None, 47, 47, 32)        0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 23, 23, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 21, 21, 64)        18496     
    _________________________________________________________________
    activation_3 (Activation)    (None, 21, 21, 64)        0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 10, 10, 64)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 6400)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                409664    
    _________________________________________________________________
    activation_4 (Activation)    (None, 64)                0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 3)                 195       
    _________________________________________________________________
    activation_5 (Activation)    (None, 3)                 0         
    =================================================================
    Total params: 438,499
    Trainable params: 438,499
    Non-trainable params: 0
    _________________________________________________________________
    

### 4. Image Operation


```python
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
```


```python
train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)
```

### 5. fit cnn model to train data with epochs=30 generally epochs not more than 50


```python
history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)
```

### 6. Save Final Model Weights


```python
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')
```

### 7. Test Data prepration


```python
test=os.listdir("plant_seed_ds/test")
print(test)
filepath_test="plant_seed_ds/test/"

X_test=[]
Y_test=[]

for i in test:
    image = scipy.misc.imread(filepath_test+i)
    X_test.append(image)
```

    ['1ec9ab1b8.png', '5283b8c96.png', '9959fb099.png', '9b941ac1b.png']
    

    C:\Users\sanket.patel\Anaconda3\envs\py3.6\lib\site-packages\ipykernel_launcher.py:9: DeprecationWarning: `imread` is deprecated!
    `imread` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    Use ``imageio.imread`` instead.
      if __name__ == '__main__':
    


```python
for i in range(0,len(X_test)):
    X_test[i]=cv2.resize(X_test[i],(100,100))
```


```python
test_datagen = ImageDataGenerator(rescale=1. / 255)
```

### 8. Test Images class prediction higher probability depicts images belongs to that class


```python
test_generator = val_datagen.flow(np.array(X_test), batch_size=batch_size)
prediction_probabilities = model.predict_generator(test_generator, verbose=1)
prediction_probabilities
```

    1/1 [==============================] - 0s 114ms/step
    




    array([[0.289565  , 0.21341552, 0.49701947],
           [0.03481712, 0.5608863 , 0.40429658],
           [0.00619659, 0.77305776, 0.2207457 ],
           [0.03949131, 0.8740396 , 0.08646912]], dtype=float32)


