import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
 
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

#print(X_train.shape) ->(60000,28,28)

X_train = X_train.reshape(-1,28,28,1).astype('float32')/255.0
X_test = X_test.reshape(-1,28,28,1).astype('float32')/255.0

#reshaping as ut deep learning models in TensorFlow often require 
# 4D input:(batch_size, height, width, channels)

model = keras.Sequential([
    
    keras.layers.Reshape(target_shape= (28,28,1),input_shape =(28,28)),

    keras.layers.Conv2D(32,(3,3),activation='relu'),
    
#  MaxPooling2D layer in Keras is used to downsample feature maps by 
#   selecting the maximum value in a defined window.pool_size=(2,2) → Defines a 2×2 window (kernel) that moves over the input feature map.
#    It picks the maximum value in each 2×2 region.
#    This reduces the spatial size (height and width) by half, improving computational efficiency.'''

    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64,(3,3),activation='relu'),

    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),

    keras.layers.Dense(128,activation='relu'),
#    '''The Dense layer in Keras is a fully connected (FC) layer, meaning each neuron 
#     is connected to every neuron in the previous layer.It performs matrix 
#     multiplication between the input and weight matrix, then adds a bias.'''


    keras.layers.Dense(10,activation='softmax')

])
model.compile(optimizer='adam',loss ='sparse_categorical_crossentropy',
              metrics =['accuracy'])



#ADAM -It combines momentum (helps accelerate learning)
#and RMSprop (adjusts learning rate dynamically).

#sparse_categorical_crossentropy -Use sparse_categorical_crossentropy
#when your labels are integer-encoded (e.g., 0, 1, 2, ..., 9 for MNIST'''





model.fit(X_train,Y_train,epochs=2,validation_data=(X_test,Y_test))
#An epoch is one complete pass through the entire training 
#dataset during model training.



predictions  = model.predict(X_test)

image = 0
plt.imshow(X_test[image],cmap = "binary")
plt.title(f'Predicted:{np.argmax(predictions[image])} ,Actual :{Y_test[image]}')
plt.show()





