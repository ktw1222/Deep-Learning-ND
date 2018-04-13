# from keras.models import Sequential
# from keras.layers import Conv2D
#
# # create a CNN in Keras by first creating a Sequential model.
# model = Sequential()
# # add layers to the network by using the .add() method
# model.add(Conv2D(filters=16, kernel_size=2, strides=2, padding='valid',
#     activation='relu', input_shape=(200, 200, 1)))
# model.summary()



# Example 2
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same',
    activation='relu', input_shape=(128, 128, 3)))
model.summary()

# Width of convolutional layer = 64
# Depth of convolutional layer = 32 (# of filters)
# Number of parameter in convolutional layer = 896 (32*3*3*3 + 32)
# Output shape = (none, 64, 64, 32)
