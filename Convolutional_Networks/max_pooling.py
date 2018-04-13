from keras.layers import MaxPooling2D

MaxPooling2D(pool_size, strides, padding)


# Checking Dimentionality
from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=2, input_shape=(100, 100, 15)))
model.summary()
# Output shape = (None, 50, 50, 15)
