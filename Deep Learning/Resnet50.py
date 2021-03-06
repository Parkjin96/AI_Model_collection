
# 출처 : 

from keras.applications import ResNet50
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=input, include_top=False, weights=None, pooling='max')
model.summary()
input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=input, include_top=True, weights=None, pooling='max')
model.summary()


input = Input(shape=(224, 224, 3))
model = ResNet50(input_tensor=input, include_top=False, weights=None, pooling='max')
 
x = model.output
x = Dense(1024, name='fully', init='uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(512, init='uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dense(3, activation='softmax', name='softmax')(x)
model = Model(model.input, x)
model.summary()

