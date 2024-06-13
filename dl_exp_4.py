from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.preprocessing.image import ImageDataGenerator


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory(r"/content/drive/MyDrive/trainset",
target_size=(64,64),batch_size=32,class_mode="categorical")
x_test=test_datagen.flow_from_directory(r"/content/drive/MyDrive/testset",
target_size=(64,64),batch_size=32,class_mode="categorical")

print(x_train.class_indices)

model=Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=5,activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(x_train,steps_per_epoch=47,epochs=10,validation_data=x_test,validation_steps=20)

model.save("animal.h5")

from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

model=load_model("animal.h5")

img=image.load_img(r"/content/drive/MyDrive/testset/crows/2Q__ (5).jpeg")

img

x=image.img_to_array(img)

x

x.shape


x=np.expand_dims(x,axis=0)

x.shape

y=model.predict(x)
pred=np.argmax(y, axis=1)

y

pred

x_train.class_indices

index=['bears', 'crows', 'elephants', 'racoons', 'rats']
result=str(index[pred[0]])

result

