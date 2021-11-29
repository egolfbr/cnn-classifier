import keras
import numpy as np
import cv2
import ssl
import chirp
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.applications import resnet50 
from sklearn.model_selection import train_test_split


ssl._create_default_https_context = ssl._create_unverified_context

#preprocess
def load_preprocess_gray(path):
    x_data = []
    y_data = []
    l = 10000
    chirp.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in range (1,10001):
        lin_img = cv2.imread(path + "lin_" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        geo_img = cv2.imread(path + "geo_" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        sin_img = cv2.imread(path + "sin_" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        
        chirp.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        lin_img_rs = cv2.resize(lin_img,(128,128))
        geo_img_rs = cv2.resize(geo_img,(128,128))
        sin_img_rs = cv2.resize(sin_img,(128,128))
        

        x_data.append(lin_img_rs)
        x_data.append(geo_img_rs)
        x_data.append(sin_img_rs)


        # Images are read in a specific order so we can automatically label the data in that order
        # 0 = lin 
        # 1 = geo 
        # 2 = sin
        y_data.append(0)
        y_data.append(1)
        y_data.append(2)
    y_data = np.asarray(y_data)
    x_data = np.asarray(x_data)
    print("Initial Shape: ", x_data.shape)
    x_data = x_data / 255.0
    x_data = np.reshape(x_data, (30000,128,128,1))
    print("Reshaped Shape: ", x_data.shape)
    return x_data, y_data

#preprocess
def load_preprocess_color(path):
    x_data = []
    y_data = []
    l = 10000
    chirp.printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in range (1,10001):
        lin_img = cv2.imread(path + "lin_" + str(i) + ".png")
        geo_img = cv2.imread(path + "geo_" + str(i) + ".png")
        sin_img = cv2.imread(path + "sin_" + str(i) + ".png")

        chirp.printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        lin_img_rs = cv2.resize(lin_img,(128,128))
        geo_img_rs = cv2.resize(geo_img,(128,128))
        sin_img_rs = cv2.resize(sin_img,(128,128))
        

        x_data.append(lin_img_rs)
        x_data.append(geo_img_rs)
        x_data.append(sin_img_rs)

        # Images are read in a specific order so we can automatically label the data in that order
        # 0 = lin 
        # 1 = geo 
        # 2 = sin
        y_data.append(0)
        y_data.append(1)
        y_data.append(2)
    y_data = np.asarray(y_data)
    x_data = np.asarray(x_data)
    print("Initial Shape: ", x_data.shape)
    x_data = x_data / 255.0
    return x_data, y_data




def gen_custom_model():

    model = Sequential()
    #add model layers
    model.add(Conv2D(30, kernel_size=7, activation='relu', input_shape=(128,128,1),padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Conv2D(60, kernel_size=3, activation='relu',padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Flatten())
    model.add(Dense(200,activation='relu'))
    model.add(Dense(3,activation='softmax'))    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

def gen_resnet():
    model = ResNet50(input_shape=(128,128,3), include_top=False, classes=3)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model

def main():
    # path to img directory
    path = "./imgs/"

    # load and preprocess (normalize) the images
    x_data, y_data = load_preprocess_gray(path)

    print("Images loaded!")
    # generate the model to use
    print("Generating model... ...")
    model = gen_custom_model()
    print("Model Generated!")

    # split into train and testing groups 
    print("Splitting data... ...")
    x_train, x_test, y_train, y_test = train_test_split(x_data,y_data)
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 3)
    y_test = keras.utils.to_categorical(y_test, 3)
    # train the model 
    print("==============Begin Model Training==================")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
    model.evaluate()

main()







    

