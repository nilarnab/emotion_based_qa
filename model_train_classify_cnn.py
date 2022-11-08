import imp
import cv2
import tensorflow as tf
import keras
import os
from keras.models import load_model
import numpy as np
from tqdm import tqdm

from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

print("import complete")

filepath = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/train/disgusted"
filepath1 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/train/angry"
filepath2 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/train/fearful"
filepath3 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/train/happy"
filepath4 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/train/neutral"
filepath5 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/train/sad"
filepath6 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/train/surprised"

filepath_test = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/disgusted"
filepath1_test = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/angry"
filepath2_test = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/fearful"
filepath3_test = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/happy"
filepath4_test = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/neutral"
filepath5_test = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/sad"
filepath6_test = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/surprised"


def add_in_image_list(folder_path, label):
    global image_list, label_list

    for image_name in tqdm(os.listdir(folder_path)):
        image_list.append(cv2.imread(os.path.join(folder_path, image_name)))
        label_list.append(int(label))


def add_in_image_list_test(folder_path, label):
    global image_list_test, label_list_test

    for image_name in tqdm(os.listdir(folder_path)):
        image_list_test.append(cv2.imread(os.path.join(folder_path, image_name)))
        label_list_test.append(int(label))


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y

def preprocess_x(x):
      return tf.cast(x, tf.float32) / 255.0

def preprocess_y(x):
      return tf.cast(x, tf.int64)
    

def create_dataset(xs, ys, n_classes=10):
  print("trying for ys", ys)
  ys = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, ys)) \
    .map(preprocess) \
    .shuffle(len(ys)) \
    .batch(128)

def get_gray(images):
  gray_images = []
  for image in images:
    res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_images.append(res)
  return gray_images





if __name__ == '__main__':

    global image_list, label_list
    image_list = []
    label_list = []

    image_list_test = []
    label_list_test = []

    add_in_image_list(filepath, "0")
    add_in_image_list(filepath1, "1")
    add_in_image_list(filepath2, "2")
    add_in_image_list(filepath3, "3")
    add_in_image_list(filepath4, "4")
    add_in_image_list(filepath5, "5")
    add_in_image_list(filepath6, "6")

    add_in_image_list_test(filepath_test, "0")
    add_in_image_list_test(filepath1_test, "1")
    add_in_image_list_test(filepath2_test, "2")
    add_in_image_list_test(filepath3_test, "3")
    add_in_image_list_test(filepath4_test, "4")
    add_in_image_list_test(filepath5_test, "5")
    add_in_image_list_test(filepath6_test, "6")

    print("convertin image to gray")
    print(np.shape(image_list[0]))

    image_list = get_gray(image_list)
    image_list_test = get_gray(image_list_test)

    
    

    print(np.shape(image_list[0]))

    # exit(0)
    

    dataset_train = create_dataset(image_list, label_list, 7)
    dataset_test = create_dataset(image_list_test, label_list_test, 7)

    

    

    # old model 
#     model = tf.keras.Sequential(
#     [
#     tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu",input_shape=(48, 48, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),

#     tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),

#     tf.keras.layers.Flatten(),
#     keras.layers.Dense(units=128, activation='relu'),
#     tf.keras.layers.Dense(100, activation="relu"),
#     tf.keras.layers.Dense(10, activation="softmax"),
#     keras.layers.Dense(units=7, activation='softmax')
# ]
# )

  # new model 
  # Initialising the CNN
    model = Sequential()
    # 1 - Convolution
    model.add(Conv2D(32,(3,3), padding='same', input_shape=(48, 48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 2nd Convolution layer
    model.add(Conv2D(64,(5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 3rd Convolution layer
    model.add(Conv2D(128,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 4th Convolution layer
    model.add(Conv2D(256,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Flattening
    model.add(Flatten())
    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(7, activation='softmax'))
  ####################


    model = load_model("./trained_models/trained_7_94_64.h5")

    model.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(
    dataset_train.repeat(),
    epochs=100, 
    shuffle=True,
    validation_data=dataset_test.repeat(),
    validation_steps=500,
    steps_per_epoch=500
)

    model.save("./trained_models/trained_model_new.h5")