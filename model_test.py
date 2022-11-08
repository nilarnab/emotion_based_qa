import cv2
import tensorflow as tf
import keras
from keras.models import load_model
import os
import numpy as np
from tqdm import tqdm
import sys
import math

def add_in_image_list(folder_path, label):
    global image_list, label_list

    for image_name in tqdm(os.listdir(folder_path)):
        image_list.append(cv2.imread(os.path.join(folder_path, image_name)))
        label_list.append(int(label))


filepath = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/disgusted"
filepath1 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/angry"
filepath2 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/fearful"
filepath3 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/happy"
filepath4 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/neutral"
filepath5 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/sad"
filepath6 = r"/home/nilarnab/Documents/emotion_based_qa/dataset/archive/test/surprised"

def get_gray(image):
    return cv2.cvtColor(cv2.resize(image, (48, 48)), cv2.COLOR_BGR2GRAY)

image_list = []
label_list = []

add_in_image_list(filepath, "0")
add_in_image_list(filepath1, "1")
add_in_image_list(filepath2, "2")
add_in_image_list(filepath3, "3")
add_in_image_list(filepath4, "4")
add_in_image_list(filepath5, "5")
add_in_image_list(filepath6, "6")

model = load_model("./trained_models/trained_7_97_50.h5")
model.summary()

correct = 0
ind = 0
for image in image_list:
    tests = []
    gray_face = get_gray(image)
    img_array = tf.keras.utils.img_to_array(gray_face)
    feedable_data = tf.expand_dims(img_array, 0)
    feedable_data /= 255

    predictions = model.predict(feedable_data)
    # print("predictions", predictions)
    
    pred_index = np.argmax(predictions[0])
    print(pred_index)

    if int(pred_index) == int(label_list[ind]):
        correct += 1


    # print("accuracy: ", correct/ (ind + 1))
    # sys.stdout.write("\r{0}>".format("="* math.floor(20 * ind/len(image_list))))
    sys.stdout.write("\raccuracy " + str(correct/ (ind + 1)))
    sys.stdout.flush()

    ind += 1
    
