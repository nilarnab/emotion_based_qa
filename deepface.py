# from deepface import DeepFace
from cgi import test
from retinaface import RetinaFace
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
from decision_tree import DecisionMaker

def get_gray(images):
    gray_images = []
    for image in images:
        res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_images.append(res)
    return gray_images

# faces = RetinaFace.extract_faces("./winCeleb.jpg")
# faces = RetinaFace.extract_faces("./mixed_mood.jpg")
# faces = RetinaFace.extract_faces("./astronauts.jpeg")
faces = RetinaFace.extract_faces("./lonely_girl.webp")
# faces = RetinaFace.extract_faces("./lonely_girl_2.webp")

print("There are ", len(faces), " faces in the image")



def preprocess(x):
    x = tf.cast(x, tf.float32) / 255.0

    return x

new_faces = []
for face in faces:
    face = cv2.resize(face, (48, 48))
    cv2.imshow("some face", face)
    new_faces.append(face)



gray_faces = get_gray(new_faces)
print("gray faces", len(gray_faces), np.shape(gray_faces[0]))


tests = []
for gray_face in gray_faces:
    img_array = tf.keras.utils.img_to_array(gray_face)
    feedable_data = tf.expand_dims(img_array, 0)
    feedable_data /= 255

    tests.append(feedable_data)

model = load_model("./trained_models/trained_7_95.h5")
model.summary()

classes = ["disgusted", "angry", "fearful", "happy", "neutral", "sad", "surprised"]
emotional_context = ""

moods = {clas: 0 for clas in classes}

for test in tests:
    predictions = model.predict(test)
    print("predictions", predictions)
    
    pred_index = np.argmax(predictions[0])
    pred_value = classes[pred_index]

    moods[pred_value] += 1


decisionMaker = DecisionMaker(moods)
res = decisionMaker.find_emotional_caption()

print(res)

para = '. '.join(res) + '.'

print("\n\n")
print("Generated Prediction\n==========================")
print(para)
print("=====================")
    

