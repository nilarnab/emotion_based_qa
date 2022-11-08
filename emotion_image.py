import cv2
from deepface import DeepFace
import numpy as np  #this will be used later in the process

imgpath = "./winCeleb.jpg"
image = cv2.imread(imgpath)

analyze = DeepFace.analyze(image,actions=['emotions'])  #here the first parameter is the image we want to analyze #the second one there is the action
print(analyze)