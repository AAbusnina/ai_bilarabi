# import the necessary packages
from gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from matplotlib import pyplot as plt
from imutils import paths




model_path = "path_to_you_model"



#initialize the model
model = load_model(model_path)

imagePaths = list(paths.list_images(path_to_data))

for image_path in imagePaths:
    # load the original image from disk (in OpenCV format) and then
    # resize the image to its target dimensions
    orig = cv2.imread(image_path)
    resized = cv2.resize(orig, (224, 224))
    scaled_image = np.array(resized) / 255.0
    image = np.expand_dims(scaled_image, axis=0)


    preds = model.predict(image)
    i = np.argmax(preds[0])



    if i==1:
        label = "Healthy"
    else:
        label = "Infected"


    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # draw the predicted label on the output image
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    output = np.vstack([orig, heatmap, output])
    output = imutils.resize(output, height=700)


    #outputt = cv2.resize(outputt, (1989, 1482))



    # display the original image and resulting heatmap and output image
    # to our screen
    #output = np.vstack([orig, heatmap, output])
    #output = imutils.resize(output, height=700)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
