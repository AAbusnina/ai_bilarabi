import pandas as pd
import streamlit as st
import numpy as np
import os
from keras.layers.core import Dense
from keras.models import Model, load_model
import glob
import pickle
from keras.layers import concatenate
import cv2
from models import create_mlp, create_cnn
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from io import StringIO, BytesIO

@st.cache(hash_funcs={StringIO: StringIO.getvalue})
def load_house_attributes(inputPath):
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()
    for (zipcode, count) in zip(zipcodes, counts):
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
    return df


@st.cache
def process_house_attributes(df, train, test):
   continuous = ["bedrooms", "bathrooms", "area"]
   cs = MinMaxScaler()
   trainContinuous = cs.fit_transform(train[continuous])
   testContinuous = cs.transform(test[continuous])
   zipBinarizer = LabelBinarizer().fit(df["zipcode"])
   trainCategorical = zipBinarizer.transform(train["zipcode"])
   testCategorical = zipBinarizer.transform(test["zipcode"])
   trainX = np.hstack([trainCategorical, trainContinuous])
   testX = np.hstack([testCategorical, testContinuous])
   return (trainX, testX, zipBinarizer, cs)



#********************************** CACHED FUNCTIONS ***************************
@st.cache
def load_it():
    model = load_model("/Users/abusnina/Documents/Training/Streamlit/house_prices_demo/my_model.h5")
    labelizer = pickle.load(open("/Users/abusnina/Documents/Training/Streamlit/house_prices_demo/labelizer.pkl"  , 'rb'))
    cs = pickle.load(open("/Users/abusnina/Documents/Training/Streamlit/house_prices_demo/cs.pkl"  , 'rb'))
    out_scaler = pickle.load(open("/Users/abusnina/Documents/Training/Streamlit/house_prices_demo/out_scaler.pkl"  , 'rb'))
    return (model, labelizer, cs, out_scaler)



@st.cache
def train_model(struct_list, opt, no_epochs, test_size, data, images ):

    split = train_test_split(data, images, test_size=test_size, random_state=42)
    (trainAttrX, testAttrX, trainImagesX, testImagesX) = split
    out_scaler = MinMaxScaler()
    trainY = out_scaler.fit_transform(trainAttrX["price"].values.reshape(-1, 1))
    testY = out_scaler.fit_transform(testAttrX["price"].values.reshape(-1, 1))
    trainAttrX, testAttrX, labelizer, cs = process_house_attributes(data, trainAttrX, testAttrX)

    mlp =  create_mlp(trainAttrX.shape[1], regress=False)
    cnn =  create_cnn(64, 64, 3, struct_list, regress=False)

    combinedInput = concatenate([mlp.output, cnn.output])
    x = Dense(4, activation="relu", name="fc_0")(combinedInput)
    x = Dense(1, activation="linear", name="fc_1")(x)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt, metrics=['accuracy'])
    history_callback = model.fit( [trainAttrX, trainImagesX], trainY,
        	         validation_data=([testAttrX, testImagesX], testY),
        	         epochs=no_epochs, batch_size=25)
    hist_df = pd.DataFrame(history_callback.history)
    return (model, hist_df, labelizer, cs, out_scaler)


@st.cache
def load_house_images(df, inputPath):
    images = []
    paths = []
	# loop over the indexes of the houses
    for i in df.index.values:
		# find the four images for the house and sort the file paths,
		# ensuring the four are always in the *same order*
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))
        paths.append(housePaths)
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")
        for housePath in housePaths:
            image = cv2.imread(housePath)
            image = cv2.resize(image, (32, 32))
            inputImages.append(image)
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]
        images.append(outputImage)
    return (np.array(images), paths)




@st.cache(hash_funcs={BytesIO: BytesIO.getvalue})
def montage_image(img_list):
    images = []
    outputImage = np.zeros((64, 64, 3), dtype="uint8")
    inputImages = []
    for img in img_list:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image_small = cv2.resize(opencv_image, (32, 32))
        inputImages.append(opencv_image_small)
    outputImage[0:32, 0:32] = inputImages[0]
    outputImage[0:32, 32:64] = inputImages[1]
    outputImage[32:64, 32:64] = inputImages[2]
    outputImage[32:64, 0:32] = inputImages[3]
    images.append(outputImage)

    return (np.array(images) , np.array(opencv_image) )
