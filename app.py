import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import numpy as np
import os
import h5py

st.header("Plant Species Detection")


def main():
    file_uploaded=st.file_uploader("Upload Your Image", type=['jpg','png','jpeg'])
    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        figure=plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result=predict_class(image)
        st.write(result)
        print(result)
        st.pyplot(figure)

def predict_class(image):
    # Classifier_model=tf.keras.models.load_model(r"plantspecies.h5")
    # model=tf.keras.models.load_model(r"plantspecies_detection.h5")
    model= tf.keras.models.load_model("models/1")
    shape=((256,256,3))
    # model=tf.keras.Sequential([hub.kerasLayer(Classifier_model,input_shape=shape)])
    test_image=image.resize((256,256))
    test_image=preprocessing.image.img_to_array(test_image)
    # test_image=test_image/255.0
    test_image=np.expand_dims(test_image,axis=0)
    CLASS_NAMES = ['Amaltas', 'False ashoka', 'Mauritious hemph', 'bougainvillea glabra', 'daisy', 'dandelion', 'rose', 'sunflower', 'thuja']
    predictions=model.predict(test_image)
    print(predictions)
    # scores=tf.nn.softmax(predictions[0])
    # scores=scores.numpy()
    # image_class=CLASS_NAMES[np.argmax(scores)]
    print(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    # result="The Image Uploded is : {}".format(image_class)
    result=predicted_class
    return result

if __name__ == '__main__':
    main()