import streamlit as st
import tensorflow
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import time

@st.cache
def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras modelp
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

def call_bar():
    my_bar = st.progress(0)
    suppress_st_warning = True
    for i in range(100):
        time.sleep(0.1)
        my_bar.progress(i+1)
    return my_bar

#Streamlit app starts here
st.markdown("<h2 style='text-align: center; color: red;'>Image Classification with Google's Teachable Machine</h2>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: black;'>Brain Tumor MRI Classification Example</h3>", unsafe_allow_html=True)

st.markdown("<h6 style='text-align: center; color: black;'>Upload a brain MRI Image for image classification as tumor or no-tumor</h6>", unsafe_allow_html=True)

st.sidebar.title('Navigation')

uploaded_file = st.sidebar.file_uploader("", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    call_bar()
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
        
if st.sidebar.button('predict'):
    label = teachable_machine_classification(image, 'keras_model.h5')
    suppress_st_warning = True
    if label == 0:
        st.write("The MRI scan has a brain tumor")
    else:
        st.write("The MRI scan is healthy")

