import streamlit as st
from visualization import *
from data import *
from PIL import Image
from tflite_loader import *
from model_loader import *
import cv2
import segmentation_models as sm
import tensorflow.keras.models
from mlruns import mlflow_server
from time import time
import logging
import psutil as ps
from resource_manager import Resource_Manager
import os


logger = logging.getLogger("Streamlit_Server")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('../logs/runtime.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# set framework for segmentation_models
sm.set_framework('tf.keras')
# print(sm.framework())


st.markdown('''
            # Lung Segmentation App
            Select a model, then upload your **CXR** image and 
            choose a pre-process for your input image
            then hit the `submit` button to get your **segmented** mask.    
            For **VAE** just select to numbers between `1` and `30`, 
            hit the `submit` button and get your **generated** image.
            
            ------------------------------------------------------
            ''')

# """
    # load models in memory
    # H5 models
    # heavier models
# """
model_unet = load_model("U-Net", pretrained=True)
model_runet = load_model("ResidualU", pretrained=True)
model_eff = tensorflow.keras.models.load_model('../weights/EFF_UNET.h5', compile=False)
model_vgg = tensorflow.keras.models.load_model('../weights/VGG_UNET.h5', compile=False)
model_decoder = load_model("Autoencoder (VAE)", pretrained=True)

# """
#     tflite models
#     low size models
# """
model_unet_lt = TFLiteModel("../weights/cxr_unet.tflite")
model_runet_lt = TFLiteModel("../weights/cxr_resunet.tflite")

# monitor CPU and memory usage
resources = Resource_Manager()
resources = resources.monitor()

# """
#     Code complete checker
#     :200: mask detected successfuly
#     :201: images generated successfuly
#     :400: Code not completed yet.
# """
response_code = 400

# """
#     sidebar submit button
#     select model to detect lung or to generate images
#     also check each nodel results by select MLFlow
# """
with st.form(key='segmentation'):
    with st.sidebar:
        model_name = st.sidebar.selectbox(
            'Select model',
            [None, "U-Net", "Residual U-Net", "Efficientnet-U-Net", "VGG16-U-Net", "Autoencoder (VAE)", "MLFlow"])

        if model_name == "U-Net":
            # """
            #     take image in shape (1, 256, 256, 1)
            #     return a mask in shape (1, 256, 256, 1)
            # """
            st.write("A `U-Net` model.\n\n**Accuracy**: `98%`")
            model_type = st.sidebar.radio("model_type", ["H5", "tflite"])
            
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])
                if file is not None:
                    processed_image = stream_data(file, pre_process=pre_process)
                    image = load_data(Image.open(file), pre_process)
                    image = image.reshape(1, 256, 256, 1)

                    st.image(processed_image, use_column_width=True)
                    submit_button = st.form_submit_button(label='Submit')

                    if submit_button:
                        if model_type == "H5":
                            ts = time()
                            mask = predict(model_unet, image)
                            resources = Resource_Manager()
                            resources = resources.monitor()
                            te = time()
                            te -= ts
                        else:
                            ts = time()
                            mask = model_unet_lt.predict(image)
                            resources = Resource_Manager()
                            resources = resources.monitor()
                            te = time()
                            te -= ts

                        _ = visualize_output(image, mask)
                        logger.info('Model: {}:model type {}:PreProcess: {}:Process time: {}'.format(model_name, model_type, pre_process, te))
                        
                        
                        
                        response_code = 200
                    else: response_code = 400
                        
        
        elif model_name == "Residual U-Net":
            # """
            #     take image in shape (1, 256, 256, 1)
            #     return a mask in shape (1, 256, 256, 1)
            # """
            st.write("A `U-Net` base model with `Residual` blocks.\n\n**Accuracy**: `99%`")
            model_type = st.sidebar.radio("model_type", ["H5", "tflite"])
            
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])
                if file is not None:
                    processed_image = stream_data(file, pre_process=pre_process)
                    image = load_data(Image.open(file), pre_process)
                    image = image.reshape(1, 256, 256, 1)
    

                    st.image(processed_image, use_column_width=True)
                    submit_button = st.form_submit_button(label='Submit')

                    if submit_button:
                        if model_type == "H5":
                            ts = time()
                            mask = predict(model_unet, image)
                            resources = Resource_Manager()
                            resources = resources.monitor()
                            te = time()
                            te -= ts
                        else:
                            ts = time()
                            mask = model_unet_lt.predict(image)
                            resources = Resource_Manager()
                            resources = resources.monitor()
                            te = time()
                            te -= ts
                        _ = visualize_output(image, mask)
                        logger.info('Model: {}:model type {}:PreProcess: {}:Process time: {}'.format(model_name, model_type, pre_process, te))
                        response_code = 200
                    else: response_code = 400
                    
        
        
        elif model_name == "Efficientnet-U-Net":
            # """
            #     take image in shape (1, 256, 256, 3)
            #     return a mask in shape (1, 256, 256, 1)
            # """
            st.write("A `U-Net` base model with pre-trained `Efficientnet` as **Backbone**.\n\n**Accuracy**: `98%`")
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])   
            if file is not None:
                pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])
                processed_image = stream_data(file, pre_process=pre_process)

                image = cv2.merge((processed_image, processed_image, processed_image)) # 1 channel to 3 channels convertor
                image = image.reshape(1, 256, 256, 3)
                
                st.image(processed_image, use_column_width=True)
                submit_button = st.form_submit_button(label='Submit')
                

                if submit_button:
                    ts = time()
                    mask = model_eff.predict(image)
                    resources = Resource_Manager()
                    resources = resources.monitor()
                    te = time()
                    te -= ts
                    _ = visualize_output(image, mask)
                    logger.info('Model: {}:PreProcess: {}:Process time: {}'.format(model_name, pre_process, te))
                    response_code = 200
                else: response_code = 400


        elif model_name == "VGG16-U-Net":
            # """
            #     take image in shape (1, 256, 256, 3)
            #     return a mask in shape (1, 256, 256, 1)
            # """
            st.write("A `U-Net` base model with pre-trained `VGG16` as **Backbone**.\n\n**Accuracy**: `97%`")
            
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])   
            if file is not None:
                pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])
                processed_image = stream_data(file, pre_process=pre_process)
                
                image = cv2.merge((processed_image, processed_image, processed_image))  # 1 channel to 3 channels convertor
                image = image.reshape(1, 256, 256, 3)
                
                st.image(processed_image, use_column_width=True)
                submit_button = st.form_submit_button(label='Submit')
                

                if submit_button:
                    ts = time()
                    mask = model_vgg.predict(image)
                    resources = Resource_Manager()
                    resources = resources.monitor()
                    te = time()
                    te -= ts
                    _ = visualize_output(image, mask)
                    logger.info('Model: {}:PreProcess: {}:Process time: {}'.format(model_name, pre_process, te))
                    response_code = 200
                else: response_code = 400

        
        elif model_name == "Autoencoder (VAE)":
            # """
            #     take to number and return (output_number ^ 2) images in range (-vae_range, +vae_range)
            # """
            vae_range = st.sidebar.slider("Autoencoder range", 0, 30, step=1)
            output_number = st.sidebar.slider("How many image?", 0, 30, step=1)

            submit_button = st.form_submit_button(label='Submit')
            if submit_button:
                if (vae_range != 0) and (output_number != 0):
                    ts = time()
                    figure = visualize_vae(model_decoder, output_number, vae_range)
                    resources = Resource_Manager()
                    resources = resources.monitor()
                    te = time()
                    te -= ts
                    logger.info('Model: {}:Settings: {}, {}:Process time: {}'.format(model_name, vae_range, output_number, te))
                    response_code = 201


        elif model_name == "MLFlow":
            # """
            #     model tracker
            # """
            st.write("A **service** to track all runs and model's scores.\n\nFor exit this press `Ctrl`+`C` in **Terminal**.")
            model_name = st.sidebar.selectbox(
                                'Select model for mlflow',
                                [None, "unet", "resunet", "efficientnet", "vggU"])
            mlflow_button = st.form_submit_button(label='Mlflow')
            if (mlflow_button) and (model_name is not None):
                logger.info(":mlflow for: {}".format(model_name))
                st.write("Check here:\n`http://127.0.0.1:5000/`")
                mlflow_server(model_name)
                

if response_code == 200:
    
    st.write("""
                    ### input CXR and detected lung
                    """)
    st.image("../images/output_figure.png", use_column_width=False)
    st.success('Mask detected successfully! :thumbsup:')

elif response_code == 201:
    st.write("""
                ### Generated Image(s)
                -----------------------
                """)
    st.image(figure, use_column_width=True)
    st.success('Images successfully generated! :thumbsup:')
    