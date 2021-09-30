from resunet import ResUnet_Builder
from unet import Unet_Builder
from data import *
from vae import decoder
from visualization import visualize_vae
import streamlit as st
# UI
st.write("""
    # Lung Segmentation App
    **Chest Xray**
""")
# select model
model_runet = ResUnet_Builder(pretrained_weights="weigths/cxr_seg_res_unet.hdf5",
                              input_size=(256, 256, 1))
model_unet = Unet_Builder(pretrained_weights="weigths/cxr_seg_unet.hdf5",
                              input_size=(256, 256, 1))
model_decoder = decoder(pretrained_weights="weigths/decoder.hdf5")

"""select a model"""
model_name = st.sidebar.selectbox(
    'Select model',
    [None, "U-Net", "Residual U-Net", "Autoencoder (VAE)"])

if model_name == "Residual U-Net":
    pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])


    # upload your image
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    if file is not None:
        uploaded_image = Image.open(file)
        image = load_data(uploaded_image, pre_process)
        image = image.reshape(1, 256, 256, 1)
        # get result
        mask = model_runet.predict(image)

        st.write("""
                ### input image
                -----------------------
            """)
        st.image(uploaded_image, use_column_width=True)
        st.write("""
                        ### Predicted Mask
                        -----------------------
                    """)
        st.image(mask, use_column_width=False)
    else:
        st.text("Please upload an image file:")

elif model_name == "U-Net":
    pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])


    # upload your image
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    if file is not None:
        uploaded_image = Image.open(file)
        image = load_data(uploaded_image, pre_process)
        image = image.reshape(1, 256, 256, 1)
        # get result
        mask = model_unet.predict(image)

        st.write("""
                ### input image
                -----------------------
            """)
        st.image(uploaded_image, use_column_width=True)
        st.write("""
                        ### Predicted Mask
                        -----------------------
                    """)
        st.image(mask, use_column_width=False)
    else:
        st.text("Please upload an image file:")

elif model_name == "Autoencoder (VAE)":

    vae_range = st.sidebar.slider("Autoencoder range", 0, 20, step=1)
    output_number = st.sidebar.slider("How many image?", 0, 20, step=1)

    if (vae_range != 0) and (output_number != 0):
        figure = visualize_vae(model_decoder, output_number, vae_range)

        st.write("""
                    ### Generated Image(s)
                    -----------------------
                    """)
        st.image(figure, use_column_width=True)
