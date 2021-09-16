from resunet import *
from unet import *
from data import *
from vae import decoder
import streamlit as st

st.write("""
    # Lung Segmentation App
    **Chest Xray**
""")
# select model
model_name = st.sidebar.selectbox(
    'Select model',
    [None, "U-Net", "Residual U-Net", "Autoencoder (VAE)"])

if model_name == "Residual U-Net":
    pre_process = st.sidebar.radio("Pre-Process", ["Original", "DHE"])
    model = ResUNet(pretrained_weights="weigths/cxr_seg_res_unet.hdf5")

    # upload your image
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    if file is not None:
        uploaded_image = Image.open(file)
        image = load_data(uploaded_image, pre_process)
        image = image.reshape(1, 256, 256, 1)
        # get result
        mask = model.predict(image)

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
    model = unet(pretrained_weights="weigths/cxr_seg_unet.hdf5")

    # upload your image
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    if file is not None:
        uploaded_image = Image.open(file)
        image = load_data(uploaded_image, pre_process)
        image = image.reshape(1, 256, 256, 1)
        # get result
        mask = model.predict(image)

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

    decoder = decoder(pretrained_weights="weigths/decoder.hdf5")

    if (vae_range != 0) and (output_number != 0):
        dim = 256
        figure = np.zeros((dim * output_number, dim * output_number, 1))

        grid_x = np.linspace(-vae_range, vae_range, output_number)
        grid_y = np.linspace(-vae_range, vae_range, output_number)[::-1]

        # decoder for each square in the grid
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(dim, dim, 1)
                figure[i * dim: (i + 1) * dim,
                j * dim: (j + 1) * dim] = digit

        plt.figure(figsize=(10, 10))
        # Reshape for visualization
        fig_shape = np.shape(figure)
        figure = figure.reshape((fig_shape[0], fig_shape[1]))
        st.write("""
                    ### Generated Image(s)
                    -----------------------
                    """)
        st.image(figure, use_column_width=True)

