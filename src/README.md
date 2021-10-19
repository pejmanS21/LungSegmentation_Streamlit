# LungSegmentation_Streamlit
CXR segmentation with **streamlit**

# Dataset
* Montgomery
* Shenzhen

https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels

## Pre-Process
* Original (None)
* DHE

# model
### U-Net

![Convolutional-neural-network-CNN-architecture-based-on-UNET-Ronneberger-et-al](https://user-images.githubusercontent.com/73995528/133255424-8fc99c61-e163-4f27-bd24-2760102ee121.png)

### residual U-Net

![arch](https://user-images.githubusercontent.com/73995528/133255516-5b2183ac-ebdc-4795-b8fd-e2481793c0e9.png)

_pretrained weights are stored in weights folder_

# Usage
    chmod +x ./run.sh
or 

    streamlit run ./ui.py

Automatically open up this link
* Local URL: http://localhost:8502

# Results
## APP 
![streamlit_new](https://user-images.githubusercontent.com/73995528/137863603-496c0356-6286-48ab-80e8-58c94342f489.gif)

--------------------------------------------------------------------

# Parser
    python service.py -h
| small flag  |  flag     | Description     |
| :----:      |    :----: |     :---:      |
|-h| --help|            show this help message and exit|
|-p | --path|           Path to Image [images/00000165_008.png]|
|-d| --dhe |            Select Pre-Process|
|-n | --n_samples|      Number of Output [1, as many as in folder path]|
|-S| --streamlit   |    Run App|
|-U| --unet       |     Select U-Net|
|-R| --resunet     |    Select Residual U-Net|
|-V| --vae          |   Select Variational Autoencoder|
|-vr | --vae_range   |  Autoencoder Range [1, 30]|
|-on | --output_number| Number of Output [1, 30]|

**all outputs stored in images directory**




# Dockerfile

    docker build -t <image-name> .
    docker run -p 8501:8501 -d <image-name>


https://github.com/pejmanS21/LungSegmentation_Streamlit.git
