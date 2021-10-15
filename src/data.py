import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("PreProcess")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('../logs/preprocess.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

reference_shape = (256, 256, 1)

# run locally (NOT in streamlit)
def get_data(image_s_path, dim=256, n_samples=1, pre_process=False):
    """
        load images for models when you run in your local system.
        :image_s_path: where is image
        :dim: shape of image 
        :n_samples: number of loaded image(s)
        :pre_process: pre-process for loaded images {DHE:True, Original: False}
    """
    im_array = []
    shape = (dim, dim)
    if os.path.isdir(image_s_path):
        test_files = random.choices(list(os.listdir(image_s_path)), k=n_samples)
        for i in tqdm(test_files):
            im = cv2.imread(os.path.join(image_s_path, i))
            im = cv2.resize(im, shape)[:, :, 0]
            if pre_process:
                im = cv2.equalizeHist(im)
            im_array.append(im)
    elif os.path.isfile(image_s_path):
        im = cv2.imread(os.path.join(image_s_path))
        im = cv2.resize(im, shape)[:, :, 0]
        if pre_process:
            im = cv2.equalizeHist(im)
        im_array.append(im)
    # reshape & normalize
    im_array = np.array(im_array).reshape(len(im_array), dim, dim, 1)
    im_array = (im_array - 127.0) / 127.0
    logger.debug("Process Complete!")
    return im_array

# run on streamlit
def load_data(image, pre_process="Original", dim=256):
    """
        load images for models when you run streamlit.
        :image: uploaded image in streamlit.
        :dim: shape of image 
        :pre_process: pre-process for loaded images (DHE or Original)
    """
    image = ImageOps.fit(image, (dim, dim))
    image = np.asarray(image)
    # pil to cv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # check channel
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compatible size
        if image.shape != reference_shape:
            image = cv2.resize(image, (dim, dim))

    # selected pre_process
    if pre_process == "DHE":
        image = cv2.equalizeHist(image)

    # reshape & normalize
    image = image.reshape(1, dim, dim, 1)
    image = (image - 127.0) / 127.0
    logger.debug("Process Complete!")
    return image


def dataset(dim=256, path_to_dataset:str = '../training/data', pre_process: bool = False):
    im_array = []
    mask_array = []
    shape = (dim, dim)
    images_path = os.path.join(path_to_dataset, 'images')
    masks_path = os.path.join(path_to_dataset, 'masks')
    files = os.listdir(images_path)

    # X_shape = image_size
    for i in tqdm(files): 
        
        # im.shape = (X_shape, X_shape, 1)
        im = cv2.imread(os.path.join(images_path, i))
        im = cv2.resize(im, shape)[:, :, 0]
        if pre_process:
            im = cv2.equalizeHist(im)

        # mask.shape = (X_shape, X_shape, 1)
        mask = cv2.imread(os.path.join(masks_path, i))
        mask = cv2.resize(mask, shape)[:, :, 0]
        
        im_array.append(im)
        mask_array.append(mask)

    im_array = np.array(im_array).reshape(len(im_array), dim, dim, 1)
    mask_array = np.array(mask_array).reshape(len(mask_array), dim, dim, 1)
    
    # return list
    return im_array, mask_array



def stream_data(file, pre_process="Original", dim=256):
    """
        :file:  path that pointed to the uploaded image in streamlit
        :pre_process:   Original or DHE for image
        :dim:  size for image
    """
    image = Image.open(file)
    image = ImageOps.fit(image, (dim, dim))
    image = np.asarray(image)
    # pil to cv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # check channel
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compatible size
        if image.shape != reference_shape:
            image = cv2.resize(image, (dim, dim))

    # selected pre_process
    if pre_process == "DHE":
        image = cv2.equalizeHist(image)

    # reshape & normalize
    image = image.reshape(1, dim, dim, 1)
    image = image / 255.
    return image

    