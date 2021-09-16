import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

reference_shape = (256, 256, 1)

# run locally (NOT in streamlit)
def get_data(image_s_path, dim=256, n_samples=1):
    im_array = []
    shape = (dim, dim)
    if len(os.listdir(image_s_path)) != 1:
        test_files = random.choices(list(os.listdir(image_s_path)), k=n_samples)
        for i in tqdm(test_files):
            im = cv2.imread(os.path.join(image_s_path, i))
            im = cv2.resize(im, shape)[:, :, 0]
            im = cv2.equalizeHist(im)
            im_array.append(im)
    else:
        im = cv2.imread(os.path.join(image_s_path))
        im = cv2.resize(im, shape)[:, :, 0]
        im = cv2.equalizeHist(im)
        im_array.append(im)
    # reshape & normalize
    im_array = np.array(im_array).reshape(len(im_array), dim, dim, 1)
    im_array = (im_array - 127.0) / 127.0
    return im_array

# run on streamlit
def load_data(image, pre_process="Original", dim=256):
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
    return image
