from resunet import ResUnet_Builder
from unet import Unet_Builder
from vae import decoder
from memory_profiler import LogFile, profile
import sys
sys.stdout = LogFile('model_memory_usage_log')

fp=open('../logs/model_memory_usage.log', 'w+')
"""
    load model by it's name
    :model_name: network to use
    :pretrained: use pretrained_weights or not
"""
@profile(stream=fp)
def load_model(model_name, pretrained: bool = True):
    if model_name == 'U-Net':
        model = Unet_Builder(pretrained=pretrained, pretrained_weights="../weights/cxr_seg_unet.hdf5",
                              input_size=(256, 256, 1))
    
    elif model_name == 'ResidualU':
        model = ResUnet_Builder(pretrained=pretrained, pretrained_weights="../weights/cxr_seg_res_unet.hdf5",
                              input_size=(256, 256, 1))
    
    elif model_name == 'Autoencoder (VAE)':
        model = decoder(pretrained=pretrained, pretrained_weights="../weights/decoder.hdf5")
    
    return model

# predictions
@profile(stream=fp)
def predict(model, image):
    return model.predict(image)


@profile(stream=fp)
def del_model(model):
    del model

model_runet = load_model("ResidualU", pretrained=True)
model_unet = load_model("U-Net", pretrained=True)
model_decoder = load_model("Autoencoder (VAE)", pretrained=True)

del_model(model_runet)
del_model(model_unet)
del_model(model_decoder)