import sys
sys.path.append('../src')

from model_loader import *
from data import dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.tensorflow
import argparse

parser = argparse.ArgumentParser(description="CXR Segmentation")
# where is dataset?
parser.add_argument('-p', '--path', type=str, metavar='',
                    default="../training/data", help='Path to dataset [../training/data]')
# Original Images or pre-processed ones?
process = parser.add_mutually_exclusive_group()
process.add_argument('-d', '--dhe', action='store_true', help='Select Pre-Process')

# U-Net Or Residual U-Net?
network = parser.add_mutually_exclusive_group()
network.add_argument('-N', '--network', type=str, 
                    metavar='', default="U-Net", 
                    help="Select Model['U-Net', 'Residual U-Net']")
parser.add_argument('-su', '--summary', action='store_true', help="Print model summary") # print model summary

"""
    :hyper parameters: for training
"""
parser.add_argument('-bs', "--batch_size", type=int, metavar='', default=16, help='batch_size')
parser.add_argument('-e', "--epochs", type=int, metavar='', default=2, help='Number of epochs')
parser.add_argument('-H', '--history', action='store_true', help='Show history')

"""
    :mlflow: track all model resualts
"""
mlrecords = parser.add_mutually_exclusive_group()
mlrecords.add_argument('-ML', '--mlflow', action='store_true', help="Save Models with MLFlow")

"""
    :save model: save model for future
    :model name: a name for saved file
    :weights: save weights as hdf5 or save model as h5 
    :location: model will be saved in weights directory
"""
save_model = parser.add_mutually_exclusive_group()
save_model.add_argument('-S', '--save', action='store_true', help="Save Model")
parser.add_argument('-mn', '--modelName', type=str, metavar='', default="saved_model", help="Model Name for Saving")
parser.add_argument('-W', '--weights', action='store_true', help="Save Model's weights")

args = parser.parse_args()


if __name__ == "__main__":
    if args.mlflow:
        mlflow.autolog()
    
    # load dataset
    images, masks = dataset(256, path_to_dataset=args.path, pre_process=args.dhe)

    X_train, X_val, Y_train, Y_val = train_test_split((images - 127.0) / 127.0, 
                                                    (masks > 127).astype(np.float32), 
                                                    test_size = 0.15, 
                                                    random_state = 2018)

    model = load_model(args.network, pretrained=False)
    if args.summary:
        model.summary() ## print summary
    res = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=args.batch_size, epochs=args.epochs)

    if args.save:
        model.save_model(saved_model_name=args.modelName, weights=args.weights)

    # Show history for train and validation
    if args.history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
        ax1.plot(res.history['loss'], '-o', label = 'Loss')
        ax1.plot(res.history['val_loss'], '-o', label = 'Validation Loss')
        ax1.legend()

        ax2.plot(100 * np.array(res.history['binary_accuracy']), '-o', 
                label = 'Accuracy')
        ax2.plot(100 * np.array(res.history['val_binary_accuracy']), '-o',
                label = 'Validation Accuracy')
        ax2.legend()
        plt.show()
    