from data import *
from visualization import *
import argparse

parser = argparse.ArgumentParser(description="CXR Segmentation & Autoencoder")

parser.add_argument('-p', '--path', type=str, metavar='',
                    default="../images/00000165_008.png", help='Path to Image [../images/00000165_008.png]')

process = parser.add_mutually_exclusive_group()
process.add_argument('-d', '--dhe', action='store_true', help='Select Pre-Process')
parser.add_argument('-n', "--n_samples", type=int, metavar='',
                    default=1, help='Number of Output [1, as many as in folder path]')


network = parser.add_mutually_exclusive_group()
network.add_argument('-S', '--streamlit', action='store_true', help='Run App')
network.add_argument('-U', '--unet', action='store_true', help='Select U-Net')
network.add_argument('-R', '--resunet', action='store_true', help='Select Residual U-Net')
network.add_argument('-V', '--vae', action='store_true', help='Select Variational Autoencoder')

parser.add_argument('-vr', "--vae_range", type=int, metavar='', default=1, help='Autoencoder Range [1, 20]')
parser.add_argument('-on', "--output_number", type=int, metavar='', default=1, help='Number of Output [1, 20]')

args = parser.parse_args()

if __name__ == '__main__':
    if args.unet:
        from unet import Unet_Builder

        # X.shape = (n, 256, 256, 1)
        X = get_data(args.path, n_samples=args.n_samples, pre_process=args.dhe)
        model = Unet_Builder(pretrained_weights='../weigths/cxr_seg_unet.hdf5',
                             input_size=(256, 256, 1))

        predicted = model.predict(X)
        visualize_output(X, predicted)
        print("\n\noutput stored in images/output_figure.png")
        print("Code Complete!")

    elif args.resunet:
        from resunet import ResUnet_Builder

        # X.shape = (n, 256, 256, 1)
        X = get_data(args.path, n_samples=args.n_samples, pre_process=args.dhe)
        model = ResUnet_Builder(pretrained_weights='../weigths/cxr_seg_res_unet.hdf5',
                                input_size=(256, 256, 1))

        predicted = model.predict(X)
        visualize_output(X, predicted)
        print("\n\noutput stored in images/output_figure.png")
        print("Code Complete!")

    elif args.vae:
        from vae import decoder

        decoder = decoder(pretrained_weights="../weigths/decoder.hdf5")

        if (args.vae_range != 0) and (args.output_number != 0):
            visualize_vae(decoder, args.output_number, args.vae_range)
            print("\n\noutput stored in images/output_vae.png")
            print("Code Complete!")
        else:
            print("Wrong inputs")

    elif args.streamlit:
        os.system('./run.sh')

    else:
        print('Run "python service.py -h" for more information')
