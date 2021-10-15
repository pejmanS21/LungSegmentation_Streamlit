# Lung Segmentation

To run `streamlit` app go to **/src** directory.

To run your own segmentation model go to **/training** directory.

To test modules go to **/tests** directory.

To see runtime history go to **/logs** directory.

To see models result (`metrics`) go to **/Records** directory.

# Dockerfile

First create an image by running:

    docker build -t <image-name> .

Then to run created image:

    docker run -p 8501:8501 -d <image-name>
