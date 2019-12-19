# AI Workshop: Computer Vision Tools #

**Installation notes for Ubuntu platforms:**

- install Python and Jupyter notebook or IDE (like Pycharm)

- pip install -r requirements.txt

- sudo apt install tesseract-ocr

Note: use Tensorflow 1.x, not 2.x


**Key-points of discussion:**
1. Image pre-processing
2. Object detection
3. Image classification
4. Optical character recognition (OCR)
5. Common tools basic description

**Brief files description:**

"object_detection.ipynb"

Contains examples of common objects detection from COCO dataset along with face detection from images.

Libraries used: CVlib, OpenCV, Matplotlib

"webcam_face_gender_detect.ipynb"

Performs face detection from webcam image flow with the following gender detection over each face.

Libraries used: CVlib, OpenCV, Numpy

"image_classification_keras_MNIST.ipynb"

Shows an example of building the CNN from scratch for hand written digits classification. 
Includes all the pre-processing pipe-line for the images along with the model training and performance metrics evaluation. 
Uses MNIST dataset.

Libraries used: Keras, Tensorflow, Numpy, Matplotlib, Scikit-learn

"ocr_with_pytesseract.ipynb"

Contains examples of optical character recognition, 
based on Google's Tesseract with it's Python wrapper pytesseract on different images. 
Shows bounding boxes extraction pipe-line.

Libraries used: Tesseract, pytesseract, Pillow, Matplotlib, OpenCV


**Useful links:**

CVlib: https://github.com/arunponnusamy/cvlib

OpenCV: https://pypi.org/project/opencv-python/

Numpy: https://numpy.org/

Matplotlib: https://matplotlib.org/

Keras: https://keras.io/

Tensorflow: https://www.tensorflow.org/

Scikit-learn: https://scikit-learn.org/stable/

Pytesseract: https://pypi.org/project/pytesseract/

Pillow: https://pypi.org/project/Pillow/

MNIST: https://en.wikipedia.org/wiki/MNIST_database

COCO: http://cocodataset.org/#home

Optical character recognition: https://en.wikipedia.org/wiki/Optical_character_recognition

Nice review for further reading on some uncovered here CV Tools: 
https://opensource.com/article/19/3/python-image-manipulation-tools
