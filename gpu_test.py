import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


import cv2
import numpy as np
import tensorflow as tf

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("TF:", tf.__version__)