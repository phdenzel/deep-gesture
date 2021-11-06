"""
deep_gesture.__init__

@author: phdenzel
"""
import os
import numpy as np

import deep_gesture.camera
import deep_gesture.holistic
import deep_gesture.record
import deep_gesture.process
import deep_gesture.utils

# import cv2
# from matplotlib import pyplot as plt
# import mediapipe as mp

# from scipy import stats
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# from tensorflow.keras.utils import to_categorical

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import TensorBoard


# Directories
HOME_DIR = os.path.expanduser("~")
DOT_DIR = os.path.join(HOME_DIR, ".deep_gesture")
DATA_DIR = os.path.join(DOT_DIR, "data")
TMP_DIR = os.path.join(DOT_DIR, "tmp")

# Settings
device_id = 0

# Training settings
gestures = ['swipe_left', 'swipe_right',
            'scroll_down', 'scroll_up',
            'play', 'stop',
            'zoom_in', 'zoom_out']
N_training_sequences = 3
sequence_length = 30




