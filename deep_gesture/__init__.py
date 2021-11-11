"""
deep_gesture.__init__

@author: phdenzel
"""
import os

import deep_gesture.camera
import deep_gesture.holistic
import deep_gesture.record
import deep_gesture.models
import deep_gesture.process
import deep_gesture.utils

# from scipy import stats


# Directories
HOME_DIR = os.path.expanduser("~")
DOT_DIR = os.path.join(HOME_DIR, ".deep_gesture")
DATA_DIR = os.path.join(DOT_DIR, "data")
MDL_DIR = os.path.join(DOT_DIR, "models")
LOG_DIR = os.path.join(DOT_DIR, "log")
TMP_DIR = os.path.join(DOT_DIR, "tmp")

# Settings
device_id = 0

# Collect settings
gestures = ['hello', 'good', 'bad', 'ok']
N_training_sequences = 3
sequence_length = 30

# Training settings
optimizer = 'Adam'
learning_rate = 0.01
epochs = 1000
batch_size = 32
