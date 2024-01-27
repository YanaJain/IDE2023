import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from moviepy.editor import *

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split

LRCN_model=tf.keras.load_model("model_LRCN.model")

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

IMAGE_HEIGHT , IMAGE_WIDTH = 128, 128

SEQUENCE_LENGTH = 20

DATASET_DIR = "UCF50/UCF50"

CLASSES_LIST = ["SkateBoarding", "PullUps"]


video_reader = cv2.VideoCapture(0)

original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                               video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

frames_queue = deque(maxlen = SEQUENCE_LENGTH)

predicted_class_name = ''

while video_reader.isOpened():

    ok, frame = video_reader.read() 

    if not ok:
        break

    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

    normalized_frame = resized_frame / 255

    frames_queue.append(normalized_frame)

    if len(frames_queue) == SEQUENCE_LENGTH:

        predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

        predicted_label = np.argmax(predicted_labels_probabilities)

        predicted_class_name = CLASSES_LIST[predicted_label]

    cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    video_writer.write(frame)

video_reader.release()
video_writer.release()
