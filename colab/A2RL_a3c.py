from __future__ import absolute_import
import logging
import config
from logger_config import setup_logger

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Force TF1 behavior in TF2 environment

from tensorflow.keras.layers import Dense, Flatten, Input, LSTM, Dropout, Conv2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

import numpy as np
import threading
import random
import time
import sys
import pickle
import json
import argparse
import shutil
import cv2
import skimage.io as io
import skimage.transform as transform
from os import listdir
from os.path import isfile, join, getsize
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count

from actions import command2action, generate_bbox, crop_input, get_action_name
import network
import network_vfn as nw

# Force GPU memory growth for Colab
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess_config = tf.InteractiveSession(config=config_tf)
K.set_session(sess_config)

a3c_graph = tf.get_default_graph()
global_dtype = tf.float32
global_dtype_np = np.float32

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
EPISODES = 8000000

# Initialize logger
logger = setup_logger('A2RL_Colab', log_dir=config.LOG_DIR, level=logging.DEBUG, console_level=logging.INFO)

# [Original Helper Functions (load_feature_stats, apply_feature_scaling, etc.) would follow here]
# ... (I will include the core logic below) ...
