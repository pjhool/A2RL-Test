# -*- coding: utf-8 -*-
"""
Configuration settings for A2RL project.
Centralizes hyperparameters, paths, and validation thresholds.
"""

# --- Paths ---
import datetime
import os
now = datetime.datetime.now()
date_str = now.strftime('%Y%m%d')
time_str = now.strftime('%H%M%S')

#TRAIN_PATH = '../AVA/Train8954'
TRAIN_PATH = 'Y:\\Project_A2RL\\flickr-cropping-dataset\\data'
      
MODEL_SNAPSHOT = './a2rl_model/model-spp-max'
SAVE_MODEL_DIR = os.path.join('./save_model', date_str)
SUMMARY_DIR = os.path.join('./summary/A2RL_a3c', date_str)
LOG_DIR = os.path.join('./logs', date_str)
ALEXNET_NPY = 'alexnet.npy'

# --- A3C Hyperparameters ---
#ACTOR_LR = 2.5e-4
#CRITIC_LR = 2.5e-4
ACTOR_LR = 1.0e-4
CRITIC_LR = 1.0e-4
DISCOUNT_FACTOR = 0.99
#BETA = 0.2 # 
#BETA = 0.05
#BETA = 0.1
#BETA = 0.1
BETA = 0.05

THREADS = 4  # Number of agent threads

# --- RL Agent Parameters ---
T_MAX = 50          # Maximum steps per episode
UPDATE_FREQ = 10    # Model update frequency (t_max)
#STEP_PENALTY = 0.0001
STEP_PENALTY = 0.0
MIN_STEPS = 5      # Encourage at least 5 steps before STOP
BATCH_SIZE = 32
TRAIN_SIZE = 100
NUM_BATCHES = TRAIN_SIZE // BATCH_SIZE  # 3
EPOCH_SIZE = 1
INITIAL_SCORE_THRESHOLD = 10.0
PREPROCESS_BATCH_SIZE = 64
PREPROCESS_WORKERS = 6 # Number of worker processes for parallel preprocessing
FILTERED_TRAIN_PATH = '../AVA/Filtered_Train'
USE_FILTERED_DATA = True # Set to True to use filtered data for training

# --- K-fold Cross-Validation Settings ---
USE_K_FOLD = False
K_FOLDS = 5
VALIDATION_FREQ = 1 # Perform validation after every epoch

# --- Image Validation Settings ---
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
MIN_FILE_SIZE = 1024  # 1 KB
MIN_IMAGE_DIMENSION = 50  # minimum width/height in pixels

# --- Aspect Ratio Validation ---
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0
ASPECT_RATIO_PENALTY = 5.0

# --- Network Architecture ---
STATE_SIZE = (1, 2000)
EMBEDDING_DIM = 1000
RANKING_LOSS = 'svm'
SPP = True
POOLING = 'max'
ACTION_SIZE = 14

# --- Feature Scaling Settings ---
#ENABLE_FEATURE_SCALING = True  # Enable/disable feature scaling
ENABLE_FEATURE_SCALING = False  # Enable/disable feature scaling
FEATURE_SCALING_METHOD = 'global'  # 'standardization', 'minmax', or 'global'
#FEATURE_SCALING_METHOD = 'standardization'  # 'standardization', 'minmax', or 'global'
FEATURE_STATS_PATH = 'feature_stats.json'  # Path to pre-computed statistics
FEATURE_EPSILON = 1e-8  # Small value to prevent division by zero

# --- Mini-Batch Training Settings ---
ENABLE_MINI_BATCH = True
MINI_BATCH_SIZE = 32 # Accumulate gradients over 32 steps/updates before applying

# --- Save Settings ---
SAVE_INTERVAL_MINUTES = 10
