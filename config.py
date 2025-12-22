# -*- coding: utf-8 -*-
"""
Configuration settings for A2RL project.
Centralizes hyperparameters, paths, and validation thresholds.
"""

# --- Paths ---
TRAIN_PATH = '../AVA/Train8954'
MODEL_SNAPSHOT = '../a2rl_model/model-spp-max'
SAVE_MODEL_DIR = '../save_model'
SUMMARY_DIR = '../summary/A2RL_a3c'
LOG_DIR = '../logs'
ALEXNET_NPY = 'alexnet.npy'

# --- A3C Hyperparameters ---
ACTOR_LR = 2.5e-4
CRITIC_LR = 2.5e-4
DISCOUNT_FACTOR = 0.99
BETA = 0.05
THREADS = 1  # Number of agent threads

# --- RL Agent Parameters ---
T_MAX = 50          # Maximum steps per episode
UPDATE_FREQ = 10    # Model update frequency (t_max)
STEP_PENALTY = 0.001
BATCH_SIZE = 32
TRAIN_SIZE = 9000
NUM_BATCHES = TRAIN_SIZE // BATCH_SIZE  # 281
EPOCH_SIZE = 20

# --- K-fold Cross-Validation Settings ---
USE_K_FOLD = True
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
