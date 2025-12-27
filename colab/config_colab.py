# -*- coding: utf-8 -*-
"""
Configuration settings for A2RL project (Colab Version).
Centralizes hyperparameters, paths, and validation thresholds.
"""

# --- Paths (Colab/Drive) ---
import datetime
import os
now = datetime.datetime.now()
date_str = now.strftime('%Y%m%d')
time_str = now.strftime('%H%M%S')
# Environment Detection
IS_COLAB = 'COLAB_GPU' in os.environ
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_COLAB:
    print("Config: Detected Google Colab Environment")
    # Assuming Google Drive is mounted at /content/drive
    DRIVE_ROOT = '/content/drive/MyDrive/A2RL/A2RL-Test'
    DATA_ROOT = '/content/drive/MyDrive/A2RL/data'
    LOG_SUMMARY_ROOT = '/content'  # Local VM disk for speed, sync later
    
elif IS_KAGGLE:
    print("Config: Detected Kaggle Environment")
    # Kaggle directory structure
    DRIVE_ROOT = '/kaggle/working/A2RL-Test' # Output directory
    DATA_ROOT = '/kaggle/input/a2rl-data'    # Read-only input data
    LOG_SUMMARY_ROOT = '/kaggle/working'     # Writable output directory
    
    # Ensure working dirs exist in Kaggle
    if not os.path.exists(DRIVE_ROOT):
        os.makedirs(DRIVE_ROOT, exist_ok=True)
        
else:
    print("Config: Detected Local Environment (Fallback)")
    DRIVE_ROOT = './'
    DATA_ROOT = './data'
    LOG_SUMMARY_ROOT = './'

TRAIN_PATH = DATA_ROOT # Update this if data is elsewhere
      
MODEL_SNAPSHOT = os.path.join(DRIVE_ROOT, 'a2rl_model/model-spp-max')
if IS_COLAB:
    SAVE_MODEL_DIR = os.path.join(DRIVE_ROOT, 'save_model', date_str)
elif IS_KAGGLE:
    SAVE_MODEL_DIR = os.path.join(LOG_SUMMARY_ROOT, 'save_model', date_str) 
else:
    SAVE_MODEL_DIR = os.path.join(LOG_SUMMARY_ROOT, 'save_model', date_str) 

SUMMARY_DIR = os.path.join(LOG_SUMMARY_ROOT, 'summary', 'A2RL_a3c', date_str)
LOG_DIR = os.path.join(LOG_SUMMARY_ROOT, 'logs', date_str)
ALEXNET_NPY = os.path.join(DRIVE_ROOT, 'alexnet.npy')

# --- A3C Hyperparameters ---
ACTOR_LR = 1.0e-4
CRITIC_LR = 1.0e-4
DISCOUNT_FACTOR = 0.99
BETA = 0.05

THREADS = 2  # Colab usually gives 2 cores, so 2-4 threads is appropriate

# --- RL Agent Parameters ---
T_MAX = 50          # Maximum steps per episode
UPDATE_FREQ = 10    # Model update frequency (t_max)
STEP_PENALTY = 0.0
MIN_STEPS = 5      # Encourage at least 5 steps before STOP
BATCH_SIZE = 32
TRAIN_SIZE = 100
NUM_BATCHES = TRAIN_SIZE // BATCH_SIZE

EPOCH_SIZE = 100

INITIAL_SCORE_THRESHOLD = 10.0
PREPROCESS_BATCH_SIZE = 64
PREPROCESS_WORKERS = 2 # Reduced for Colab
FILTERED_TRAIN_PATH = DATA_ROOT + '/Filtered_Train'
USE_FILTERED_DATA = True 

# --- K-fold Cross-Validation Settings ---
#USE_K_FOLD = True
USE_K_FOLD = False
K_FOLDS = 5
VALIDATION_FREQ = 1 

# --- Image Validation Settings ---
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
MIN_FILE_SIZE = 1024  # 1 KB
MIN_IMAGE_DIMENSION = 50 

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
ENABLE_FEATURE_SCALING = False
FEATURE_SCALING_METHOD = 'global'
FEATURE_STATS_PATH = DRIVE_ROOT + '/feature_stats.json'
FEATURE_EPSILON = 1e-8

# --- Mini-Batch Training Settings ---
ENABLE_MINI_BATCH = True
MINI_BATCH_SIZE = 32 # Accumulate gradients over 32 steps/updates before applying

# --- Save Settings ---
SAVE_INTERVAL_MINUTES = 10

# --- Logging Settings ---
# Frequency for diagnostic logs (feature stats, policy stats, etc.)
# Kaggle: Log every 100 episodes, Local: Log every 10 episodes
LOG_FREQ_DIAGNOSTICS = 100 if IS_KAGGLE else 10

# Frequency for cumulative statistics logs
# Kaggle: Log every 100 episodes, Local: Log every 10 episodes  
LOG_FREQ_STATS = 100 if IS_KAGGLE else 10

# Image loading progress log interval
# Kaggle: Log every 50 images, Local: Log every 10 images
LOG_INTERVAL_IMAGE_LOADING = 50 if IS_KAGGLE else 10

# Preprocessing progress log interval
# Kaggle: Log every 500 images, Local: Log every 100 images
LOG_INTERVAL_PREPROCESSING = 500 if IS_KAGGLE else 100
