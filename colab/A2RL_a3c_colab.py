print("DEBUG: VERSION 20251226_03 - Using manual set_learning_phase check")
import sys
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import logging
try:
    import colab.config_colab as config
except ImportError:
    import config_colab as config

from logger_config import setup_logger

from skimage.color import rgb2gray
from skimage.transform import resize
# Robust Keras/TensorFlow Compatibility Layer
try:
    import tf_keras as keras
    from tf_keras.layers import Dense, Flatten, Input, LSTM, Dropout, Conv2D
    from tf_keras import backend as K
    from tf_keras.models import Model
    try:
        from tf_keras.optimizers.legacy import RMSprop
    except ImportError:
        from tf_keras.optimizers import RMSprop
    USING_TF_KERAS = True
    print("DEBUG: Using tf_keras (Legacy Keras)")
except ImportError:
    from tensorflow.keras.layers import Dense, Flatten, Input, LSTM, Dropout, Conv2D
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    try:
        from tensorflow.keras.optimizers.legacy import RMSprop
    except ImportError:
        from tensorflow.keras.optimizers import RMSprop
    USING_TF_KERAS = False
    print("DEBUG: Using tensorflow.keras (Keras 3)")

# Determine which argument to use for learning rate
# Older Keras uses 'lr', newer Keras uses 'learning_rate'
import inspect
try:
    _sig = inspect.signature(RMSprop.__init__)
    if 'learning_rate' in _sig.parameters:
        K_LR_NAME = 'learning_rate'
    else:
        K_LR_NAME = 'lr'
except (AttributeError, ValueError, NameError):
    # Fallback for very old Python or Keras versions
    K_LR_NAME = 'lr'

print("DEBUG: Optimizer LR arg: {}".format(K_LR_NAME))
import numpy as np
import threading
import random
import time

#import gym

import sys


import pickle
import json
import argparse
import shutil
import cv2
import skimage.io as io
import skimage.transform as transform

from actions import command2action, generate_bbox, crop_input, get_action_name


import network
import network_vfn as nw

from os import listdir
from os.path import isfile, join, getsize
import os

from datetime import datetime
from multiprocessing import Pool, cpu_count


a3c_graph  = tf.get_default_graph()

global_dtype = tf.float32

global_dtype_np = np.float32

#vfn_sess = None

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
episode_lock = threading.Lock()
EPISODES = 8000000
# 환경 생성
env_name = "BreakoutDeterministic-v4"

drop_ratio = 0.5

# Initialize logger
logger = setup_logger('A2RL', log_dir=config.LOG_DIR, level=logging.DEBUG, console_level=logging.INFO)

# Feature Scaling Helper Functions
def load_feature_stats(stats_path):
    """
    Load pre-computed feature statistics from JSON file.
    
    Args:
        stats_path: Path to feature statistics JSON file
    
    Returns:
        dict: Feature statistics or None if file doesn't exist
    """
    if not os.path.exists(stats_path):
        logger.warning('Feature stats file not found: %s', stats_path)
        logger.warning('Feature scaling will use per-sample statistics')
        return None
    
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        logger.info('Loaded feature statistics from: %s', stats_path)
        logger.info('  Mean: %.6f, Std: %.6f', stats['feature_mean'], stats['feature_std'])
        logger.info('  Computed from %d samples on %s', stats['num_samples'], stats['computed_date'])
        return stats
    except Exception as e:
        logger.error('Failed to load feature stats: %s', e)
        return None


def apply_feature_scaling(features, method='standardization', stats=None, epsilon=1e-8):
    """
    Apply feature scaling to input features.
    
    Args:
        features: numpy array of features to scale
        method: 'standardization', 'minmax', or 'global'
        stats: dict with 'feature_mean' and 'feature_std' for global method
        epsilon: small value to prevent division by zero
    
    Returns:
        scaled features (numpy array)
    """
    if method == 'standardization':
        # Per-sample standardization
        mean = np.mean(features)
        std = np.std(features) + epsilon
        scaled = (features - mean) / std
        
    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        min_val = np.min(features)
        max_val = np.max(features)
        if max_val - min_val > epsilon:
            scaled = (features - min_val) / (max_val - min_val)
        else:
            logger.warning('Feature range too small for min-max scaling, using original')
            scaled = features
            
    elif method == 'global':
        # Global statistics-based standardization
        if stats is None:
            logger.warning('Global method requires stats, falling back to per-sample standardization')
            mean = np.mean(features)
            std = np.std(features) + epsilon
            scaled = (features - mean) / std
        else:
            mean = stats['feature_mean']
            std = stats['feature_std'] + epsilon
            scaled = (features - mean) / std
    else:
        logger.warning('Unknown scaling method: %s, using original features', method)
        scaled = features
    
    return scaled

# This is the definition of helper function
def load_and_validate_image(filepath):
    """
    Load and validate an image file.
    
    Args:
        filepath: Full path to the image file
    
    Returns:
        tuple: (image_array, error_message)
               image_array is None if error occurred
    """
    try:
        # Check file extension
        _, ext = os.path.splitext(filepath.lower())
        if ext not in config.VALID_IMAGE_EXTENSIONS:
            return None, "Invalid extension: {}".format(ext)
        
        # Check file size
        try:
            file_size = getsize(filepath)
            if file_size < config.MIN_FILE_SIZE:
                return None, "File too small ({} bytes)".format(file_size)
        except Exception as e:
            return None, "Cannot get file size: {}".format(e)
        
        # Read image
        img = io.imread(filepath)
        
        # Check dimensions
        if img.ndim != 3:
            return None, "Not a 3-channel image (ndim={})".format(img.ndim)
        
        # Extract RGB channels
        img_rgb = img[:, :, :3]
        
        # Check minimum dimensions
        if img_rgb.shape[0] < config.MIN_IMAGE_DIMENSION or img_rgb.shape[1] < config.MIN_IMAGE_DIMENSION:
            return None, "Image too small: {}x{}".format(img_rgb.shape[0], img_rgb.shape[1])
        
        return img_rgb, None
        
    except Exception as e:
        return None, "Error loading image: {}".format(str(e))


def validate_aspect_ratio(bbox, min_ratio=config.MIN_ASPECT_RATIO, max_ratio=config.MAX_ASPECT_RATIO):
    """
    Validate aspect ratio of bounding box.
    
    Args:
        bbox: Bounding box coordinates [[x1, y1, x2, y2]]
        min_ratio: Minimum acceptable aspect ratio
        max_ratio: Maximum acceptable aspect ratio
    
    Returns:
        tuple: (is_valid, aspect_ratio, penalty)
    """
    x_width = bbox[0][2] - bbox[0][0]
    y_height = bbox[0][3] - bbox[0][1]
    
    # Prevent division by zero
    if y_height < 1e-6:
        return False, 0.0, config.ASPECT_RATIO_PENALTY
    
    aspect_ratio = x_width / y_height
    
    # Check if aspect ratio is within acceptable range
    if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
        return False, aspect_ratio, config.ASPECT_RATIO_PENALTY
    
    return True, aspect_ratio, 0.0


def _preprocess_worker(args):
    """
    Worker function for parallel preprocessing.
    Checks grayscale, persons, faces, and stats.
    Returns (filename, resized_img, passed, error)
    """
    source_path, filename = args
    filepath = os.path.join(source_path, filename)
    
    try:
        # Load image
        img = io.imread(filepath)
        
        # 1. Format & Channel Check
        if img.ndim != 3 or img.shape[2] != 3:
            return filename, None, False, None
            
        # Initialize detectors (local to worker)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # 2. Person Detection (HOG)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        (rects, weights) = hog.detectMultiScale(img_cv, winStride=(8, 8), padding=(16, 16), scale=1.05)
        if len(rects) > 0:
            return filename, None, False, None
            
        # 3. Face Detection
        gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_cv, 1.1, 4)
        if len(faces) > 0:
            return filename, None, False, None
            
        # 4. Variance & Stats
        img_std = np.std(img)
        img_mean = np.mean(img)
        channel_stds = np.std(img, axis=(0, 1))
        max_channel_std = np.max(channel_stds)
        
        total_pixels = img.shape[0] * img.shape[1]
        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        color_ratio = float(unique_colors) / total_pixels
        
        if img_std < 5.0 or img_mean < 10.0 or img_mean > 245.0 or max_channel_std < 3.0 or color_ratio < 0.02:
            return filename, None, False, None
            
        # 5. Success! Prepare for VFN
        # VFN expects (227, 227) resized and normalized
        img_vfn = img.astype(np.float32) / 255
        img_resized = transform.resize(img_vfn, (227, 227)) - 0.5
        
        return filename, img_resized, True, None
        
    except Exception as e:
        return filename, None, False, str(e)


def get_k_fold_splits(train_path, k=5):
    """
    Split files in train_path into K folds.
    """
    files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
    random.shuffle(files)
    
    fold_size = len(files) // k
    folds = []
    for i in range(k):
        if i == k - 1:
            folds.append(files[i * fold_size:])
        else:
            folds.append(files[i * fold_size : (i + 1) * fold_size])
    
    splits = []
    for i in range(k):
        val_files = folds[i]
        train_files = []
        for j in range(k):
            if i != j:
                train_files.extend(folds[j])
        splits.append((train_files, val_files))
        
    return splits

def preprocess_dataset(source_path, target_path, threshold, num_workers=None):
    """
    Scan source_path for images, calculate aesthetic scores in parallel, 
    and copy images with scores below threshold to target_path.
    """
    if num_workers is None:
        num_workers = config.PREPROCESS_WORKERS
        
    if os.path.exists(target_path):
        logger.info("Clearing existing target directory: %s", target_path)
        shutil.rmtree(target_path)
    
    os.makedirs(target_path)
    logger.info("Created clean target directory: %s", target_path)

    all_files = [f for f in os.listdir(source_path) if isfile(join(source_path, f))]
    image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in config.VALID_IMAGE_EXTENSIONS)]
    
    logger.info("Found %d images in %s. Filtering in parallel with %d workers and threshold %.4f...", 
                len(image_files), source_path, num_workers, threshold)
    
    # Initialize Pool
    pool = Pool(processes=num_workers)
    worker_args = [(source_path, f) for f in image_files]
    
    count = 0
    total = len(image_files)
    
    batch_images = []
    batch_filenames = []
    
    try:
        # Use imap_unordered for better memory management and progress tracking
        for i, result in enumerate(pool.imap_unordered(_preprocess_worker, worker_args)):
            filename, img_resized, passed, error = result
            
            if error:
                logger.warning("Failed to process %s: %s", filename, error)
                continue
            
            if passed:
                batch_images.append(img_resized)
                batch_filenames.append(filename)
                
            # Aesthetic scoring in batches for efficiency
            if len(batch_images) >= config.PREPROCESS_BATCH_SIZE:
                scores, _ = evaluate_aesthetics_score_resized(batch_images)
                for j, score in enumerate(scores):
                    if 0 < score < threshold:
                        shutil.copy2(join(source_path, batch_filenames[j]), join(target_path, batch_filenames[j]))
                        count += 1
                
                batch_images = []
                batch_filenames = []
                
            if (i + 1) % 100 == 0:
                logger.info("Checked %d/%d images... (Passed CPU filters: %d, Kept: %d)", 
                            i + 1, total, count + len(batch_images), count)
                            
        # Final batch
        if batch_images:
            scores, _ = evaluate_aesthetics_score_resized(batch_images)
            for j, score in enumerate(scores):
                if 0 < score < threshold:
                    shutil.copy2(join(source_path, batch_filenames[j]), join(target_path, batch_filenames[j]))
                    count += 1
                    
        pool.close()
        pool.join()
        
    except KeyboardInterrupt:
        logger.error("Preprocessing interrupted by user.")
        pool.terminate()
        pool.join()
        sys.exit(1)
    except Exception as e:
        logger.error("Parallel preprocessing failed: %s", e)
        pool.terminate()
        pool.join()
        raise

    logger.info("Preprocessing complete. Total images kept: %d/%d", count, total)

# input : original image
def evaluate_aesthetics_score(images):
    global vfn_sess, a3c_graph
    with a3c_graph.as_default():
        # If a single image is passed as a numpy array, wrap it in a list
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
            
        scores = np.zeros(len(images))
        features = []
        for i, img in enumerate(images):
            # img_resize = transform.resize(img, (227, 227), mode='constant')
            img_resize = img.astype(np.float32)/255
            img_resize = transform.resize(img_resize, (227, 227))-0.5
            img_resize = np.expand_dims(img_resize, axis=0)
            
            # Ensure session is valid
            if vfn_sess is None:
                logger.error("vfn_sess is None in evaluate_aesthetics_score")
                return scores, features
                
            score, feature = vfn_sess.run([score_func], feed_dict={image_placeholder: img_resize})[0]
            scores[i] = score
            features.append(feature)
        return scores, features

def evaluate_aesthetics_score_resized(images):
    with a3c_graph.as_default():
        # If a single image is passed as a numpy array, wrap it in a list
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
            
        scores = np.zeros(len(images))
        features = []
        for i, img in enumerate(images):
            #img = images[i].astype(np.float32)/255
            #img_resize = transform.resize(img, (227, 227))-0.5
            img = images[i].astype(np.float32)
            img_resize = img
            img_resize = np.expand_dims(img_resize, axis=0)
            score  , feature = vfn_sess.run([ score_func ], feed_dict={image_placeholder: img_resize})[0]
            scores[i] = score
            features.append( feature)
        return scores , features



# 브레이크아웃에서의 A3CAgent 클래스(글로벌신경망)
class A3CAgent:
    def __init__(self, action_size):
        global a3c_graph
        a3c_graph = tf.get_default_graph()
        # 상태크기와 행동크기를 갖고옴
        self.state_size = config.STATE_SIZE
        self.action_size = action_size
        # A3C 하이퍼파라미터
        self.discount_factor = config.DISCOUNT_FACTOR
        self.no_op_steps = 30
        self.actor_lr = config.ACTOR_LR
        self.critic_lr = config.CRITIC_LR
        self.beta = config.BETA
        # 쓰레드의 갯수
        self.threads = config.THREADS
        #self.threads = 8
        self.initial_load_path = None
        self._weights_loaded = False


        # 정책신경망과 가치신경망을 생성
        self.actor, self.critic = self.build_model()
        # 정책신경망과 가치신경망을 업데이트하는 함수 생성
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        # 텐서보드 설정

        logger.debug('Parent default graph: %s', tf.get_default_graph())

        # GPU Config for Colab
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config_tf)
        if hasattr(K, 'set_session'):
            K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        #self.load_model("../save_model/A2RL_a3c_run-20181112124128") 
        #self.load_model("../save_model/A2RL_a3c_run-20181112221820") 
        #self.load_model("../save_model/A2RL_a3c_run-20181113044510") 


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()


        # Explicitly create summary directory with timestamp subfolder
        # Structure: .../summary/A2RL_a3c/YYYYMMDD/events_HHMMSS
        time_str = datetime.now().strftime("%H%M%S")
        summary_dir = os.path.join(config.SUMMARY_DIR, "events_{}".format(time_str))

        if not os.path.exists(summary_dir):
            try:
                os.makedirs(summary_dir)
                logger.info("✓ Created summary directory: %s", summary_dir)
            except OSError as e:
                logger.error("✗ Failed to create summary directory %s: %s", summary_dir, e)
                
        try:
            # Create the FileWriter
            self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
            
            # 1. Force file creation by adding a dummy system summary
            dummy_summary = tf.Summary(value=[tf.Summary.Value(tag='System/Status', simple_value=1.0)])
            self.summary_writer.add_summary(dummy_summary, 0)
            
            # 2. Flush to force write to OS buffer
            self.summary_writer.flush()
            
            logger.info("Summary writer initialized, dummy summary added, and flushed at: %s", summary_dir)
        except Exception as e:
            logger.error("Failed to initialize summary writer: %s", e, exc_info=True)




    # 쓰레드를 만들어 학습을 하는 함수
    def train(self, train_path=None, start_fold=0, start_epoch=0):
        self.start_time = time.time()
        logger.info('Training started at: %s', datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'))
        
        if train_path is None:
            train_path = config.TRAIN_PATH
            
        if config.USE_K_FOLD:
            logger.info('Starting K-fold cross-validation (K={}) using path: {}'.format(config.K_FOLDS, train_path))
            splits = get_k_fold_splits(train_path, config.K_FOLDS)
            
            for fold_idx, (train_files, val_files) in enumerate(splits):
                logger.info('=== Fold {}/{} ==='.format(fold_idx + 1, config.K_FOLDS))
                
                # Reset model weights for every fold
                if self.initial_load_path:
                    logger.info('Reloading initial model weights for fold {} from {}'.format(fold_idx + 1, self.initial_load_path))
                    self.load_model(self.initial_load_path)
                else:
                    logger.info('Resetting model weights to random for fold {}'.format(fold_idx + 1))
                    self.sess.run(tf.global_variables_initializer())
                
                # Always start from epoch 0 for each fold
                current_start_epoch = 0
                
                # 쓰레드 수만큼 Agent 클래스 생성 및 데이터 분배
                agents = []
                for i in range(self.threads):
                    # Distribute files among threads using slicing
                    agent_train_files = train_files[i::self.threads]
                    agent_val_files = val_files[i::self.threads] if val_files else None
                    
                    logger.info('Thread %d: allocated %d training images', i, len(agent_train_files))
                    
                    agents.append(Agent(self.action_size, self.state_size,
                                        [self.actor, self.critic], self.sess,
                                        self.optimizer, self.discount_factor,
                                        [self.summary_op, self.summary_placeholders,
                                         self.update_ops, self.summary_writer],
                                        train_path=train_path,
                                        train_files=agent_train_files, val_files=agent_val_files,
                                        start_epoch=current_start_epoch,
                                        current_fold=fold_idx,
                                        thread_id=i))

                # 각 쓰레드 시작
                for agent in agents:
                    time.sleep(1)
                    agent.start()
                
                # Start periodic save loop thread if not already running
                if not hasattr(self, 'save_thread') or not self.save_thread.is_alive():
                    self.save_thread = threading.Thread(target=self._periodic_save_loop)
                    self.save_thread.daemon = True
                    self.save_thread.start()
                    logger.info('Started periodic save thread.')

                # Wait for agents to finish their epochs for this fold
                for agent in agents:
                    agent.join()
                
                elapsed = time.time() - self.start_time
                logger.info('Fold {} completed. Total elapsed time: {:.2f}s ({:.2f}m)'.format(
                    fold_idx + 1, elapsed, elapsed / 60.0))
                
                # Reset start_epoch for subsequent folds
                start_epoch = 0
                
                # Save model after each fold
                now = datetime.now().strftime("%Y%m%d%H%M%S")
                logdir = "{}/A2RL_a3c_fold{}_{}".format(config.SAVE_MODEL_DIR, fold_idx + 1, now)
                metadata = {
                    'fold': fold_idx + 1,
                    'epoch': config.EPOCH_SIZE, # Fold completed
                    'episode': episode
                }
                self.save_model(logdir, metadata=metadata)
            
            logger.info('All folds completed.')
            return
        else:
            # Standard training mode
            agents = [Agent(self.action_size, self.state_size,
                            [self.actor, self.critic], self.sess,
                            self.optimizer, self.discount_factor,
                             [self.summary_op, self.summary_placeholders,
                              self.update_ops, self.summary_writer],
                             train_path=train_path,
                             start_epoch=start_epoch,
                             thread_id=i)
                       for i in range(self.threads)]

            # 각 쓰레드 시작
            for agent in agents:
                time.sleep(1)
                agent.start()
            
            # Start periodic save loop thread if not already running
            if not hasattr(self, 'save_thread') or not self.save_thread.is_alive():
                self.save_thread = threading.Thread(target=self._periodic_save_loop)
                self.save_thread.daemon = True
                self.save_thread.start()
                logger.info('Started periodic save thread.')

            # Wait for agents to finish
            for agent in agents:
                agent.join()

    def _periodic_save_loop(self):
        """
        Periodic model saving loop. Runs in a separate thread.
        Uses graph and session pinning to ensure thread safety in TF1/Keras.
        """
        while True:
            time.sleep(60 * config.SAVE_INTERVAL_MINUTES)
            logdir = None
            try:
                with a3c_graph.as_default():
                    with self.sess.as_default():
                        now = datetime.now().strftime("%Y%m%d%H%M%S")
                        logger.info('Periodic model save initiated: %s', now)
                        root_logdir = config.SAVE_MODEL_DIR
                        logdir = "{}/A2RL_a3c_periodic_{}".format(root_logdir, now)
                        
                        metadata = {
                            'fold': 0,
                            'epoch': 0,
                            'episode': episode,
                            'status': 'periodic'
                        }
                        self.save_model(logdir, metadata=metadata)
                        
                        # Explicitly flush summaries to ensure they are written to Drive
                        if hasattr(self, 'summary_writer'):
                            self.summary_writer.flush()
                            logger.info('TensorBoard summaries flushed to disk.')
                        
                        logger.info('Periodic model saved successfully: %s', logdir)
            except Exception as e:
                logger.error("Error in periodic save thread: %s", str(e), exc_info=True)

    # Save Final Model
    def save_final_model(self):
        time_str = datetime.now().strftime("%H%M%S")
        final_model_dir = os.path.join(config.SAVE_MODEL_DIR, "final_model_{}".format(time_str))
        self.save_model(final_model_dir, metadata={'episode': episode, 'status': 'final'})
        logger.info("Final model saved to: %s", final_model_dir)

    # 정책신경망과 가치신경망을 생성
    def build_model(self):

        logger.debug('Parent Model default graph: %s', tf.get_default_graph())
        if hasattr(K, 'set_learning_phase'):
            try:
                K.set_learning_phase(1)  # set learning phase
            except Exception:
                pass

        input = Input(shape = self.state_size )

        fc1 = Dense(1024, activation = 'relu') (input)
        #drop1 = Dropout(drop_ratio)(fc1)

        fc2 = Dense(1024, activation='relu')(fc1)
        #drop2 = Dropout(drop_ratio)(fc2)

        fc3 = Dense(1024, activation='relu')(fc2)
        #drop3 = Dropout(drop_ratio)(fc3)

        lstm1 = LSTM(1024)( fc3)
        #drop4 = Dropout(drop_ratio)(lstm1)

        policy = Dense(self.action_size, activation='softmax')(lstm1)
        value = Dense(1, activation='linear')(lstm1)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        # 가치와 정책을 예측하는 함수를 만들어냄
        if hasattr(actor, '_make_predict_function'):
            actor._make_predict_function()
        if hasattr(critic, '_make_predict_function'):
            critic._make_predict_function()

        # actor.summary()
        # critic.summary()

        return actor, critic



    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        # Even in Keras 3, K.placeholder is often safer for Keras Functional API
        # but if it fails, we use tf.placeholder and ensure it's compatible.
        if not USING_TF_KERAS:
            # In Keras 3 / TF 2.19, standard placeholders with Dimension(None) cause errors.
            # Using Keras Input yields compatible KerasTensors.
            action = Input(shape=(self.action_size,), name='action_input')
            # For advantages, (None, 1) is often more robust than (None,) in Keras 3.
            advantages_raw = Input(shape=(1,), name='advantages_input')
            advantages = K.flatten(advantages_raw)
            # Re-call the actor on its input to get a fresh symbolic link for Keras 3 tracing.
            policy = self.actor(self.actor.inputs[0])
        else:
            try:
                action = K.placeholder(shape=[None, self.action_size])
                advantages = K.placeholder(shape=[None, ])
            except (AttributeError, TypeError):
                action = tf.placeholder(dtype=global_dtype, shape=[None, self.action_size])
                advantages = tf.placeholder(dtype=global_dtype, shape=[None, ])
            policy = self.actor.output

        # In Keras 3, the order matters when mixing KerasTensors and raw Tensors/Placeholders.
        # Putting policy (KerasTensor) on the left allows it to handle the operation correctly.
        action_prob = K.sum(policy * action, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        
        if config.ENABLE_MINI_BATCH:
            cross_entropy = -K.mean(cross_entropy)
        else:
            cross_entropy = -K.sum(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        
        if config.ENABLE_MINI_BATCH:
            entropy = K.mean(entropy)
        else:
            entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + self.beta * entropy   # beta is 0.05

        # Use dynamic learning rate argument name for compatibility
        opt_kwargs = {'rho': 0.99, 'epsilon': 0.01}
        opt_kwargs[K_LR_NAME] = self.actor_lr
        optimizer = RMSprop(**opt_kwargs)
        updates = optimizer.get_updates(loss, self.actor.trainable_weights)
        
        # Combine model inputs with custom inputs
        actual_inputs = self.actor.inputs + [action, (advantages_raw if not USING_TF_KERAS else advantages)]
        train = K.function(actual_inputs, [loss], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        if not USING_TF_KERAS:
            # Use Keras Input for compatible KerasTensor shape in Keras 3
            # Input(shape=(1,)) results in (None, 1)
            discounted_raw = Input(shape=(1,), name='discounted_prediction_input')
            discounted_prediction = K.flatten(discounted_raw)
            # Re-call the critic on its input to get a fresh symbolic link for Keras 3 tracing.
            value = self.critic(self.critic.inputs[0])
        else:
            try:
                discounted_prediction = K.placeholder(shape=(None,))
            except (AttributeError, TypeError):
                discounted_prediction = tf.placeholder(dtype=global_dtype, shape=(None,))
            value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        # Flip order for Keras 3 compatibility (KerasTensor - Placeholder)
        loss = K.mean(K.square(value - discounted_prediction))

        # Use dynamic learning rate argument name for compatibility
        opt_kwargs = {'rho': 0.99, 'epsilon': 0.01}
        opt_kwargs[K_LR_NAME] = self.critic_lr
        optimizer = RMSprop(**opt_kwargs)
        updates = optimizer.get_updates(loss, self.critic.trainable_weights)
        
        # Combine model inputs with custom inputs
        actual_inputs = self.critic.inputs + [(discounted_raw if not USING_TF_KERAS else discounted_prediction)]
        train = K.function(actual_inputs, [loss], updates=updates)
        return train

    def load_model(self, name):
        logger.info('Loading model from: %s', name)
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")
        self._weights_loaded = True
        
        metadata_path = name + "_metadata.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info('Loaded metadata: %s', metadata)
                return metadata
            except Exception as e:
                logger.error('Failed to load metadata: %s', e)
        return None

    def evaluate(self, test_files):
        """
        Evaluate the trained model on a list of test files and compare scores.
        """
        logger.info("Starting post-training evaluation on %d files", len(test_files))
        results = []
        
        # Create a single Agent for evaluation
        eval_agent = Agent(self.action_size, self.state_size,
                          [self.actor, self.critic], self.sess,
                          self.optimizer, self.discount_factor,
                          [self.summary_op, self.summary_placeholders,
                           self.update_ops, self.summary_writer])
        
        for i, filepath in enumerate(test_files):
            logger.info("Evaluating [%d/%d]: %s", i+1, len(test_files), filepath)
            img, error = load_and_validate_image(filepath)
            if error:
                logger.warning("Skipping %s: %s", filepath, error)
                continue
                
            report = eval_agent.evaluate_cropping(img, filepath)
            results.append(report)
            
        if not results:
            logger.error("No images were successfully evaluated.")
            return
            
        # Summary Report
        avg_initial = np.mean([r['initial_score'] for r in results])
        avg_final = np.mean([r['final_score'] for r in results])
        avg_improvement = np.mean([r['improvement'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        
        logger.info("\n" + "="*60 +
                    "\nEvaluation Summary Report:" +
                    "\n  Total Images: {}".format(len(results)) +
                    "\n  Avg Initial Score: {:.4f}".format(avg_initial) +
                    "\n  Avg Final Score:   {:.4f}".format(avg_final) +
                    "\n  Avg Improvement:   {:.4f} ({:+.2f}%)".format(
                        avg_improvement, (avg_improvement/abs(avg_initial)*100 if avg_initial != 0 else 0)) +
                    "\n  Avg Steps:         {:.1f}".format(avg_steps) +
                    "\n" + "="*60)

    def save_model(self, name, metadata=None):
        with a3c_graph.as_default():
            with self.sess.as_default():
                # Ensure the directory exists (important for date-based directories)
                dir_name = os.path.dirname(name)
                if dir_name and not os.path.exists(dir_name):
                    try:
                        os.makedirs(dir_name)
                        logger.info("Created model save directory: %s", dir_name)
                    except OSError as e:
                        logger.error("Failed to create directory %s: %s", dir_name, e)
        
                logger.info('Saving model to: %s', name)
                try:
                    self.actor.save_weights(name + "_actor.h5")
                    logger.debug('Saved actor weights.')
                    self.critic.save_weights(name + "_critic.h5")
                    logger.debug('Saved critic weights.')
                except Exception as e:
                    logger.error('Failed to save model weights: %s', e, exc_info=True)
                
                if metadata:
                    metadata_path = name + "_metadata.json"
                    try:
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=4)
                        logger.info('Saved metadata to: %s', metadata_path)
                    except Exception as e:
                        logger.error('Failed to save metadata: %s', e)
                
                # Ensure summaries are flushed whenever model is saved
                if hasattr(self, 'summary_writer'):
                    self.summary_writer.flush()

    def cleanup(self):
        """
        Clean up resources before program exit.
        
        This method should be called before the program terminates to ensure
        all data is properly flushed to disk.
        """
        logger.info('='*60)
        logger.info('Starting resource cleanup...')
        logger.info('='*60)
        
        # Flush and close summary writer
        if hasattr(self, 'summary_writer') and self.summary_writer is not None:
            try:
                logger.info('Flushing summary writer...')
                self.summary_writer.flush()
                logger.info('✓ Summary writer flushed')
                
                logger.info('Closing summary writer...')
                self.summary_writer.close()
                logger.info('✓ Summary writer closed')
            except Exception as e:
                logger.error('✗ Error closing summary writer: %s', e)
        else:
            logger.info('Summary writer not available')
        
        # Close TensorFlow session
        if hasattr(self, 'sess') and self.sess is not None:
            try:
                logger.info('Closing TensorFlow session...')
                self.sess.close()
                logger.info('✓ TensorFlow session closed')
            except Exception as e:
                logger.error('✗ Error closing session: %s', e)
        else:
            logger.info('Session not available')
        
        logger.info('='*60)
        logger.info('Cleanup completed')
        logger.info('='*60)

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        """
        Setup summary placeholders and ops. 
        Note: We now use manual summary creation in Agent._process_single_image 
        for better thread safety and reliability in TF1/Colab.
        """
        return [], [], None


# 액터러너 클래스(쓰레드)
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                 optimizer, discount_factor, summary_ops, train_path=None, train_files=None, val_files=None,
                 start_epoch=0, current_fold=None, thread_id=0):
        threading.Thread.__init__(self)

        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops
        
        self.train_path = train_path
        self.train_files = train_files
        self.val_files = val_files
        self.start_epoch = start_epoch
        self.current_fold = current_fold
        self.thread_id = thread_id

        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards = [], [], []

        # 로컬 모델 생성
        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # 모델 업데이트 주기
        self.t_max = config.UPDATE_FREQ

        self.t = 0

        self.T_max = config.T_MAX

        self.step_penalty = config.STEP_PENALTY

        self.epoch_size   = config.EPOCH_SIZE   # 20
        self.train_size = config.TRAIN_SIZE
        self. batch_size = config.BATCH_SIZE    #32

        # Error tracking
        self.failed_images = []
        self.total_images_processed = 0
        self.total_images_failed = 0
        
        # Batch index for sequential sampling
        self.current_batch_index = 0
        
        # Action tracking statistics
        self.action_counts = np.zeros(self.action_size, dtype=np.int32)  # Cumulative across all episodes
        self.episode_action_counts = np.zeros(self.action_size, dtype=np.int32)  # Per-episode counts
        self.episode_action_history = []  # Stores action sequence for current episode

        # Feature scaling setup
        self.enable_feature_scaling = config.ENABLE_FEATURE_SCALING
        self.feature_scaling_method = config.FEATURE_SCALING_METHOD
        self.feature_epsilon = config.FEATURE_EPSILON
        self.feature_stats = None
        
        if self.enable_feature_scaling and self.feature_scaling_method == 'global':
            # Load pre-computed statistics for global scaling
            self.feature_stats = load_feature_stats(config.FEATURE_STATS_PATH)
            if self.feature_stats is None:
                logger.warning('Global scaling enabled but stats not available, will use per-sample scaling')

        # VFN Preload

        #with open('vfn_rl.pkl', 'rb') as f:
            #self.var_dict = pickle.load(f)





    #  This is the      definition      of      helper      function

    def _load_and_validate_batch(self, TrainPath, batch_size, use_random_sampling=False, file_list=None):
        """
        Load and validate a batch of images.
        
        Args:
            TrainPath: Path to training images folder
            batch_size: Number of images to load
            use_random_sampling: If True, use random sampling. If False, use sequential sampling.
            file_list: Optional list of files to use. If None, reads from TrainPath.
        
        Returns:
            tuple: (images, images_filename) - lists of loaded images and their filenames
        """
        if file_list is not None:
            trainfiles = file_list
        else:
            trainfiles = [f for f in listdir(TrainPath) if isfile(join(TrainPath, f))]
            
        total_files = len(trainfiles)
        
        if total_files == 0:
            logger.warning('No files found for batch loading.')
            return [], []

        if use_random_sampling:
            # Random sampling
            num_to_sample = min(batch_size, total_files)
            rand_index = np.random.choice(total_files, size=num_to_sample, replace=False)
            logger.debug('Random sampling - indices: %s', rand_index)
            trainfiles_batch = [trainfiles[index] for index in rand_index]
        else:
            # Sequential sampling
            start_idx = self.current_batch_index * batch_size
            
            # Wrap around if we reach the end
            if start_idx >= total_files:
                self.current_batch_index = 0
                start_idx = 0
                
            end_idx = min(start_idx + batch_size, total_files)
            
            logger.debug('Sequential sampling - indices [%d, %d)', start_idx, end_idx)
            
            trainfiles_batch = trainfiles[start_idx:end_idx]
            self.current_batch_index += 1
        
        trainfiles_batch_fullname = [join(TrainPath, x) for x in trainfiles_batch]
        
        logger.debug('Batch selection: %s', trainfiles_batch_fullname)
        
        # Load and validate images
        images = []
        images_filename = []
        
        for i, x in enumerate(trainfiles_batch_fullname):
            if i % 10 == 0 or i == len(trainfiles_batch_fullname) - 1:
                logger.info('Thread %d - Loading image %d/%d...', self.thread_id, i + 1, len(trainfiles_batch_fullname))
            
            img, error = load_and_validate_image(x)
            
            if error:
                logger.warning('Skipping %s: %s', x, error)
                self.failed_images.append({'filename': x, 'error': error})
                self.total_images_failed += 1
                continue
            
            images.append(img)
            images_filename.append(x)
            self.total_images_processed += 1
        
        logger.info('Loaded %d valid images', len(images))
        return images, images_filename

    def _process_single_image(self, image, filename=None):
        """
        Process a single image through the training loop.
        
        Args:
            image: Image array to process
            filename: Optional filename for logging
        """
        global episode, episode_lock
        
        step = 0
        self.t = 0
        
        current_episode = 0
        with episode_lock:
            current_episode = episode
        
        logger.debug("Thread %d - Starting aesthetic score evaluation for image", self.thread_id)
        scores, features = evaluate_aesthetics_score([image])
        logger.debug("Thread %d - Evaluation completed (Score: %.4f)", self.thread_id, scores[0])
        if filename:
            logger.info('Aesthetic scores evaluation for %s: %s', filename, scores)
        else:
            logger.info('Aesthetic scores evaluation: %s', scores)
        
        logger.debug('Feature shape: %s', features[0].shape)
        
        batch_size = 1
        terminals = np.zeros(batch_size)
        ratios = np.repeat([[0, 0, 20, 20]], batch_size, axis=0)
        
        if self.t == 0:
            global_score = scores[0]
            global_feature = features[0]
        
        logger.info('Global score: %.4f', global_score)
        
        # Filter out images with already high initial scores
        if hasattr(config, 'INITIAL_SCORE_THRESHOLD') and global_score >= config.INITIAL_SCORE_THRESHOLD:
            logger.info('Skipping image %s: Initial score %.4f exceeds threshold %.4f', 
                        filename, global_score, config.INITIAL_SCORE_THRESHOLD)
            return
        
        score = 0
        done = False
        
        # Reset episode action history and counts for new episode
        self.episode_action_history = []
        self.episode_action_counts = np.zeros(self.action_size, dtype=np.int32)
        
        while step < self.T_max and not done:  # T_max = 50
            
            if self.t == 0:
                local_feature = global_feature
                local_score = global_score
            else:
                local_feature = new_features[0]
                local_score = new_scores[0]
            
            observe = np.concatenate((global_feature, local_feature), axis=1)
            logger.debug('Observe shape: %s', observe.shape)
            
            # Apply feature scaling if enabled
            if self.enable_feature_scaling:
                observe_original = observe.copy()  # Keep for diagnostics
                observe = apply_feature_scaling(
                    observe, 
                    method=self.feature_scaling_method,
                    stats=self.feature_stats,
                    epsilon=self.feature_epsilon
                )
                
                # Log scaling effect on first step or periodically
                if step == 0 or (episode % 10 == 0 and step < 3):
                    logger.info("Feature Scaling Applied (Episode %d, Step %d):", episode, step)
                    logger.info("  Method: %s", self.feature_scaling_method)
                    logger.info("  Before - Mean: %.6f, Std: %.6f", 
                               np.mean(observe_original), np.std(observe_original))
                    logger.info("  After  - Mean: %.6f, Std: %.6f", 
                               np.mean(observe), np.std(observe))
            
            # Feature scale diagnostics
            if step == 0 or (episode % 10 == 0 and step < 3):  # First step or first 3 steps every 10 episodes
                feature_mean = np.mean(observe)
                feature_std = np.std(observe)
                feature_min = np.min(observe)
                feature_max = np.max(observe)
                feature_abs_max = np.max(np.abs(observe))
                
                logger.info("Feature Statistics (Episode %d, Step %d):", episode, step)
                logger.info("  Mean: %.6f, Std: %.6f", feature_mean, feature_std)
                logger.info("  Min: %.6f, Max: %.6f", feature_min, feature_max)
                logger.info("  Abs Max: %.6f, Range: %.6f", feature_abs_max, feature_max - feature_min)
                
                # Global vs Local feature comparison
                global_mean = np.mean(global_feature)
                global_std = np.std(global_feature)
                local_mean = np.mean(local_feature)
                local_std = np.std(local_feature)
                
                logger.info("  Global Feature - Mean: %.6f, Std: %.6f", global_mean, global_std)
                logger.info("  Local Feature  - Mean: %.6f, Std: %.6f", local_mean, local_std)
            
            history = np.expand_dims(observe, axis=0)
            
            logger.debug('History shape: %s. History: %s', history.shape, history)
            
            action, policy = self.get_action(history)
            
            # Track action (both cumulative and per-episode)
            self.episode_action_history.append(action)
            self.action_counts[action] += 1  # Cumulative count
            self.episode_action_counts[action] += 1  # Episode count
            
            # Policy distribution diagnostics
            if step == 0 or (episode % 10 == 0 and step < 3):
                policy_entropy = -np.sum(policy * np.log(policy + 1e-10))
                policy_max = np.max(policy)
                policy_max_idx = np.argmax(policy)
                
                logger.info("Policy Statistics (Episode %d, Step %d):", episode, step)
                logger.info("  Entropy: %.4f (Ideal: ~2.6, Problem: <1.0)", policy_entropy)
                logger.info("  Max Prob: %.4f at Action %d (%s)", 
                           policy_max, policy_max_idx, get_action_name(policy_max_idx))
                
                # Show top 3 actions
                top3_indices = np.argsort(policy)[-3:][::-1]
                top3_str = ', '.join(['A{}({}):{:.3f}'.format(i, get_action_name(i), policy[i]) 
                                     for i in top3_indices])
                logger.info("  Top 3 Actions: %s", top3_str)
            
            action_name = get_action_name(action)
            logger.info('Step %d - Action: %d (%s), Policy sum: %.4f', step, action, action_name, np.sum(policy))
            logger.debug('Policy: %s', policy)
            
            if action == 13:
                if step < config.MIN_STEPS:
                    done = False
                    logger.info('Action 13 (STOP) ignored (step %d < %d)', step, config.MIN_STEPS)
                else:
                    done = True
                    logger.info('Episode termination action (13) received.')
            else:
                logger.debug('Terminals: %s', terminals)
                terminals[0] = 0
            
            # Generate bounding box
            ratios, terminals = command2action([action], ratios, terminals)
            
            # FIX: If we forced continue (ignored STOP), we must reset terminal flag
            if action == 13 and step < config.MIN_STEPS:
                terminals[0] = 0
            
            logger.debug('Ratios: %s', ratios)
            
            im = image.astype(np.float32) / 255 - 0.5
            bbox = generate_bbox([im], ratios)
            
            # New Cropped Image
            img = crop_input([im], bbox)
            
            # Score, Feature of newly Cropped image
            new_scores, new_features = evaluate_aesthetics_score_resized(img)
            
            logger.info('New scores: %s', new_scores)
            
            # Calculate reward
            score_diff = new_scores[0] - local_score
            
            if action == 13:
                # Penalty for stopping too early
                if step < config.MIN_STEPS:
                    reward = -1.0
                    logger.info('Action 13 (STOP) triggered too early (step %d < %d) - Penalty: %.4f', 
                                step, config.MIN_STEPS, reward)
                else:
                    reward = 0.0
                    logger.info('Action 13 (STOP) - Reward set to 0')
            else:
                # Use continuous reward instead of sign() for more granular feedback
                # Multiplied by 5 to make the signal stronger
                reward = (score_diff * 5.0) - self.step_penalty * (self.t + 1)
                
                # Check Aspect Ratio with validation
                is_valid, asratio, penalty = validate_aspect_ratio(bbox)
                
                if not is_valid:
                    logger.warning('Invalid aspect ratio: %.2f (penalty: %.1f)', asratio, penalty)
                    reward = reward - penalty
                else:
                    logger.debug('Valid aspect ratio: %.2f', asratio)
            
            logger.info('Reward: %.4f', reward)
            
            # Policy maximum
            self.avg_p_max += np.amax(self.actor.predict(np.float32(history)))
            
            score += reward
            
            # Save sample
            self.append_sample(history, action, reward)
            logger.debug('Step %d reward: %.4f', self.t, reward)
            
            step += 1
            self.t += 1
            
            if step == self.T_max:
                done = True
            
            # Train model
            if self.t % self.t_max or done:
                self.train_model(done)
                self.update_local_model()
            
            if done:
                # Record episode statistics
                # (episode increment handled below with lock)
                
                # Log episode action history
                action_history_str = ' -> '.join(['{}({})'.format(a, get_action_name(a)) for a in self.episode_action_history])
                logger.info("Thread %d - Episode %d finished - Score: %.4f, Steps: %d, Avg Prob Max: %.4f", 
                           self.thread_id, episode, score, step, self.avg_p_max / float(step))
                logger.info("Action History: [%s]", action_history_str)
                
                # Log episode-specific action counts
                episode_total = np.sum(self.episode_action_counts)
                if episode_total > 0:
                    ep_lines = ["Thread {} - Episode {} Action Distribution:".format(self.thread_id, episode)]
                    for i in range(self.action_size):
                        count = self.episode_action_counts[i]
                        if count > 0:  # Only show actions that were used
                            percentage = 100.0 * count / episode_total
                            action_name = get_action_name(i)
                            ep_lines.append("  Action {:2d} ({:15s}): {:2d} times ({:5.1f}%)".format(
                                i, action_name, count, percentage))
                    logger.info("\n" + "\n".join(ep_lines))
                
                # Print cumulative action statistics periodically
                if episode % 10 == 0:
                    total_actions = np.sum(self.action_counts)
                    if total_actions > 0:
                        cum_lines = ["="*60,
                                     "Action Statistics (CUMULATIVE for Thread {}, Total Episode {}):".format(self.thread_id, episode)]
                        for i in range(self.action_size):
                            count = self.action_counts[i]
                            percentage = 100.0 * count / total_actions
                            action_name = get_action_name(i)
                            cum_lines.append("  Action {:2d} ({:15s}): {:6d} times ({:6.2f}%)".format(
                                i, action_name, count, percentage))
                        cum_lines.append("  Total Actions: {}".format(total_actions))
                        cum_lines.append("="*60)
                        logger.info("\n" + "\n".join(cum_lines))
                
                if terminals[0] == 1 or step >= self.T_max:
                    # 에피소드 종료시 글로벌 에피소드 카운트 증가를 Thread-Safe하게 처리
                    with episode_lock:
                        episode += 1
                
                # Print error statistics periodically
                if episode % 100 == 0:
                    total_processed = self.total_images_processed + self.total_images_failed
                    if total_processed > 0:
                        success_rate = (self.total_images_processed / float(total_processed)) * 100
                        logger.info("\n" + "="*60 +
                                   "\nImage Processing Statistics (Episode {}):".format(episode) +
                                   "\n  Total processed: {}".format(self.total_images_processed) +
                                   "\n  Total failed: {}".format(self.total_images_failed) +
                                   "\n  Success rate: {:.2f}%".format(success_rate) +
                                   "\n" + "="*60)
                
                # Manual summary creation for better reliability in Colab/Multi-threading
                # This bypasses session.run for summaries, avoiding graph/session pinning issues.
                if step > 0:
                    summary = tf.Summary()
                    summary.value.add(tag='Total Reward/Episode', simple_value=float(score))
                    summary.value.add(tag='Average Max Prob/Episode', simple_value=float(self.avg_p_max / float(step)))
                    summary.value.add(tag='Duration/Episode', simple_value=float(step))
                    
                    with episode_lock:
                        current_episode = episode
                    
                    self.summary_writer.add_summary(summary, current_episode + 1)
                    self.summary_writer.flush()
                else:
                    logger.warning("Thread %d - Episode finished with 0 steps, skipping summary.", self.thread_id)

                self.avg_p_max = 0
                self.avg_loss = 0
                step = 0

    def train_episode(self, TrainPath, num_batches=1, verbose=True, use_random_sampling=False, file_list=None):
        """
        Unified training function that replaces train_() and train9000_().
        
        Args:
            TrainPath: Path to training images folder
            num_batches: Number of batches to process
            verbose: Whether to print filenames during processing
            use_random_sampling: If True, use random sampling. If False, use sequential sampling.
            file_list: Optional list of files to use.
        """
        global episode, global_dtype, a3c_graph
        
        for batch_idx in range(num_batches):
            batch_start = time.time()
            if num_batches > 1:
                logger.info('=== Batch %d/%d ===', batch_idx + 1, num_batches)
            
            # Load and validate batch
            logger.info('Thread %d - Loading batch %d...', self.thread_id, batch_idx + 1)
            images, images_filename = self._load_and_validate_batch(
                TrainPath, self.batch_size, use_random_sampling, file_list=file_list)
            logger.info('Thread %d - Batch %d loaded (images: %d)', self.thread_id, batch_idx + 1, len(images))
            
            # Process each image
            for j in range(len(images)):
                filename = images_filename[j] if verbose else None
                self._process_single_image(images[j], filename)
            
            batch_duration = time.time() - batch_start
            logger.info('Batch %d/%d completed in %.2f seconds', batch_idx + 1, num_batches, batch_duration)

    def validate_episode(self, TrainPath, file_list, verbose=True):
        """
        Run episodes on validation set without training.
        """
        logger.info('=== Validation Phase ===')
        # Reset current_batch_index for validation if it was used sequentially
        old_batch_index = self.current_batch_index
        self.current_batch_index = 0
        
        # Calculate number of batches for validation
        num_batches = (len(file_list) + self.batch_size - 1) // self.batch_size
        
        # Temporarily override train_model to do nothing
        original_train_model = self.train_model
        self.train_model = lambda done: None
        
        try:
            for batch_idx in range(num_batches):
                images, images_filename = self._load_and_validate_batch(
                    TrainPath, self.batch_size, use_random_sampling=False, file_list=file_list)
                
                for j in range(len(images)):
                    filename = images_filename[j] if verbose else None
                    self._process_single_image(images[j], filename)
        finally:
            # Restore original train_model
            self.train_model = original_train_model
            self.current_batch_index = old_batch_index
        
        logger.info('=== Validation Finished ===')

    #  This is the      definition      of      helper      function

    # DEPRECATED: Use train_episode() instead
    def train_ (self  , TrainPath ):
        """
        DEPRECATED: This function is kept for backward compatibility.
        Please use train_episode(TrainPath, num_batches=1, verbose=False) instead.
        """
        logger.warning("train_() is deprecated. Use train_episode() instead.")
        return self.train_episode(TrainPath, num_batches=1, verbose=False)


    # DEPRECATED: Use train_episode() instead
    def train9000_(self, TrainPath ):
        """
        DEPRECATED: This function is kept for backward compatibility.
        Please use train_episode(TrainPath, num_batches=281, verbose=True) instead.
        """
        logger.warning("train9000_() is deprecated. Use train_episode() instead.")
        return self.train_episode(TrainPath, num_batches=281, verbose=True)

    def evaluate_cropping(self, image, filename=None):
        """
        Run the agent on an image and return before/after scores.
        """
        step = 0
        ratios = np.array([[0, 0, 20, 20]])
        terminals = np.array([0])
        
        # Initial scoring and feature extraction (Global)
        initial_scores, initial_features = evaluate_aesthetics_score([image])
        initial_score = initial_scores[0]
        global_feature = initial_features[0]
        
        current_score = initial_score
        local_feature = global_feature
        
        current_image = image
        done = False
        
        while step < self.T_max and not done:
            # Prepare state: concatenate global and local features (2000-dim)
            observe = np.concatenate((global_feature, local_feature), axis=1)
            
            # Apply feature scaling (same as training)
            if self.enable_feature_scaling:
                observe_original = observe.copy()
                observe = apply_feature_scaling(
                    observe,
                    method=self.feature_scaling_method,
                    stats=self.feature_stats,
                    epsilon=self.feature_epsilon
                )
                
                # Info logging for scaled features (first step of evaluation)
                if step == 0:
                    logger.info("Eval Step 0 - Scaling (%s): Before(mean=%.4f, std=%.4f), After(mean=%.4f, std=%.4f)",
                                self.feature_scaling_method, np.mean(observe_original), np.std(observe_original),
                                np.mean(observe), np.std(observe))
            
            history = np.expand_dims(observe, axis=0) # [1, 2000]
            
            # Get action from actor
            policy = self.actor.predict(history)[0]
            action_index = np.argmax(policy)
            
            # Diagnostic: Log policy every step during evaluation
            policy_entropy = -np.sum(policy * np.log(policy + 1e-10))
            top3_indices = np.argsort(policy)[-3:][::-1]
            top3_str = ', '.join(['A{}({}):{:.3f}'.format(i, get_action_name(i), policy[i]) 
                                 for i in top3_indices])
            
            logger.info("Step %d - Action: %d (%s), Entropy: %.4f, Top 3: %s", 
                        step, action_index, get_action_name(action_index), policy_entropy, top3_str)
            
            logger.debug("Step %d: action=%d, policy_sum=%.4f, policy[13]=%.4f", 
                         step, action_index, np.sum(policy), policy[13])
            
            # Action 13 is the terminal action in this environment
            # Action 13 is the terminal action in this environment
            if action_index == 13:
                if step < config.MIN_STEPS:
                    logger.info("Agent chose STOP action (13) at step %d - IGNORED (step < MIN_STEPS)", step)
                else:
                    logger.info("Agent chose STOP action (13) at step %d", step)
                    done = True
                    break
                
            # Update ratios and terminals based on action
            ratios, terminals = command2action([action_index], ratios, terminals)
            
            # CRITICAL FIX: If we forced continue (ignored STOP), we must reset the terminal flag
            if action_index == 13 and step < config.MIN_STEPS:
                 terminals[0] = 0
            
            if terminals[0] == 1:
                logger.info("Terminal condition met via actions at step %d", step)
                done = True
                break
            
            # Generate bounding box (FIX: was missing!)
            im = image.astype(np.float32) / 255.0 - 0.5
            bbox = generate_bbox([im], ratios)
            
            # crop_input returns resized (227, 227) images
            cropped_batch = crop_input([im], bbox)
            
            # Update local score and feature for next step
            # evaluate_aesthetics_score_resized expects resized, normalized images
            new_scores, new_features = evaluate_aesthetics_score_resized(cropped_batch)
            
            current_score = new_scores[0]
            local_feature = new_features[0]
            step += 1
            
        final_score = current_score
        
        report = {
            'filename': filename,
            'initial_score': initial_score,
            'final_score': final_score,
            'improvement': final_score - initial_score,
            'steps': step
        }
        
        logger.info("Evaluation for %s: Initial: %.4f -> Final: %.4f (Diff: %+.4f, Steps: %d)",
                    filename, initial_score, final_score, final_score - initial_score, step)
        return report

    def run(self):
        with a3c_graph.as_default():
            with self.sess.as_default():
                if hasattr(K, 'set_session'):
                    try:
                        K.set_session(self.sess)
                    except Exception: pass
                try:
                    self._run()
                except Exception as e:
                    logger.error("Error in thread {}: {}".format(self.thread_id, str(e)), exc_info=True)

    def _run(self):
        # Original run logic moved to _run to keep the wrapper clean
        global episode
        global global_dtype
        global a3c_graph

        for epoch_step in range(self.start_epoch, self.epoch_size):
            logger.info('Thread %d - Epoch step: %d', self.thread_id, epoch_step)
            TrainPath = self.train_path if self.train_path else config.TRAIN_PATH
            
            # Use current fold's train files or default to full path
            train_list = self.train_files
            
            # If standard mode (no pre-assigned files), load all from TrainPath
            if train_list is None:
                from os import listdir
                from os.path import isfile, join
                all_files = [f for f in listdir(TrainPath) if isfile(join(TrainPath, f))]
                train_list = [f for f in all_files if any(f.lower().endswith(ext) for ext in config.VALID_IMAGE_EXTENSIONS)]
            
            # Shuffle the dataset at the start of each epoch for better generalization
            if train_list is not None:
                random.shuffle(train_list)
                logger.info('Thread %d: Shuffled training dataset for epoch %d', self.thread_id, epoch_step)
            
            # num_batches processes images in batches
            # Calculate specifically based on current list size
            num_batches = (len(train_list) + self.batch_size - 1) // self.batch_size
            
            self.train_episode(TrainPath, num_batches=num_batches, verbose=True, 
                               use_random_sampling=False, file_list=train_list)
            
            # Validation phase
            if (epoch_step + 1) % config.VALIDATION_FREQ == 0:
                val_list = self.val_files
                if val_list:
                    self.validate_episode(TrainPath, val_list, verbose=True)
            
            # Save periodic checkpoint including metadata
            # Only one thread needs to handle periodic saving to avoid conflicts
            # However, the original code has a separate loop in A3CAgent.train
            # But A3CAgent.train waits if K-fold is used (agent.join()).
            # So the periodic save loop in A3CAgent.train won't run until ALL folds are done if we use agent.join().
            # Fix: In K-fold mode, we should handle periodic saving differently or allow it in Agent.run.

        #sys.exit(0)

    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(
                self.states[-1] ))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # 정책신경망과 가치신경망을 업데이트
    def train_model(self, done):
        logger.debug('Model training started')
        discounted_prediction = self.discounted_prediction(self.rewards, done)

        states = np.zeros((len(self.states), 1, 2000))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states )

        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        # Mini-Batch Logic: Buffer data instead of immediate update
        if config.ENABLE_MINI_BATCH:
            if not hasattr(self, 'mini_batch_buffer'):
                self.mini_batch_buffer = {'inputs': [], 'actions': [], 'advantages': [], 'targets': []}
            
            try:
                # Append current chunk's data to buffer
                self.mini_batch_buffer['inputs'].append(states)
                self.mini_batch_buffer['actions'].append(np.vstack(self.actions))
                self.mini_batch_buffer['advantages'].append(advantages)
                self.mini_batch_buffer['targets'].append(discounted_prediction)
                
                # Check if we have enough data (chunks)
                if len(self.mini_batch_buffer['advantages']) >= config.MINI_BATCH_SIZE:
                    # Concatenate all buffered data
                    batch_inputs = np.concatenate(self.mini_batch_buffer['inputs'], axis=0)
                    batch_actions = np.concatenate(self.mini_batch_buffer['actions'], axis=0)
                    batch_advantages = np.concatenate(self.mini_batch_buffer['advantages'], axis=0)
                    batch_targets = np.concatenate(self.mini_batch_buffer['targets'], axis=0)
                    
                    # Perform ONE large update
                    self.optimizer[0]([batch_inputs, batch_actions, batch_advantages])
                    self.optimizer[1]([batch_inputs, batch_targets])
                    
                    self.avg_loss += 0 
                    
                    # Clear buffer immediately after training to free memory
                    self.mini_batch_buffer = {'inputs': [], 'actions': [], 'advantages': [], 'targets': []}
            except Exception as e:
                logger.error("Error during mini-batch training: {}".format(str(e)))
                # Clear buffer on error to prevent corrupted state and memory growth
                self.mini_batch_buffer = {'inputs': [], 'actions': [], 'advantages': [], 'targets': []}
                raise e
        else:
            # Standard immediate update
            self.optimizer[0]([states, np.vstack(self.actions), advantages])
            self.optimizer[1]([states, discounted_prediction])

        self.states, self.actions, self.rewards = [], [], []

        logger.debug('Model training completed')

    # 로컬신경망을 생성하는 함수
    def build_local_model(self):
        with a3c_graph.as_default():
            with self.sess.as_default():
                logger.debug('Child build_local_model graph: %s', tf.get_default_graph())
        K.set_learning_phase(1)  # set learning phase

        input = Input(shape=self.state_size)

        #input = Input(shape=( 1 , 2000  ))
        fc1 = Dense(1024, activation = 'relu') (input)
        #drop1 = Dropout(drop_ratio)(fc1)

        fc2 = Dense(1024, activation='relu')(fc1)
        #drop2 = Dropout(drop_ratio)(fc2)

        fc3 = Dense(1024, activation='relu')(fc2)
        #drop3 = Dropout(drop_ratio)(fc3)

        lstm1 = LSTM(1024)( fc3)
        #drop4 = Dropout(drop_ratio)(lstm1)

        policy = Dense(self.action_size, activation='softmax')(lstm1)
        value = Dense(1, activation='linear')(lstm1)


        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        # local_actor.summary()
        # local_critic.summary()

        return local_actor, local_critic




    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택


    def get_action_pretest(self, history):
        # 이미 정규화 in evaluate_aesthetics_score
        #history = np.float32(history / 255.)

        action_array = [0, 1, 2, 3, 4, 9, 10 ]

        choice  = np.random.choice( action_array , 1)
        logger.debug('Pre-test choice: %s', choice)
        action_index = choice[0]
        logger.debug('Pre-test get_action index: %d', action_index)
        with a3c_graph.as_default():
            with self.sess.as_default():
                # predict_on_batch is more thread-safe and direct than predict()
                policy = self.local_actor.predict_on_batch(history)[0]

        logger.debug('Pre-test policy: %s', policy)
        #print(' pre get_action 3 ', np.argmax(policy) )
        #action_index = np.argmax(policy)


        return action_index, policy


    def get_action(self, history):
        # 이미 정규화 in evaluate_aesthetics_score
        #history = np.float32(history / 255.)
        logger.debug('Getting action...')
        with a3c_graph.as_default():
            with self.sess.as_default():
                # predict_on_batch is more thread-safe and direct than predict()
                policy = self.local_actor.predict_on_batch(history)[0]

        logger.debug('Action policy predicted.')
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # 샘플을 저장
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe

if __name__ == "__main__":
    script_start_time = time.time()
    parser = argparse.ArgumentParser(description='A2RL Training')
    parser.add_argument('--resume', type=str, help='Path to model snapshot for resumption (including metadata, fold, epoch)')
    parser.add_argument('--load_weights', type=str, help='Path to model weights to load as initial values (metadata ignored)')
    parser.add_argument('--evaluate', type=str, help='Path to model snapshot for evaluation (excluding extensions)')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess dataset (filter images by score)')
    parser.add_argument('--workers', type=int, help='Number of worker processes for preprocessing')
    args = parser.parse_args()

    # 입력 이미지
    batch_size = 1
    snapshot = config.MODEL_SNAPSHOT

    tf.reset_default_graph()
    embedding_dim = config.EMBEDDING_DIM
    ranking_loss = config.RANKING_LOSS
    net_data = np.load(config.ALEXNET_NPY, encoding='bytes', allow_pickle=True).item()
    image_placeholder = tf.placeholder(dtype=global_dtype, shape=[batch_size, 227, 227, 3])
    var_dict = nw.get_variable_dict(net_data)
    SPP = config.SPP
    pooling = config.POOLING
    with tf.variable_scope("ranker") as scope:
        feature_vec = nw.build_alexconvnet(image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
        score_func = nw.score(feature_vec)
    
    # load pre-trained model (VFN)
    # load pre-trained model (VFN)
    saver = tf.train.Saver(tf.global_variables())
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    vfn_sess = tf.Session(config=config_tf)
    vfn_sess.run(tf.global_variables_initializer())
    # Validate snapshot files (fix for 'not an sstable' / Git LFS issues)
    # TF checkpoint typically consists of .meta, .index, and .data-00000-of-00001
    snapshot_prefix = snapshot
    expected_extensions = ['.meta',  '.data-00000-of-00001']
    
    for ext in expected_extensions:
        fpath = snapshot_prefix + ext
        # Some TF versions use different naming conventions, check standard ones
        if not os.path.exists(fpath):
            # Try finding without exact match for .data part
            if ext.startswith('.data'):
                import glob
                matches = glob.glob(snapshot_prefix + '.data*')
                if matches:
                    fpath = matches[0]
                else:
                    logger.warning(f"Checkpoint component not found: {fpath}")
                    continue
            else:
                logger.warning(f"Checkpoint component not found: {fpath}")
                continue
                
        try:
            fsize = os.path.getsize(fpath)
            logger.info(f"Checking model file: {fpath} ({fsize/1024/1024:.2f} MB)")
            if fsize < 2048: # Less than 2KB implies Git LFS pointer or corruption
                logger.error(f"❌ CRITICAL ERROR: Model file {fpath} is too small ({fsize} bytes).")
                logger.error("It likely contains a Git LFS pointer instead of binary data.")
                logger.error("SOLUTION: Please install git-lfs and run 'git lfs pull', or upload models via Kaggle Datasets.")
                sys.exit(1)
        except OSError:
            pass

    logger.info(f"Restoring VFN model from snapshot: {snapshot}")
    try:
        saver.restore(vfn_sess, snapshot)
    except tf.errors.DataLossError:
        logger.error("❌ DataLossError during restore! The model file is likely corrupted or not a valid TF checkpoint.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error restoring model: {e}")
        raise e

    # Dataset Preprocessing
    if args.preprocess:
        logger.info("Starting dataset preprocessing...")
        preprocess_dataset(config.TRAIN_PATH, config.FILTERED_TRAIN_PATH, config.INITIAL_SCORE_THRESHOLD, num_workers=args.workers)
        if not args.evaluate: # If only preprocessing was requested, exit after completion
            logger.info("Preprocessing finished. Exit.")
            sys.exit(0)

    # Determine training path
    train_path = config.TRAIN_PATH
    if config.USE_FILTERED_DATA:
        if os.path.exists(config.FILTERED_TRAIN_PATH) and os.listdir(config.FILTERED_TRAIN_PATH):
            logger.info("Using filtered dataset from: %s", config.FILTERED_TRAIN_PATH)
            train_path = config.FILTERED_TRAIN_PATH
        else:
            logger.warning("USE_FILTERED_DATA is True but filtered path is empty or missing. Falling back to TRAIN_PATH.")

    global_agent = A3CAgent(action_size=config.ACTION_SIZE)
    
    start_fold = 0
    start_epoch = 0
    
    # Evaluation Mode
    if args.evaluate:
        logger.info("Evaluation mode triggered for model: %s", args.evaluate)
        global_agent.load_model(args.evaluate)
        
        # Determine test files
        eval_source = train_path
        all_files = [os.path.abspath(os.path.join(eval_source, f)) 
                    for f in os.listdir(eval_source) 
                    if os.path.isfile(os.path.join(eval_source, f))]
        all_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in config.VALID_IMAGE_EXTENSIONS)]
        random.shuffle(all_files)
        test_files = all_files[:100] # Evaluate 20 random images
        
        global_agent.evaluate(test_files)
        sys.exit(0)
    
    # Weight Loading Logic (Pre-training weights)
    if args.load_weights:
        logger.info('Loading initial weights from: %s', args.load_weights)
        global_agent.initial_load_path = args.load_weights
        metadata = global_agent.load_model(args.load_weights)
        if metadata:
            episode = metadata.get('episode', 0)
            logger.info('Continuous episode count from loaded weights: %d', episode)
        # Progress (fold, epoch) remains at 0 unless --resume also specified
    
    # Resumption Logic (Progress + Weights)
    if args.resume:
        # global_agent.initial_load_path = args.resume # DO NOT set this as initial path for all folds
        metadata = global_agent.load_model(args.resume)
        if metadata:
            start_fold = metadata.get('fold', 1) - 1 # 1-indexed to 0-indexed
            start_epoch = metadata.get('epoch', 0)
            if start_epoch >= config.EPOCH_SIZE:
                start_fold += 1
                start_epoch = 0
            
            episode = metadata.get('episode', 0)
            logger.info('Resuming from Fold {}, Epoch {}, Episode {}'.format(start_fold + 1, start_epoch, episode))
        else:
            logger.warning('Resume specified but metadata not found. Starting from scratch with loaded weights.')


    try:
        global_agent.train(train_path=train_path, start_fold=start_fold, start_epoch=start_epoch)
        
        # Save Final Model
        time_str_final = datetime.now().strftime("%H%M%S")
        final_model_dir = os.path.join(config.SAVE_MODEL_DIR, "final_model_{}".format(time_str_final))
        global_agent.save_model(final_model_dir, metadata={'episode': episode, 'status': 'final'})
        logger.info("Final model saved to: %s", final_model_dir)

        # Optional: Evaluate after full training
        logger.info("Training completed. Running final evaluation...")
        try:
            all_files = [os.path.abspath(os.path.join(train_path, f)) 
                        for f in os.listdir(train_path) 
                        if os.path.isfile(os.path.join(train_path, f))]
            all_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in config.VALID_IMAGE_EXTENSIONS)]
            random.shuffle(all_files)
            logger.info('all_files len: %d', len(all_files))
            test_files = all_files[:30]
            global_agent.evaluate(test_files)
        except Exception as e:
            logger.error('Evaluation failed: %s', e)
            
    except KeyboardInterrupt:
        logger.warning('Training interrupted by user (Ctrl+C)')
    except Exception as e:
        logger.error('Training failed: %s', e)
        import traceback
        logger.error('Traceback: %s', traceback.format_exc())
    finally:
        total_elapsed = time.time() - script_start_time
        # Calculate total epochs
        if config.USE_K_FOLD:
            total_epochs = config.K_FOLDS * config.EPOCH_SIZE
        else:
            total_epochs = config.EPOCH_SIZE
            
        logger.info("="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("Total Executed Epochs: %d", total_epochs)
        logger.info("Total Executed Episodes: %d", episode)
        logger.info("TOTAL EXECUTION TIME: {:.2f}s ({:.2f}m)".format(total_elapsed, total_elapsed / 60.0))
        logger.info("="*60)
        
        # ✅ CRITICAL: Always cleanup
        logger.info('')
        logger.info('Running cleanup before exit...')
        global_agent.cleanup()
        logger.info('Program exited cleanly')


