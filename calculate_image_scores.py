#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image Aesthetic Score Calculator with Parallel Processing

This script reads images from the ../AVA/Train8954 folder and calculates
aesthetic scores using the VFN (Visual Feature Network) model.

Supports parallel processing using multiprocessing for faster computation.

Based on A2RL_a3c.py and network_vfn.py
"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
import skimage.io as io
import skimage.transform as transform
from os import listdir
from os.path import isfile, join, getsize
import csv
from datetime import datetime
from multiprocessing import Pool, cpu_count
import sys
import os

import network_vfn as nw

# Global configuration
global_dtype = tf.float32
batch_size = 1
embedding_dim = 1000
ranking_loss = 'svm'

# Valid image extensions
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Minimum file size in bytes (to filter out corrupted/empty files)
MIN_FILE_SIZE = 1024  # 1 KB

# Global variables for multiprocessing
_vfn_sess = None
_score_func = None
_image_placeholder = None
_snapshot_path = None
_alexnet_path = None


def init_worker(snapshot_path, alexnet_path):
    """
    Initialize worker process with VFN model.
    This is called once per worker process.
    """
    global _vfn_sess, _score_func, _image_placeholder, _snapshot_path, _alexnet_path
    
    _snapshot_path = snapshot_path
    _alexnet_path = alexnet_path
    
    print("Worker {}: Loading VFN model...".format(id(Pool)))
    
    # Reset TensorFlow graph
    tf.reset_default_graph()
    
    # Load AlexNet weights
    net_data = np.load(alexnet_path, encoding='bytes').item()
    
    # Create placeholders and build network
    _image_placeholder = tf.placeholder(dtype=global_dtype, shape=[batch_size, 227, 227, 3])
    var_dict = nw.get_variable_dict(net_data)
    
    SPP = True
    pooling = 'avg'
    
    with tf.variable_scope("ranker") as scope:
        feature_vec = nw.build_alexconvnet(_image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
        _score_func = nw.score(feature_vec)
    
    # Load pre-trained model
    saver = tf.train.Saver(tf.global_variables())
    _vfn_sess = tf.Session(config=tf.ConfigProto())
    _vfn_sess.run(tf.global_variables_initializer())
    saver.restore(_vfn_sess, snapshot_path)
    
    print("Worker {}: VFN model loaded successfully!".format(id(Pool)))


def process_single_image(args):
    """
    Process a single image and calculate its aesthetic score.
    This function is called by worker processes.
    
    Args:
        args: Tuple of (image_folder, filename, idx, total)
    
    Returns:
        Dictionary with filename, score, feature_dim, and error (if any)
    """
    global _vfn_sess, _score_func, _image_placeholder
    
    image_folder, filename, idx, total = args
    
    try:
        filepath = join(image_folder, filename)
        
        # Read image
        img = io.imread(filepath)
        
        # Skip non-RGB images
        if img.ndim != 3:
            error_msg = "Not a 3-channel image (ndim={})".format(img.ndim)
            print("Skipping {}: {}".format(filename, error_msg))
            return {'filename': filename, 'error': error_msg}
        
        # Ensure RGB (take first 3 channels)
        img = img[:, :, :3]
        
        # Normalize and resize
        img = img.astype(np.float32) / 255
        img_resize = transform.resize(img, (227, 227)) - 0.5
        img_resize = np.expand_dims(img_resize, axis=0)
        
        # Calculate score
        score, feature = _vfn_sess.run([_score_func], feed_dict={_image_placeholder: img_resize})[0]
        
        feature_shape = feature.shape
        
        result = {
            'filename': filename,
            'score': float(score),
            'feature_dim': feature_shape[1] if len(feature_shape) > 1 else feature_shape[0],
            'error': None
        }
        
        print("[{}/{}] {}: score = {:.6f}".format(idx+1, total, filename, float(score)))
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print("Error processing {}: {}".format(filename, error_msg))
        return {'filename': filename, 'error': error_msg}


def evaluate_aesthetics_score(images, vfn_sess, score_func, image_placeholder):
    """
    Calculate aesthetic scores for a list of images.
    
    Args:
        images: List of images (numpy arrays)
        vfn_sess: TensorFlow session with loaded VFN model
        score_func: Score function from the VFN model
        image_placeholder: TensorFlow placeholder for input images
    
    Returns:
        scores: Numpy array of aesthetic scores
        features: List of feature vectors
    """
    scores = np.zeros(shape=(len(images),))
    features = []
    
    for i in range(len(images)):
        img = images[i].astype(np.float32) / 255
        img_resize = transform.resize(img, (227, 227)) - 0.5
        img_resize = np.expand_dims(img_resize, axis=0)
        score, feature = vfn_sess.run([score_func], feed_dict={image_placeholder: img_resize})[0]
        scores[i] = score
        features.append(feature)
    
    return scores, features


def load_vfn_model(snapshot_path='../a2rl_model/model-spp-max', 
                   alexnet_path='alexnet.npy'):
    """
    Load the VFN model and create TensorFlow session.
    
    Args:
        snapshot_path: Path to the pre-trained VFN model
        alexnet_path: Path to the AlexNet weights file
    
    Returns:
        vfn_sess: TensorFlow session
        score_func: Score function
        image_placeholder: Input placeholder
    """
    print("Loading VFN model...")
    
    # Reset TensorFlow graph
    tf.reset_default_graph()
    
    # Load AlexNet weights
    net_data = np.load(alexnet_path, encoding='bytes').item()
    
    # Create placeholders and build network
    image_placeholder = tf.placeholder(dtype=global_dtype, shape=[batch_size, 227, 227, 3])
    var_dict = nw.get_variable_dict(net_data)
    
    SPP = True
    pooling = 'avg'
    
    with tf.variable_scope("ranker") as scope:
        feature_vec = nw.build_alexconvnet(image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
        score_func = nw.score(feature_vec)
    
    # Load pre-trained model
    saver = tf.train.Saver(tf.global_variables())
    vfn_sess = tf.Session(config=tf.ConfigProto())
    vfn_sess.run(tf.global_variables_initializer())
    saver.restore(vfn_sess, snapshot_path)
    
    print("VFN model loaded successfully!")
    
    return vfn_sess, score_func, image_placeholder


def is_valid_image_file(filepath, filename):
    """
    Validate if a file is a valid image file.
    
    Args:
        filepath: Full path to the file
        filename: Name of the file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    _, ext = os.path.splitext(filename.lower())
    if ext not in VALID_IMAGE_EXTENSIONS:
        return False, "Invalid extension: {}".format(ext)
    
    # Check file size
    try:
        file_size = getsize(filepath)
        if file_size < MIN_FILE_SIZE:
            return False, "File too small ({} bytes)".format(file_size)
    except Exception as e:
        return False, "Cannot get file size: {}".format(e)
    
    return True, None


def process_images_parallel(image_folder='../AVA/Train8954', output_file='image_scores.csv', 
                           num_workers=None, snapshot_path='../a2rl_model/model-spp-max',
                           alexnet_path='alexnet.npy'):
    """
    Process all images in parallel using multiprocessing.
    
    Args:
        image_folder: Path to folder containing images
        output_file: Path to output CSV file for results
        num_workers: Number of worker processes (default: CPU count)
        snapshot_path: Path to pre-trained VFN model
        alexnet_path: Path to AlexNet weights file
    """
    print("Processing images from: {}".format(image_folder))
    
    # Get list of all files
    try:
        all_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]
        print("Found {} files in the folder".format(len(all_files)))
    except Exception as e:
        print("Error reading folder: {}".format(e))
        return
    
    # Validate and filter image files
    print("Validating files...")
    image_files = []
    validation_errors = []
    
    for filename in all_files:
        filepath = join(image_folder, filename)
        is_valid, error_msg = is_valid_image_file(filepath, filename)
        
        if is_valid:
            image_files.append(filename)
        else:
            validation_errors.append({'filename': filename, 'error': error_msg})
    
    print("Valid image files: {}".format(len(image_files)))
    print("Invalid/skipped files: {}".format(len(validation_errors)))
    
    if not image_files:
        print("No valid image files to process.")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    print("Using {} worker processes".format(num_workers))
    
    # Prepare arguments for parallel processing
    total = len(image_files)
    args_list = [(image_folder, filename, idx, total) for idx, filename in enumerate(image_files)]
    
    # Process images in parallel
    print("\nProcessing images in parallel...")
    pool = Pool(processes=num_workers, initializer=init_worker, initargs=(snapshot_path, alexnet_path))
    
    try:
        results_raw = pool.map(process_single_image, args_list)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating workers...")
        pool.terminate()
        pool.join()
        sys.exit(1)
    
    # Separate successful and failed results
    successful_results = [r for r in results_raw if r and r.get('error') is None]
    failed_results = [r for r in results_raw if r and r.get('error') is not None]
    
    # Combine all errors (validation + processing)
    all_errors = validation_errors + failed_results
    
    # Save successful results to CSV
    if successful_results:
        print("\nSaving results to {}...".format(output_file))
        with open(output_file, 'w') as csvfile:
            fieldnames = ['filename', 'score', 'feature_dim']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in successful_results:
                writer.writerow({'filename': result['filename'], 
                               'score': result['score'], 
                               'feature_dim': result['feature_dim']})
        
        print("Results saved successfully!")
        
        # Print statistics
        scores_array = np.array([r['score'] for r in successful_results])
        print("\nScore Statistics:")
        print("  Mean:   {:.6f}".format(np.mean(scores_array)))
        print("  Median: {:.6f}".format(np.median(scores_array)))
        print("  Std:    {:.6f}".format(np.std(scores_array)))
        print("  Min:    {:.6f}".format(np.min(scores_array)))
        print("  Max:    {:.6f}".format(np.max(scores_array)))
    
    # Save error log
    if all_errors:
        error_log_file = output_file.replace('.csv', '_errors.csv')
        print("\nSaving error log to {}...".format(error_log_file))
        with open(error_log_file, 'w') as csvfile:
            fieldnames = ['filename', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for error in all_errors:
                writer.writerow(error)
        
        print("Error log saved successfully!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Processing Summary:")
    print("="*60)
    print("Total files found:        {}".format(len(all_files)))
    print("Valid image files:        {}".format(len(image_files)))
    print("Successfully processed:   {}".format(len(successful_results)))
    print("Failed to process:        {}".format(len(failed_results)))
    print("Validation errors:        {}".format(len(validation_errors)))
    print("Total errors:             {}".format(len(all_errors)))
    if len(image_files) > 0:
        success_rate = (len(successful_results) / float(len(image_files))) * 100
        print("Success rate:             {:.2f}%".format(success_rate))
    print("="*60)


def process_images(image_folder='../AVA/Train8954', output_file='image_scores.csv'):
    """
    Process all images sequentially (single process).
    
    Args:
        image_folder: Path to folder containing images
        output_file: Path to output CSV file for results
    """
    print("Processing images from: {}".format(image_folder))
    
    # Load VFN model
    vfn_sess, score_func, image_placeholder = load_vfn_model()
    
    # Get list of all files
    try:
        all_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]
        print("Found {} files in the folder".format(len(all_files)))
    except Exception as e:
        print("Error reading folder: {}".format(e))
        return
    
    # Validate and filter image files
    print("Validating files...")
    image_files = []
    validation_errors = []
    
    for filename in all_files:
        filepath = join(image_folder, filename)
        is_valid, error_msg = is_valid_image_file(filepath, filename)
        
        if is_valid:
            image_files.append(filename)
        else:
            validation_errors.append({'filename': filename, 'error': error_msg})
    
    print("Valid image files: {}".format(len(image_files)))
    print("Invalid/skipped files: {}".format(len(validation_errors)))
    
    if not image_files:
        print("No valid image files to process.")
        vfn_sess.close()
        return
    
    # Prepare results storage
    successful_results = []
    failed_results = []
    
    # Process each image
    for idx, filename in enumerate(image_files):
        try:
            filepath = join(image_folder, filename)
            
            # Read image
            img = io.imread(filepath)
            
            # Skip non-RGB images
            if img.ndim != 3:
                error_msg = "Not a 3-channel image (ndim={})".format(img.ndim)
                print("Skipping {}: {}".format(filename, error_msg))
                failed_results.append({'filename': filename, 'error': error_msg})
                continue
            
            # Ensure RGB (take first 3 channels)
            img = img[:, :, :3]
            
            # Calculate score
            scores, features = evaluate_aesthetics_score([img], vfn_sess, score_func, image_placeholder)
            
            score = scores[0]
            feature_shape = features[0].shape
            
            # Store result
            successful_results.append({
                'filename': filename,
                'score': float(score),
                'feature_dim': feature_shape[1] if len(feature_shape) > 1 else feature_shape[0]
            })
            
            print("[{}/{}] {}: score = {:.6f}".format(idx+1, len(image_files), filename, float(score)))
            
        except Exception as e:
            error_msg = str(e)
            print("Error processing {}: {}".format(filename, error_msg))
            failed_results.append({'filename': filename, 'error': error_msg})
            continue
    
    # Combine all errors
    all_errors = validation_errors + failed_results
    
    # Save successful results to CSV
    if successful_results:
        print("\nSaving results to {}...".format(output_file))
        with open(output_file, 'w') as csvfile:
            fieldnames = ['filename', 'score', 'feature_dim']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in successful_results:
                writer.writerow(result)
        
        print("Results saved successfully!")
        
        # Print statistics
        scores_array = np.array([r['score'] for r in successful_results])
        print("\nScore Statistics:")
        print("  Mean:   {:.6f}".format(np.mean(scores_array)))
        print("  Median: {:.6f}".format(np.median(scores_array)))
        print("  Std:    {:.6f}".format(np.std(scores_array)))
        print("  Min:    {:.6f}".format(np.min(scores_array)))
        print("  Max:    {:.6f}".format(np.max(scores_array)))
    
    # Save error log
    if all_errors:
        error_log_file = output_file.replace('.csv', '_errors.csv')
        print("\nSaving error log to {}...".format(error_log_file))
        with open(error_log_file, 'w') as csvfile:
            fieldnames = ['filename', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for error in all_errors:
                writer.writerow(error)
        
        print("Error log saved successfully!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Processing Summary:")
    print("="*60)
    print("Total files found:        {}".format(len(all_files)))
    print("Valid image files:        {}".format(len(image_files)))
    print("Successfully processed:   {}".format(len(successful_results)))
    print("Failed to process:        {}".format(len(failed_results)))
    print("Validation errors:        {}".format(len(validation_errors)))
    print("Total errors:             {}".format(len(all_errors)))
    if len(image_files) > 0:
        success_rate = (len(successful_results) / float(len(image_files))) * 100
        print("Success rate:             {:.2f}%".format(success_rate))
    print("="*60)
    
    # Close TensorFlow session
    vfn_sess.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate aesthetic scores for images')
    parser.add_argument('--image_folder', type=str, default='../AVA/Train8954',
                        help='Path to folder containing images')
    parser.add_argument('--output_file', type=str, default='image_scores.csv',
                        help='Path to output CSV file')
    parser.add_argument('--model_path', type=str, default='../a2rl_model/model-spp-max',
                        help='Path to pre-trained VFN model')
    parser.add_argument('--alexnet_path', type=str, default='alexnet.npy',
                        help='Path to AlexNet weights file')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing using multiprocessing')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Image Aesthetic Score Calculator")
    print("=" * 60)
    print("Start time: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print()
    
    if args.parallel:
        print("Mode: Parallel processing")
        process_images_parallel(args.image_folder, args.output_file, 
                               args.num_workers, args.model_path, args.alexnet_path)
    else:
        print("Mode: Sequential processing")
        process_images(args.image_folder, args.output_file)
    
    print()
    print("End time: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("=" * 60)
