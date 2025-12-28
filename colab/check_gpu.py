import tensorflow as tf
import os
import multiprocessing

print("="*50)
print("GPU Detection Check")
print("="*50)

# CPU Info
print(f"CPU count: {multiprocessing.cpu_count()}")

# TensorFlow Version
print(f"TensorFlow version: {tf.__version__}")

# GPU Detection
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU [{i}]: {gpu}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Name: {details.get('device_name', 'Unknown')}")
        except:
            print("    Could not get GPU details")
else:
    print("  No GPU available to TensorFlow.")
    print("  Check if GPU is enabled in Kaggle Settings (Accelerator).")

# Environment Variables
print("\nRelevant Environment Variables:")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"  TF_CPP_MIN_LOG_LEVEL: {os.environ.get('TF_CPP_MIN_LOG_LEVEL', 'Not set')}")
print("="*50)
