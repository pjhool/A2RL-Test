# -*- coding: utf-8 -*-
"""
TensorBoard helper for Kaggle notebooks.
Use this in a separate cell to view TensorBoard while training is running.
"""

def start_tensorboard(log_dir='/kaggle/working/summary/A2RL_a3c'):
    """
    Start TensorBoard in Kaggle notebook.
    
    Args:
        log_dir: Path to TensorBoard log directory
    
    Usage in Kaggle notebook:
        # Cell 1: Start training (this will run for a long time)
        !python /kaggle/working/A2RL-Test/colab/A2RL_a3c_colab.py
        
        # Cell 2: In a separate cell, run this to view TensorBoard
        from tensorboard_kaggle import start_tensorboard
        start_tensorboard()
    """
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Check if log directory exists
    if not os.path.exists(log_dir):
        logger.info(f"Warning: Log directory does not exist yet: {log_dir}")
        logger.info("TensorBoard will start once logs are created.")
    import subprocess
    import time

    # Kill existing tensorboard processes to avoid port conflicts/stale instances
    logger.info("Checking for existing TensorBoard instances...")
    try:
        # Find pids of tensorboard
        pids = subprocess.check_output(["pgrep", "-f", "tensorboard"]).decode().split()
        if pids:
            logger.info(f"Found existing TensorBoard processes: {pids}. Killing them...")
            for pid in pids:
                try:
                    os.kill(int(pid), 9) # SIGKILL (Aggressive kill)
                except OSError:
                    pass
            time.sleep(2) # Wait for cleanup
            logger.info("Cleanup complete.")
        else:
            logger.info("No existing TensorBoard instances found.")
    except Exception as e:
        logger.info(f"Cleanup check skipped/failed (not critical): {e}")

    # Load TensorBoard extension
    try:
        # For Jupyter/Kaggle notebooks
        # Try loading first, if already loaded it might print a warning but that's fine.
        # If it fails (e.g. strict mode), we try reload.
        try:
            get_ipython().run_line_magic('load_ext', 'tensorboard')
        except:
            get_ipython().run_line_magic('reload_ext', 'tensorboard')
            
        # Start TensorBoard with bind_all which is often needed in containerized environments
        logger.info(f"Starting TensorBoard on {log_dir} ...")
        get_ipython().run_line_magic('tensorboard', f'--logdir {log_dir} --bind_all')
        
        logger.info(f"TensorBoard started successfully!")
        logger.info("\nNote: If the graph is empty, wait for the first summary write (approx. 10-20 mins).")
        logger.info("Note: If the UI doesn't appear, try refreshing the page.")
    except Exception as e:
        logger.error(f"Error starting TensorBoard: {e}")
        logger.info("\nAlternative: Use command line in a separate terminal:")
        logger.info(f"  tensorboard --logdir={log_dir} --bind_all")


def list_tensorboard_logs(base_dir='/kaggle/working/summary/A2RL_a3c'):
    """
    List available TensorBoard log directories.
    
    Args:
        base_dir: Base directory containing TensorBoard logs
    """
    import logging
    import os
    from datetime import datetime
    
    # Configure logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if not os.path.exists(base_dir):
        logger.info(f"Log directory does not exist: {base_dir}")
        return
    
    logger.info(f"Available TensorBoard logs in {base_dir}:\n")
    
    # List all subdirectories (date folders)
    for date_folder in sorted(os.listdir(base_dir)):
        date_path = os.path.join(base_dir, date_folder)
        if os.path.isdir(date_path):
            logger.info(f"📅 {date_folder}/")
            
            # List event files in each date folder
            for event_folder in sorted(os.listdir(date_path)):
                event_path = os.path.join(date_path, event_folder)
                if os.path.isdir(event_path):
                    # Get file count and size
                    files = [f for f in os.listdir(event_path) if f.startswith('events.out')]
                    if files:
                        total_size = sum(os.path.getsize(os.path.join(event_path, f)) for f in files)
                        size_mb = total_size / (1024 * 1024)
                        logger.info(f"  📊 {event_folder}/ ({len(files)} files, {size_mb:.2f} MB)")
    
    logger.info(f"\nTo view a specific date's logs:")
    logger.info(f"  start_tensorboard('{base_dir}/YYYYMMDD')")


if __name__ == "__main__":
    # If run as script, start TensorBoard with default settings
    import sys
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = '/kaggle/working/summary/A2RL_a3c'
    
    logger.info("=" * 60)
    logger.info("TensorBoard Helper for Kaggle")
    logger.info("=" * 60)
    
    list_tensorboard_logs(log_dir)
    logger.info("\n" + "=" * 60)
    logger.info("Starting TensorBoard...")
    logger.info("=" * 60 + "\n")
    
    start_tensorboard(log_dir)
