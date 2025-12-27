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
    import os
    
    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"Warning: Log directory does not exist yet: {log_dir}")
        print("TensorBoard will start once logs are created.")
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    # Load TensorBoard extension
    try:
        # For Jupyter/Kaggle notebooks
        get_ipython().run_line_magic('load_ext', 'tensorboard')
        get_ipython().run_line_magic('tensorboard', f'--logdir {log_dir}')
        print(f"TensorBoard started successfully!")
        print(f"Monitoring: {log_dir}")
    except Exception as e:
        print(f"Error starting TensorBoard: {e}")
        print("\nAlternative: Use command line in a separate terminal:")
        print(f"  tensorboard --logdir={log_dir} --host=0.0.0.0 --port=6006")


def list_tensorboard_logs(base_dir='/kaggle/working/summary/A2RL_a3c'):
    """
    List available TensorBoard log directories.
    
    Args:
        base_dir: Base directory containing TensorBoard logs
    """
    import os
    from datetime import datetime
    
    if not os.path.exists(base_dir):
        print(f"Log directory does not exist: {base_dir}")
        return
    
    print(f"Available TensorBoard logs in {base_dir}:\n")
    
    # List all subdirectories (date folders)
    for date_folder in sorted(os.listdir(base_dir)):
        date_path = os.path.join(base_dir, date_folder)
        if os.path.isdir(date_path):
            print(f"📅 {date_folder}/")
            
            # List event files in each date folder
            for event_folder in sorted(os.listdir(date_path)):
                event_path = os.path.join(date_path, event_folder)
                if os.path.isdir(event_path):
                    # Get file count and size
                    files = [f for f in os.listdir(event_path) if f.startswith('events.out')]
                    if files:
                        total_size = sum(os.path.getsize(os.path.join(event_path, f)) for f in files)
                        size_mb = total_size / (1024 * 1024)
                        print(f"  📊 {event_folder}/ ({len(files)} files, {size_mb:.2f} MB)")
    
    print(f"\nTo view a specific date's logs:")
    print(f"  start_tensorboard('{base_dir}/YYYYMMDD')")


if __name__ == "__main__":
    # If run as script, start TensorBoard with default settings
    import sys
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = '/kaggle/working/summary/A2RL_a3c'
    
    print("=" * 60)
    print("TensorBoard Helper for Kaggle")
    print("=" * 60)
    
    list_tensorboard_logs(log_dir)
    print("\n" + "=" * 60)
    print("Starting TensorBoard...")
    print("=" * 60 + "\n")
    
    start_tensorboard(log_dir)
