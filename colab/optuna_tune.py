import optuna
import os
import subprocess
import re
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target script path logic
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(CURRENT_DIR, 'A2RL_a3c_colab.py')

def objective(trial):
    """
    Optuna objective function to optimize A2RL hyperparameters.
    """
    # 1. Suggest Hyperparameters
    stop_reward = trial.suggest_float('A2RL_STOP_REWARD', -1.0, 0.0, step=0.1)
    min_steps = trial.suggest_int('A2RL_MIN_STEPS', 3, 5)
    beta = trial.suggest_float('A2RL_BETA', 0.01, 0.2, log=True)
    
    # 2. Setup Environment Variables for this Trial
    env = os.environ.copy()
    env['A2RL_STOP_REWARD'] = str(stop_reward)
    env['A2RL_MIN_STEPS'] = str(min_steps)
    env['A2RL_BETA'] = str(beta)
    
    # Additional User Configuration
    env['A2RL_BATCH_SIZE'] = '8'
    env['A2RL_THREADS'] = '8'
    env['A2RL_T_MAX'] = '50'
    env['A2RL_CONSOLE_LOG_LEVEL'] = 'WARNING'
    
    # Speed up trials: Reduce epochs and episodes for faster feedback
    env['A2RL_EPOCH_SIZE'] = '50'        # Reduce from 200 to 50
    env['A2RL_MAX_EPISODES'] = '2000'    # Reduce from 20000 to 2000
    env['A2RL_GDRIVE_BACKUP_ENABLED'] = '0' # Disable backups for speed
    
    logger.info(f"Trial {trial.number}: Starting training with STOP_REWARD={stop_reward}, MIN_STEPS={min_steps}, BETA={beta}")
    
    # 3. Run Training Script as Subprocess
    try:
        # Capture output to parse the final score
        # Using simplified command, assuming python is in path
        result = subprocess.run(
            [sys.executable, SCRIPT_PATH],
            cwd=CURRENT_DIR, # Execute in the script's directory to ensure imports work
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stderr + result.stdout # Use both streams as logger usually prints to stderr
        
        # 4. Parse Final Evaluation Score
        # Look for the pattern: "Avg Final Score:   7.6307"
        match = re.search(r"Avg Final Score:\s+([\d\.]+)", output)
        
        if match:
            final_score = float(match.group(1))
            logger.info(f"Trial {trial.number} finished. Score: {final_score}")
            return final_score
        else:
            logger.error(f"Trial {trial.number} failed to produce a score. Output excerpt:\n{output[-500:]}")
            # Prune invalid trials
            raise optuna.exceptions.TrialPruned()
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        logger.error(f"Output:\n{e.stderr}")
        raise optuna.exceptions.TrialPruned()
    except Exception as e:
        logger.error(f"Trial {trial.number} unexpected error: {e}")
        raise e

if __name__ == "__main__":
    # Create Study
    study = optuna.create_study(direction="maximize")
    
    # Optimize
    # n_trials: Number of trials to run. Adjust based on available time.
    logger.info("Starting Optuna optimization...")
    study.optimize(objective, n_trials=10) # Start with 10 trials for demonstration
    
    # Report Results
    logger.info("Optimization finished.")
    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best score: {study.best_value}")
    
    # Save results
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv")
    logger.info("Results saved to optuna_results.csv")
