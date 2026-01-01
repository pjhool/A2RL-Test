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
    # User Request:
    # beta = trial.suggest_float('beta', 0.01, 0.3, log=True)
    # stop_reward = trial.suggest_float('stop_reward', -10.0, -0.5)
    # step_penalty = trial.suggest_float('step_penalty', 0.0, 0.1)
    # min_steps = trial.suggest_int('min_steps', 3, 10)
    
    beta = trial.suggest_float('A2RL_BETA', 0.01, 0.15, log=True)
    
    # Fixed Reward Scale, Step Penalty, and Min Steps
    reward_scale = 1.0
    step_penalty = 0.005
    min_steps = 5
    
    # 2. Setup Environment Variables for this Trial
    env = os.environ.copy()
    env['A2RL_MIN_STEPS'] = str(min_steps)
    env['A2RL_BETA'] = str(beta)
    env['A2RL_REWARD_SCALE'] = str(reward_scale)
    env['A2RL_STEP_PENALTY'] = str(step_penalty)

    
    # Additional User Configuration
    
    # Speed up trials: Reduce epochs and episodes for faster feedback
    env['A2RL_EPOCH_SIZE'] = '50'        # Reduce from 200 to 50

    #env['A2RL_MAX_EPISODES'] = '5000'    # Increase to 5000 for better convergence
    env['A2RL_GDRIVE_BACKUP_ENABLED'] = '0' # Disable backups for speed
    
    logger.info(f"Trial {trial.number}: Starting training with MIN_STEPS={min_steps}, BETA={beta}, SCALE={reward_scale}")

    logger.info(f"  Fixed Config: BATCH_SIZE={env['A2RL_BATCH_SIZE']}, THREADS={env['A2RL_THREADS']}, T_MAX={env['A2RL_T_MAX']}, LOG_LEVEL={env['A2RL_CONSOLE_LOG_LEVEL']}")
    
    # 3. Run Training Script as Subprocess
    output_buffer = []
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            [sys.executable, SCRIPT_PATH],
            cwd=CURRENT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,
            bufsize=1 # Line buffered
        )
        
        # Read output line by line
        for line in process.stdout:
            print(line, end='') # Print to console for user visibility
            output_buffer.append(line)
            
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args, output="".join(output_buffer))
            
        output = "".join(output_buffer)
        

        # 4. Parse Final Evaluation Metrics
        # Patterns:
        # Avg Final Score:   7.6307
        # Avg Improvement:   -0.1252 (-1.58%)
        # Avg Steps:         50.0
        
        score_match = re.search(r"Avg Final Score:\s+([\d\.]+)", output)
        imp_match = re.search(r"Avg Improvement:\s+([\-\d\.]+)", output)
        steps_match = re.search(r"Avg Steps:\s+([\d\.]+)", output)
        
        final_score = float(score_match.group(1)) if score_match else 0.0
        improvement = float(imp_match.group(1)) if imp_match else -999.0
        avg_steps = float(steps_match.group(1)) if steps_match else 0.0
        
        if score_match and imp_match:
            # Multi-objective: Improvement maximize + Steps validity penalty
            step_penalty_score = 0.0
            if avg_steps < 5:
                step_penalty_score = -0.5 * (5 - avg_steps)
            elif avg_steps > 20:
                step_penalty_score = -0.2 * (avg_steps - 20)
                
            final_composite_score = improvement + step_penalty_score
            
            logger.warning(f"Trial {trial.number} Result: Composite={final_composite_score:.4f} (Imp={improvement:.4f}, Steps={avg_steps:.1f}, StepPenalty={step_penalty_score:.4f})")
            logger.warning(f"  Params: MIN={min_steps}, BETA={beta:.4f}, SCALE={reward_scale}, PENALTY={step_penalty:.4f}")
            return final_composite_score

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
    
    # User Request: Tune Beta from high to low (Prioritize Exploration)
    # Enqueue specific Beta values to try first
    study.enqueue_trial({'A2RL_BETA': 0.15})
    study.enqueue_trial({'A2RL_BETA': 0.1})
    study.enqueue_trial({'A2RL_BETA': 0.05})
    study.enqueue_trial({'A2RL_BETA': 0.01})
    
    # Optimize
    # n_trials: Number of trials to run. Adjust based on available time.
    logger.info("Starting Optuna optimization...")
    study.optimize(objective, n_trials=20) # Start with 10 trials for demonstration
    
    # Report Results
    logger.info("Optimization finished.")
    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best score: {study.best_value}")
    
    # Save results
    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv")
    logger.info("Results saved to optuna_results.csv")
