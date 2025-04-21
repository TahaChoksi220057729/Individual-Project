#!/usr/bin/env python
import os
import argparse
import subprocess
import time
import shutil
import sys
import nltk
import torch
import numpy as np
import random

# Force use of GPU
os.environ["HIP_VISIBLE_DEVICES"] = "0"  # For AMD GPU
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:512"
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU memory

# Set fixed seed for reproducibility 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

nltk.download('punkt_tab')

# Get the current Python executable path for subprocess calls
PYTHON_EXECUTABLE = sys.executable
print(f"Using Python executable: {PYTHON_EXECUTABLE}")

def run_command(command, description):
    """
    Run a command and print its output.
    
    Args:
        command (list): Command to run
        description (str): Description of the command
    """
    print(f"\n{'-'*80}")
    print(f"Running {description}...")
    print(f"{'-'*80}")
    
    # Start the command
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Stream the output
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the command to finish
    process.wait()
    
    # Check the return code
    if process.returncode != 0:
        print(f"Error running {description}!")
        return False
    
    return True


def setup_directories(args):
    """
    Set up the necessary directories.
    
    Args:
        args: Command line arguments
    """
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    
    # Clear existing files if requested
    if args.clean:
        print("Cleaning output directories...")
        
        models_dir = os.path.join(args.output_dir, 'models')
        results_dir = os.path.join(args.output_dir, 'results')
        
        for file in os.listdir(models_dir):
            file_path = os.path.join(models_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        for file in os.listdir(results_dir):
            file_path = os.path.join(results_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def main():
    parser = argparse.ArgumentParser(description='Run the transfer learning pipeline')
    parser.add_argument('--un_data_path', type=str, default='un_general_debates.csv',
                        help='Path to UN debates CSV file')
    parser.add_argument('--mun_data_path', type=str, default='training_data.json',
                        help='Path to MUN speeches JSON file')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save all output')
    parser.add_argument('--base_epochs', type=int, default=3,
                        help='Number of epochs for base model training')
    parser.add_argument('--transfer_epochs', type=int, default=5,
                        help='Number of epochs for transfer model training')
    parser.add_argument('--base_batch_size', type=int, default=8,
                        help='Batch size for base model training')
    parser.add_argument('--transfer_batch_size', type=int, default=4,
                        help='Batch size for transfer model training')
    parser.add_argument('--base_lr', type=float, default=2e-5,
                        help='Learning rate for base model training')
    parser.add_argument('--transfer_lr', type=float, default=1e-5,
                        help='Learning rate for transfer model training')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length for tokenisation')
    parser.add_argument('--clean', action='store_true',
                        help='Clean output directories before running')
    parser.add_argument('--skip_base', action='store_true',
                        help='Skip base model training (use existing model)')
    parser.add_argument('--skip_transfer', action='store_true',
                        help='Skip transfer model training (use existing model)')
    parser.add_argument('--base_model_name', type=str, default='un_base_model',
                        help='Name for the base model')
    parser.add_argument('--transfer_model_name', type=str, default='mun_transfer_model',
                        help='Name for the transfer model')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories(args)
    
    # Calculate paths
    models_dir = os.path.join(args.output_dir, 'models')
    results_dir = os.path.join(args.output_dir, 'results')
    base_model_path = os.path.join(models_dir, args.base_model_name)
    transfer_model_path = os.path.join(models_dir, args.transfer_model_name)
    
    # Start timer
    start_time = time.time()
    
    # Step 1: Train the base model on UN debates
    if not args.skip_base:
        base_command = [
            PYTHON_EXECUTABLE, 'base_model.py',
            '--data_path', args.un_data_path,
            '--output_dir', models_dir,
            '--batch_size', str(args.base_batch_size),
            '--epochs', str(args.base_epochs),
            '--learning_rate', str(args.base_lr),
            '--max_length', str(args.max_length),
            '--output_model_name', args.base_model_name
        ]
        
        if not run_command(base_command, "base model training"):
            print("Base model training failed! Exiting.")
            return
    else:
        print("\nSkipping base model training as requested.")
    
    # Step 2: Fine-tune with MUN data
    if not args.skip_transfer:
        transfer_command = [
            PYTHON_EXECUTABLE, 'transfer_model.py',
            '--base_model_path', base_model_path,
            '--mun_data_path', args.mun_data_path,
            '--output_dir', models_dir,
            '--batch_size', str(args.transfer_batch_size),
            '--epochs', str(args.transfer_epochs),
            '--learning_rate', str(args.transfer_lr),
            '--max_length', str(args.max_length),
            '--output_model_name', args.transfer_model_name
        ]
        
        if not run_command(transfer_command, "transfer model training"):
            print("Transfer model training failed! Exiting.")
            return
    else:
        print("\nSkipping transfer model training as requested.")
    
    # Step 3: Evaluate and compare models
    evaluate_command = [
        PYTHON_EXECUTABLE, 'evaluate_model.py',
        '--base_model_path', base_model_path,
        '--transfer_model_path', transfer_model_path,
        '--mun_data_path', args.mun_data_path,
        '--output_dir', results_dir,
        '--max_length', str(args.max_length)
    ]
    
    if not run_command(evaluate_command, "model evaluation"):
        print("Model evaluation failed! Exiting.")
        return
    
    # Calculate and print elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'-'*80}")
    print(f"Transfer learning pipeline completed successfully!")
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Results saved to {results_dir}")
    print(f"{'-'*80}")


if __name__ == "__main__":
    main() 