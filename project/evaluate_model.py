import os
import argparse
import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_processor import DataProcessor


def evaluate_model(model_path, test_dataset, device):
    """
    Evaluate a model on test data.
    
    Args:
        model_path (str): Path to the model
        test_dataset: Test dataset
        device: PyTorch device
        
    Returns:
        dict: Evaluation metrics
    """
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False
    )
    
    # Evaluate
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {os.path.basename(model_path)}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Store predictions and ground truth
            predictions.extend(outputs.logits.squeeze().cpu().numpy())
            ground_truth.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)
    
    return {
        'model': os.path.basename(model_path),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'ground_truth': ground_truth
    }


def compare_models(base_model_path, transfer_model_path, test_dataset, device, output_dir):
    """
    Compare the performance of the base and transfer models.
    
    Args:
        base_model_path (str): Path to the base model
        transfer_model_path (str): Path to the transfer model
        test_dataset: Test dataset
        device: PyTorch device
        output_dir (str): Directory to save results
        
    Returns:
        tuple: (base_results, transfer_results)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate both models
    base_results = evaluate_model(base_model_path, test_dataset, device)
    transfer_results = evaluate_model(transfer_model_path, test_dataset, device)
    
    # Print results
    print("\nModel Comparison:")
    print(f"Base Model ({base_model_path}):")
    print(f"  MSE: {base_results['mse']:.4f}")
    print(f"  RMSE: {base_results['rmse']:.4f}")
    print(f"  MAE: {base_results['mae']:.4f}")
    print(f"  R^2: {base_results['r2']:.4f}")
    
    print(f"\nTransfer Model ({transfer_model_path}):")
    print(f"  MSE: {transfer_results['mse']:.4f}")
    print(f"  RMSE: {transfer_results['rmse']:.4f}")
    print(f"  MAE: {transfer_results['mae']:.4f}")
    print(f"  R^2: {transfer_results['r2']:.4f}")
    
    # Calculate improvement
    mse_improvement = (base_results['mse'] - transfer_results['mse']) / base_results['mse'] * 100
    rmse_improvement = (base_results['rmse'] - transfer_results['rmse']) / base_results['rmse'] * 100
    mae_improvement = (base_results['mae'] - transfer_results['mae']) / base_results['mae'] * 100
    r2_improvement = (transfer_results['r2'] - base_results['r2']) / abs(base_results['r2']) * 100 if base_results['r2'] != 0 else float('inf')
    
    print("\nImprovement:")
    print(f"  MSE: {mse_improvement:.2f}%")
    print(f"  RMSE: {rmse_improvement:.2f}%")
    print(f"  MAE: {mae_improvement:.2f}%")
    print(f"  R^2: {r2_improvement:.2f}%")
    
    # Save results
    results = {
        'base_model': {
            'path': base_model_path,
            'metrics': {
                'mse': float(base_results['mse']),
                'rmse': float(base_results['rmse']),
                'mae': float(base_results['mae']),
                'r2': float(base_results['r2'])
            }
        },
        'transfer_model': {
            'path': transfer_model_path,
            'metrics': {
                'mse': float(transfer_results['mse']),
                'rmse': float(transfer_results['rmse']),
                'mae': float(transfer_results['mae']),
                'r2': float(transfer_results['r2'])
            }
        },
        'improvement': {
            'mse': float(mse_improvement),
            'rmse': float(rmse_improvement),
            'mae': float(mae_improvement),
            'r2': float(r2_improvement)
        }
    }
    
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot predictions
    plot_predictions_comparison(
        base_results['predictions'], 
        transfer_results['predictions'], 
        base_results['ground_truth'],
        output_dir
    )
    
    # Plot error distribution
    plot_error_distribution(
        base_results['predictions'],
        transfer_results['predictions'],
        base_results['ground_truth'],
        output_dir
    )
    
    # Plot metrics comparison
    plot_metrics_comparison(
        base_results,
        transfer_results,
        output_dir
    )
    
    return base_results, transfer_results


def plot_predictions_comparison(base_preds, transfer_preds, ground_truth, output_dir):
    """
    Plot predictions comparison between base and transfer models.
    
    Args:
        base_preds (list): Base model predictions
        transfer_preds (list): Transfer model predictions
        ground_truth (list): Ground truth values
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Sort by ground truth for better visualisation
    indices = np.argsort(ground_truth)
    sorted_truth = np.array(ground_truth)[indices]
    sorted_base_preds = np.array(base_preds)[indices]
    sorted_transfer_preds = np.array(transfer_preds)[indices]
    
    # Plot ground truth and predictions
    plt.scatter(range(len(sorted_truth)), sorted_truth, 
               label='Ground Truth', alpha=0.7, s=50)
    plt.scatter(range(len(sorted_base_preds)), sorted_base_preds, 
               label='Base Model', alpha=0.5, s=30)
    plt.scatter(range(len(sorted_transfer_preds)), sorted_transfer_preds, 
               label='Transfer Model', alpha=0.5, s=30)
    
    # Plot ideal line
    plt.plot([0, len(sorted_truth)], 
            [min(sorted_truth), max(sorted_truth)], 
            'r--', label='Ideal')
    
    plt.xlabel('Sample Index (sorted by ground truth)')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Model Predictions Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_comparison.png'))
    plt.close()


def plot_error_distribution(base_preds, transfer_preds, ground_truth, output_dir):
    """
    Plot error distribution histograms.
    
    Args:
        base_preds (list): Base model predictions
        transfer_preds (list): Transfer model predictions
        ground_truth (list): Ground truth values
        output_dir (str): Directory to save the plot
    """
    base_errors = np.array(base_preds) - np.array(ground_truth)
    transfer_errors = np.array(transfer_preds) - np.array(ground_truth)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(base_errors, bins=20, alpha=0.7, label='Base Model')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Base Model Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(transfer_errors, bins=20, alpha=0.7, label='Transfer Model')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Transfer Model Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()


def plot_metrics_comparison(base_results, transfer_results, output_dir):
    """
    Plot metrics comparison bar chart.
    
    Args:
        base_results (dict): Base model results
        transfer_results (dict): Transfer model results
        output_dir (str): Directory to save the plot
    """
    metrics = ['mse', 'rmse', 'mae']
    base_values = [base_results[m] for m in metrics]
    transfer_values = [transfer_results[m] for m in metrics]
    
    # Normalise values for better visualisation
    max_values = np.maximum(base_values, transfer_values)
    base_normalised = np.array(base_values) / max_values
    transfer_normalised = np.array(transfer_values) / max_values
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, base_normalised, width, label='Base Model')
    plt.bar(x + width/2, transfer_normalised, width, label='Transfer Model')
    
    plt.xlabel('Metric')
    plt.ylabel('Normalised Value (lower is better)')
    plt.title('Model Performance Comparison')
    plt.xticks(x, ['MSE', 'RMSE', 'MAE'])
    
    # Add value labels
    for i, v in enumerate(base_values):
        plt.text(i - width/2, base_normalised[i] + 0.05, f'{v:.4f}', 
                ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(transfer_values):
        plt.text(i + width/2, transfer_normalised[i] + 0.05, f'{v:.4f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()
    
    # R^2 comparison (separate because higher is better)
    plt.figure(figsize=(8, 5))
    
    r2_values = [base_results['r2'], transfer_results['r2']]
    
    plt.bar(['Base Model', 'Transfer Model'], r2_values, color=['#1f77b4', '#ff7f0e'])
    
    plt.xlabel('Model')
    plt.ylabel('R² Score (higher is better; <1)')
    plt.title('R² Score Comparison')
    
    # Add value labels
    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.05, f'{v:.4f}', ha='center', va='bottom')
    
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_comparison.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare models')
    parser.add_argument('--base_model_path', type=str, default='models/un_base_model',
                        help='Path to the base model')
    parser.add_argument('--transfer_model_path', type=str, default='models/mun_transfer_model',
                        help='Path to the transfer model')
    parser.add_argument('--mun_data_path', type=str, default='training_data.json',
                        help='Path to MUN speeches JSON file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialise processor and load data
    processor = DataProcessor(max_length=args.max_length)
    mun_df = processor.load_mun_speeches(args.mun_data_path)
    
    # Split data
    _, _, _, _, mun_test = processor.prepare_datasets(None, mun_df)
    
    # Create test dataset
    test_dataset = processor.create_torch_dataset(mun_test)
    
    # Compare models
    compare_models(
        args.base_model_path,
        args.transfer_model_path,
        test_dataset,
        device,
        args.output_dir
    )
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main() 