import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_processor import DataProcessor

# Set device with better AMD GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 
                     ('rocm' if hasattr(torch, 'is_hip_available') and torch.is_hip_available() else 'cpu'))
print(f"Using device: {device}")

class TransferModel:
    """
    Transfer learning model that fine-tunes a base model trained on UN debates
    with specialized MUN speech data.
    """
    
    def __init__(self, base_model_path, output_dir='models'):
        """
        Initialize the transfer model.
        
        Args:
            base_model_path (str): Path to the base model
            output_dir (str): Directory to save model checkpoints
        """
        self.base_model_path = base_model_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the pre-trained base model
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
        self.model.to(device)
        
        # Initialize data processor
        self.processor = DataProcessor(max_length=512)
        
    def fine_tune(self, train_dataset, val_dataset, batch_size=4, epochs=5, 
                 learning_rate=1e-5, warmup_steps=0, output_model_name='mun_transfer_model'):
        """
        Fine-tune the base model with MUN speech data.
        
        Args:
            train_dataset: MUN training dataset
            val_dataset: MUN validation dataset
            batch_size (int): Batch size for training
            epochs (int): Number of epochs
            learning_rate (float): Learning rate (typically lower for fine-tuning)
            warmup_steps (int): Number of warmup steps
            output_model_name (str): Name for the saved model
            
        Returns:
            dict: Training history
        """
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Prepare optimizer and scheduler
        # Using different learning rates for different layers
        # This is a common technique for fine-tuning
        # We'll use a lower learning rate for the base model's parameters
        # and a higher learning rate for the newly added classifier layers
        
        # Parameters from the base model
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        # Different parameter groups with different learning rates
        optimizer_grouped_parameters = [
            # Base model params with weight decay
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) 
                          and 'classifier' not in n],
                'weight_decay': 0.01,
                'lr': learning_rate / 10  # Lower learning rate for base model
            },
            # Base model params without weight decay
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) 
                          and 'classifier' not in n],
                'weight_decay': 0.0,
                'lr': learning_rate / 10  # Lower learning rate for base model
            },
            # Classifier params with weight decay
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) 
                          and 'classifier' in n],
                'weight_decay': 0.01,
                'lr': learning_rate  # Higher learning rate for classifier
            },
            # Classifier params without weight decay
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) 
                          and 'classifier' in n],
                'weight_decay': 0.0,
                'lr': learning_rate  # Higher learning rate for classifier
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters)
        
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc="Fine-tuning")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})
            
            avg_train_loss = train_loss / len(train_dataloader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            predictions = []
            ground_truth = []
            
            progress_bar = tqdm(val_dataloader, desc="Validation")
            with torch.no_grad():
                for batch in progress_bar:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Store predictions and ground truth
                    predictions.extend(outputs.logits.squeeze().cpu().numpy())
                    ground_truth.extend(batch['labels'].cpu().numpy())
                    
                    # Update progress bar
                    val_loss += loss.item()
                    progress_bar.set_postfix({'loss': val_loss / (progress_bar.n + 1)})
            
            avg_val_loss = val_loss / len(val_dataloader)
            history['val_loss'].append(avg_val_loss)
            
            # Calculate MSE
            mse = np.mean((np.array(predictions) - np.array(ground_truth)) ** 2)
            history['val_mse'].append(mse)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f} - "
                  f"Val MSE: {mse:.4f}")
            
            # Save model if it's the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                output_path = os.path.join(self.output_dir, output_model_name)
                self.model.save_pretrained(output_path)
                print(f"Model saved to {output_path}")
        
        self.plot_training_history(history)
        return history
    
    def evaluate(self, test_dataset, batch_size=4):
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset: Test dataset
            batch_size (int): Batch size for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        self.model.eval()
        predictions = []
        ground_truth = []
        
        progress_bar = tqdm(test_dataloader, desc="Testing")
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Store predictions and ground truth
                predictions.extend(outputs.logits.squeeze().cpu().numpy())
                ground_truth.extend(batch['labels'].cpu().numpy())
        
        # Convert lists to numpy arrays
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        mse = np.mean((ground_truth - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ground_truth - predictions))
        
        # Calculate R^2 score
        mean_true = np.mean(ground_truth)
        ss_total = np.sum((ground_truth - mean_true) ** 2)
        ss_residual = np.sum((ground_truth - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
        # Print evaluation metrics
        print(f"\nEvaluation Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R^2: {r2:.4f}")
        
        # Plot predictions vs ground truth
        self.plot_predictions(predictions, ground_truth)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions.tolist(),
            'ground_truth': ground_truth.tolist()
        }
    
    def plot_training_history(self, history):
        """
        Plot training and validation loss.
        
        Args:
            history (dict): Training history
        """
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Fine-tuning Loss')
        
        # Plot validation MSE
        plt.subplot(1, 2, 2)
        plt.plot(history['val_mse'], label='Val MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('Validation MSE')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'transfer_history.png'))
        plt.close()
    
    def plot_predictions(self, predictions, ground_truth):
        """
        Plot predictions vs ground truth.
        
        Args:
            predictions (list): Model predictions
            ground_truth (list): Ground truth values
        """
        plt.figure(figsize=(10, 6))
        
        # Sort by ground truth for better visualization
        indices = np.argsort(ground_truth)
        sorted_truth = np.array(ground_truth)[indices]
        sorted_preds = np.array(predictions)[indices]
        
        # Plot ground truth and predictions
        plt.scatter(range(len(sorted_truth)), sorted_truth, label='Ground Truth', alpha=0.6)
        plt.scatter(range(len(sorted_preds)), sorted_preds, label='Predictions', alpha=0.6)
        
        # Plot ideal line
        plt.plot([0, len(sorted_truth)], 
                [min(sorted_truth), max(sorted_truth)], 
                'r--', label='Ideal')
        
        plt.xlabel('Sample Index (sorted by ground truth)')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Predictions vs Ground Truth')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'predictions.png'))
        plt.close()


def main():
    """
    Main function to fine-tune the base model with MUN data.
    """
    parser = argparse.ArgumentParser(description='Fine-tune base model with MUN data')
    parser.add_argument('--base_model_path', type=str, default='models/un_base_model',
                       help='Path to the base model trained on UN debates')
    parser.add_argument('--mun_data_path', type=str, default='training_data.json',
                       help='Path to MUN speeches JSON file')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--output_model_name', type=str, default='mun_transfer_model',
                       help='Name for the saved model')
    
    args = parser.parse_args()
    
    # Initialize processor and load data
    processor = DataProcessor(max_length=args.max_length)
    mun_df = processor.load_mun_speeches(args.mun_data_path)
    
    # Split data
    _, _, mun_train, mun_val, mun_test = processor.prepare_datasets(None, mun_df)
    
    # Create datasets
    train_dataset = processor.create_torch_dataset(mun_train)
    val_dataset = processor.create_torch_dataset(mun_val)
    test_dataset = processor.create_torch_dataset(mun_test)
    
    # Initialize and fine-tune model
    transfer_model = TransferModel(base_model_path=args.base_model_path, 
                                  output_dir=args.output_dir)
    
    # Fine-tune
    transfer_model.fine_tune(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_model_name=args.output_model_name
    )
    
    # Evaluate
    transfer_model.evaluate(test_dataset, batch_size=args.batch_size)
    
    print("Transfer learning complete!")


if __name__ == "__main__":
    main() 