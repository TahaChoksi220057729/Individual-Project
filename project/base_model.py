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

# Force use of GPU
os.environ["HIP_VISIBLE_DEVICES"] = "0"  # For AMD GPU
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()  # Clear GPU memory

# Set fixed seed for reproducibility 
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device with better AMD GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 
                     ('cuda' if hasattr(torch, 'is_hip_available') and torch.is_hip_available() else 'cpu'))
print(f"Using device: {device}")
if device.type == 'cuda' and torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

class BaseModel:
    """
    Base model for speech evaluation trained on UN general debates.
    """
    
    def __init__(self, model_name='bert-base-uncased', output_dir='models'):
        """
        Initialize the base model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            output_dir (str): Directory to save model checkpoints
        """
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        self.model.to(device)
        
        # Initialize data processor
        self.processor = DataProcessor(max_length=512)
        
    def train(self, train_dataset, val_dataset, batch_size=8, epochs=3, 
              learning_rate=2e-5, warmup_steps=0, output_model_name='un_base_model'):
        """
        Train the base model on UN general debates.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size (int): Batch size
            epochs (int): Number of epochs
            learning_rate (float): Learning rate
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
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
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
            
            progress_bar = tqdm(train_dataloader, desc="Training")
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
        plt.title('Training and Validation Loss')
        
        # Plot validation MSE
        plt.subplot(1, 2, 2)
        plt.plot(history['val_mse'], label='Val MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('Validation MSE')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close()


def main():
    """
    Main function to train the base model.
    """
    parser = argparse.ArgumentParser(description='Train base model on UN debates')
    parser.add_argument('--data_path', type=str, default='un_general_debates.csv',
                        help='Path to UN debates CSV file')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pre-trained model name')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--output_model_name', type=str, default='un_base_model',
                        help='Name for the saved model')
    
    args = parser.parse_args()
    
    # Initialize processor and load data
    processor = DataProcessor(max_length=args.max_length)
    un_df = processor.load_un_debates(args.data_path)
    
    # Split data
    un_train, un_val = processor.prepare_datasets(un_df, None)[0:2]
    
    # Create datasets
    train_dataset = processor.create_torch_dataset(un_train)
    val_dataset = processor.create_torch_dataset(un_val)
    
    # Initialize and train model
    base_model = BaseModel(model_name=args.model_name, output_dir=args.output_dir)
    base_model.train(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_model_name=args.output_model_name
    )
    
    print("Base model training complete!")


if __name__ == "__main__":
    main() 