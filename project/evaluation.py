import os
import torch
from transformers import AutoModelForSequenceClassification
from data_processor import DataProcessor
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import joblib
from feature_extraction import feature_extractor

class SpeechEvaluationModel:
    """  
    This class handles model loading and prediction for evaluating MUN speeches.
    Requires a pre-trained model to function.
    """
    
    def __init__(self):
        """Initialise the model components."""
        self.model = None
        self.vectoriser = None
        self.feature_names = None
        self.processor = DataProcessor(max_length=512)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                 ('rocm' if hasattr(torch, 'is_hip_available') and torch.is_hip_available() else 'cpu'))
        self.trained = False
        self.topic_encoder = None
    
    def evaluate(self, speech: str, topic: str = None) -> float:
        """
        Evaluate a speech and return a score.
        
        Args:
            speech (str): The speech text to evaluate
            topic (str, optional): The speech topic
            
        Returns:
            float: Evaluation score between 1 and 10
            
        Raises:
            ValueError: If the model is not loaded or there's an error
        """
        if not self.trained or self.model is None:
            raise ValueError("Model has not been loaded yet")
        
        try:
            # Process the speech
            processed_data = self.processor.process_data([speech], [0])  # Dummy score
            dataloader = torch.utils.data.DataLoader(processed_data, batch_size=1)
            
            # Set model to evaluation mode
            self.model.eval()
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    score = outputs.logits.item()
            
            # Clip score to range 1-10
            return max(1.0, min(10.0, float(score)))
        except Exception as e:
            raise ValueError(f"Error evaluating speech: {str(e)}")
    
    def load_model(self, file_path: str) -> bool:
        """
        Load a trained model from file.
        
        Args:
            file_path (str): Path to the model file
            
        Returns:
            bool: True if model was loaded successfully
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")
            
        try:
            # Get the directory containing the model files
            model_dir = os.path.dirname(file_path)
            
            # Load the model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                local_files_only=True,
                num_labels=1,  # For regression
                problem_type="regression"
            )
            
            # Move model to device
            self.model.to(self.device)
            
            # Try to load extra components
            extra_path = os.path.join(model_dir, "extra_components.joblib")
            if os.path.exists(extra_path):
                try:
                    extra_data = joblib.load(extra_path)
                    self.vectoriser = extra_data.get('vectoriser', self.vectoriser)
                    self.feature_names = extra_data.get('feature_names', None)
                    self.topic_encoder = extra_data.get('topic_encoder', None)
                except Exception as e:
                    print(f"Warning: Could not load extra components: {e}")
            
            # Set trained flag
            self.trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.trained = False
            raise ValueError(f"Error loading model: {str(e)}")

# Create a global instance for shared access
evaluation_model = SpeechEvaluationModel()