import os
import torch
from transformers import AutoModelForSequenceClassification
from data_processor import DataProcessor
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from feature_extraction import feature_extractor

class SpeechEvaluationModel:
    """  
    This class handles model training, prediction, and persistence for
    evaluating MUN speeches.
    Model for evaluating MUN speeches using transfer learning.
    """
    
    def __init__(self):
        """Initialise the model and vectoriser."""
        self.model = None
        self.vectoriser = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.feature_names = None
        self.processor = DataProcessor(max_length=512)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                 ('rocm' if hasattr(torch, 'is_hip_available') and torch.is_hip_available() else 'cpu'))
        self.trained = False
        self.topic_encoder = None
    
    def train(self, speeches: List[str], scores: List[float], topics: List[str] = None) -> None:
        """
        Train the evaluation model using speech texts and scores.
        
        Args:
            speeches (List[str]): List of speech texts
            scores (List[float]): List of scores (1-10)
            topics (List[str], optional): List of speech topics
            
        Raises:
            ValueError: If input data is invalid
            speeches (list): List of speech texts
            scores (list): List of scores (1-10)
            topics (list, optional): List of speech topics
        """
        if not speeches or not scores:
            raise ValueError("Speeches and scores cannot be empty")
            
        if len(speeches) != len(scores):
            raise ValueError("Number of speeches must match number of scores")
            
        if topics and len(topics) != len(speeches):
            raise ValueError("Number of topics must match number of speeches")
        
        # Extract features from all speeches
        features_list = []
        text_list = []
        for i, speech in enumerate(speeches):
            try:
                # Extract features
                topic = topics[i] if topics else None
                extracted_features = feature_extractor.extract_features(speech, topic)
                flat_features = feature_extractor.get_flat_features(extracted_features)
                features_list.append(flat_features)
                
                # Get processed text for TF-IDF
                text_list.append(speech)
            except Exception as e:
                raise ValueError(f"Error processing speech {i}: {e}")
        # Process the speeches
        processed_data = self.processor.process_data(speeches, scores)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        # Create model if it doesn't exist
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=1
            )
            self.model.to(self.device)
        
        # Process text features (TF-IDF)
        text_features = self.vectoriser.fit_transform(text_list)
        # Set model to training mode
        self.model.train()
        
        # Check if we have topic information
        has_topics = topics is not None and 'topic' in features_df.columns
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            processed_data,
            batch_size=4,
            shuffle=True
        )
        
        # Prepare features
        numeric_features = features_df.select_dtypes(include=['number']).columns.tolist()
        # Training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        if has_topics:
            # One-hot encode topics as categorical features
            topic_values = features_df['topic'].values.reshape(-1, 1)
            self.topic_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            topic_encoded = self.topic_encoder.fit_transform(topic_values)
        for epoch in range(3):  # You can adjust the number of epochs
            total_loss = 0
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            # Combine numeric and text features with topics
            X = np.hstack([
                features_df[numeric_features].values,
                text_features.toarray(),
                topic_encoded
            ])
        else:
            # Combine numeric and text features without topics
            X = np.hstack([
                features_df[numeric_features].values,
                text_features.toarray()
            ])
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
        
        # Create target array
        y = np.array(scores)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.model.fit(X, y)
        
        # Save feature names for prediction
        self.feature_names = numeric_features
        self.trained = True
    
    def evaluate(self, speech: str, topic: str = None) -> float:
        """
        Evaluate a speech and return a score.
        
        Args:
            speech (str): The speech text to evaluate
            topic (str, optional): The speech topic
            
        Returns:
            float: Evaluation score between 1 and 10
            
        Raises:
            ValueError: If the model is not trained or there's an error
        """
        if not self.trained or self.model is None:
            raise ValueError("Model has not been trained yet")
        
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
    
    def save_model(self, file_path: str) -> bool:
        """
        Save the trained model to a file.
        # Process the speech
        processed_data = self.processor.process_data([speech], [0])  # Dummy score
        dataloader = torch.utils.data.DataLoader(processed_data, batch_size=1)
        
        Args:
            file_path (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValueError: If the model hasn't been trained
            IOError: If there's an error saving the file
        """
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
    
    def save_model(self, path: str) -> None:
        """Save the model to a file."""
        if not self.trained or self.model is None:
            raise ValueError("Cannot save: model has not been trained")
            raise ValueError("Model has not been trained yet")
        
        try:
            # Create a dictionary with all required objects
            model_data = {
                'model': self.model,
                'vectoriser': self.vectoriser,
                'feature_names': self.feature_names,
                'topic_encoder': self.topic_encoder
            }
            
            # Save to disk
            joblib.dump(model_data, file_path)
            return True
        except Exception as e:
            raise IOError(f"Failed to save model: {str(e)}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_pretrained(path)
    
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
        try:
            # Get the directory containing the model files
            model_dir = os.path.dirname(file_path)
            print(f"Loading model from directory: {model_dir}")
            
            # Load the BERT model with regression configuration
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                local_files_only=True,
                num_labels=1,  # For regression
                problem_type="regression"
            )
            print("Model loaded successfully")
            
            # Move model to device
            self.model.to(self.device)
            print(f"Model moved to device: {self.device}")
            
            # Set trained flag
            self.trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.trained = False
            return False

# Create a global instance for shared access
# Create a global instance
evaluation_model = SpeechEvaluationModel()