import os
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
    Class responsible for the machine learning aspects of speech evaluation.
    
    This class handles model training, prediction, and persistence for
    evaluating MUN speeches.
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
                print(f"Error processing speech {i}: {e}")
                raise ValueError(f"Error processing speech {i}: {e}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Process text features (TF-IDF)
        text_features = self.vectoriser.fit_transform(text_list)
        
        # Check if we have topic information
        has_topics = topics is not None and 'topic' in features_df.columns
        
        # Prepare features
        numeric_features = features_df.select_dtypes(include=['number']).columns.tolist()
        
        if has_topics:
            # One-hot encode topics as categorical features
            topic_values = features_df['topic'].values.reshape(-1, 1)
            self.topic_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            topic_encoded = self.topic_encoder.fit_transform(topic_values)
            
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
            # Extract features
            extracted_features = feature_extractor.extract_features(speech, topic)
            flat_features = feature_extractor.get_flat_features(extracted_features)
            
            # Convert to DataFrame (only numeric features)
            features_df = pd.DataFrame([flat_features])
            numeric_features = features_df.select_dtypes(include=['number']).columns.tolist()
            
            # Make sure we have all the required features
            missing_features = set(self.feature_names) - set(numeric_features)
            if missing_features:
                # Add missing features with zero values
                for feature in missing_features:
                    features_df[feature] = 0
            
            # Process text feature
            text_feature = self.vectoriser.transform([speech])
            
            # Check if we need to handle topic
            if self.topic_encoder is not None and topic:
                topic_value = np.array([[topic]])
                try:
                    topic_encoded = self.topic_encoder.transform(topic_value)
                    
                    # Combine features
                    X = np.hstack([
                        features_df[self.feature_names].values,
                        text_feature.toarray(),
                        topic_encoded
                    ])
                except (ValueError, TypeError):
                    # Topic wasn't in training data, proceed without topic
                    X = np.hstack([
                        features_df[self.feature_names].values,
                        text_feature.toarray(),
                        np.zeros((1, self.topic_encoder.get_feature_names_out().shape[0]))
                    ])
            else:
                # Combine features without topic
                X = np.hstack([
                    features_df[self.feature_names].values,
                    text_feature.toarray()
                ])
            
            # Predict score
            score = self.model.predict(X)[0]
            
            # Clip score to range 1-10
            return max(1.0, min(10.0, float(score)))
        except Exception as e:
            raise ValueError(f"Error evaluating speech: {str(e)}")
    
    def save_model(self, file_path: str) -> bool:
        """
        Save the trained model to a file.
        
        Args:
            file_path (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.trained or self.model is None:
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
            
            # Save model and necessary components
            joblib.dump({
                'model': self.model,
                'vectoriser': self.vectoriser,
                'feature_names': self.feature_names,
                'topic_encoder': self.topic_encoder
            }, file_path)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, file_path: str) -> bool:
        """
        Load a trained model from a file.
        
        Args:
            file_path (str): Path to the model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
                
            # Load model and components
            saved_data = joblib.load(file_path)
            
            self.model = saved_data['model']
            self.vectoriser = saved_data['vectoriser']
            self.feature_names = saved_data['feature_names']
            self.topic_encoder = saved_data.get('topic_encoder')
            self.trained = True
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

# Create a global instance for shared access
evaluation_model = SpeechEvaluationModel() 