import os
import json
from typing import Dict, List, Optional, Tuple
from text_processor import text_processor
from keyword_analysis import keyword_analyser
from feature_extraction import feature_extractor
from evaluation import evaluation_model
from feedback_generation import feedback_generator

# Constants
DEFAULT_MODEL_PATH = "mun_speech_evaluator.joblib"
SAFE_MODEL_PATH = "mun_speech_evaluator_safe.joblib"
TRAINING_DATA_PATH = "training_data.json"

class SpeechEvaluator:
    """
    Main interface to the MUN Speech Evaluator system.
    
    This class integrates the various modules to provide a complete
    speech evaluation solution.
    """
    
    def __init__(self):
        """
        Initialise the Speech Evaluator.
        """
        # Get the absolute path to the current directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create absolute paths for model and training data
        self.model_path = os.path.join(self.base_dir, DEFAULT_MODEL_PATH)
        self.training_data_path = os.path.join(self.base_dir, TRAINING_DATA_PATH)
        
        self.initialise_model()
        
    def _log(self, message):
        """Simple logging helper, does nothing by default."""
        pass
        
    def initialise_model(self) -> bool:
        """
        Initialise or reload the evaluation model.
        
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        # Try to load the model if it exists
        if os.path.exists(self.model_path):
            try:
                evaluation_model.load_model(self.model_path)
                self._log(f"Loaded model from {self.model_path}")
                return True
            except (FileNotFoundError, ValueError) as e:
                # Handle model loading errors
                self._log(f"Error loading model: {e}")
                self._log("Will create a new model")
            except Exception as e:
                # Handle any other unexpected errors
                self._log(f"Unexpected error loading model: {e}")
        
        # If we get here, we need to create a new model
        return self._train_new_model()
    
    def _train_new_model(self) -> bool:
        """
        Train a new model using available training data.
        
        Returns:
            bool: True if model was trained successfully, False otherwise
        """
        try:
            # Check if we have a training dataset
            if os.path.exists(self.training_data_path):
                self._log("Training a new model using available data...")
                try:
                    # Load training data
                    training_data = self.load_training_data()
                    if not training_data:
                        self._log("No training data available")
                        return False
                    
                    # Extract speeches, scores, and topics
                    speeches = [item['speech'] for item in training_data]
                    scores = [item['score'] for item in training_data]
                    
                    # Try training with topics first
                    try:
                        if all('topic' in item for item in training_data):
                            topics = [item['topic'] for item in training_data]
                            self._log(f"Training model with {len(speeches)} speeches and {len(set(topics))} topics...")
                            evaluation_model.train(speeches, scores, topics)
                            self._log("Model trained successfully with topic information")
                        else:
                            # Train without topics
                            evaluation_model.train(speeches, scores)
                            self._log("Model trained successfully without topics")
                        
                        # Save the model
                        try:
                            evaluation_model.save_model(self.model_path)
                            self._log(f"Saved model to {self.model_path}")
                            return True
                        except (ValueError, IOError) as e:
                            self._log(f"Error saving model: {e}")
                            return False
                    except Exception as e:
                        self._log(f"Error training model: {e}")
                        return False
                except Exception as e:
                    self._log(f"Error loading training data: {e}")
                    return False
            else:
                self._log(f"No training data found at {self.training_data_path}")
                return False
        except Exception as e:
            self._log(f"Unexpected error during model training: {e}")
            return False
    
    def evaluate_speech(self, speech: str, topic: str = None, safe_mode: bool = False) -> Dict:
        """
        Evaluate a speech and generate feedback.
        
        Args:
            speech (str): The speech text to evaluate
            topic (str, optional): The speech topic
            safe_mode (bool): Whether to use safe mode (avoid topic for scoring)
            
        Returns:
            Dict: Evaluation result with score and feedback
        """
        if not speech:
            raise ValueError("Speech text cannot be empty")
        
        try:
            # In safe mode, don't use topic for evaluation score
            if safe_mode and topic:
                self._log(f"Safe mode enabled: Using topic '{topic}' for feedback only, not for scoring")
                score = evaluation_model.evaluate(speech)
            else:
                try:
                    score = evaluation_model.evaluate(speech, topic)
                except ValueError as e:
                    error_str = str(e)
                    # Check if it's a feature dimension mismatch
                    if "features" in error_str and ("expecting" in error_str or "shape" in error_str):
                        self._log(f"Warning: Feature mismatch when using topic '{topic}'. Falling back to no topic.")
                        score = evaluation_model.evaluate(speech)
                    else:
                        # If it's not a feature mismatch issue, re-raise the error
                        raise e
            
            # Always use topic for feedback, as it doesn't affect feature dimensions
            feedback = feedback_generator.generate_feedback(speech, topic)
            
            return {
                'score': round(float(score), 1),
                'feedback': feedback,
                'safe_mode_used': safe_mode
            }
        except Exception as e:
            error_message = str(e)
            self._log(f"Error during evaluation: {error_message}")
            
            if "features" in error_message and "expecting" in error_message:
                error_message += " (Try a different topic, leave the topic field empty, or enable safe mode)"
                
            raise ValueError(error_message)
    
    def train_model(self, speeches: List[str], scores: List[float], topics: List[str] = None) -> bool:
        """
        Train the evaluation model with new data.
        
        Args:
            speeches (List[str]): List of speech texts
            scores (List[float]): List of scores (1-10)
            topics (List[str], optional): List of topics
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        if len(speeches) != len(scores):
            raise ValueError("Number of speeches must match number of scores")
            
        if len(speeches) < 3:
            raise ValueError("Need at least 3 speech samples for training")
        
        try:
            # Train the model
            if topics and len(topics) == len(speeches):
                self._log(f"Training model with {len(speeches)} speeches and topics...")
                evaluation_model.train(speeches, scores, topics)
            else:
                self._log(f"Training model with {len(speeches)} speeches (no topics)...")
                evaluation_model.train(speeches, scores)
            
            # Save the model
            try:
                evaluation_model.save_model(self.model_path)
                self._log(f"Saved model to {self.model_path}")
                return True
            except (ValueError, IOError) as e:
                self._log(f"Error saving model: {e}")
                return False
        except Exception as e:
            self._log(f"Error training model: {e}")
            return False
    
    def save_training_data(self, data: List[Dict]) -> bool:
        """
        Save training data to a JSON file.
        
        Args:
            data (List[Dict]): List of training data items
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.training_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self._log(f"Training data saved to {self.training_data_path}")
            return True
        except Exception as e:
            self._log(f"Error saving training data: {e}")
            return False
    
    def load_training_data(self) -> List[Dict]:
        """
        Load training data from a JSON file.
        
        Returns:
            List[Dict]: Training data items or empty list if file doesn't exist
        """
        if not os.path.exists(self.training_data_path):
            return []
            
        try:
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self._log(f"Error loading training data: {e}")
            return []
    
    def add_training_sample(self, speech: str, score: float, topic: str = None) -> bool:
        """
        Add a new training sample to the dataset.
        
        Args:
            speech (str): Speech text
            score (float): Score (1-10)
            topic (str, optional): Speech topic
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not speech:
            raise ValueError("Speech text cannot be empty")
            
        try:
            score = float(score)
            if not (1 <= score <= 10):
                raise ValueError("Score must be between 1 and 10")
        except (ValueError, TypeError):
            raise ValueError("Invalid score format")
        
        # Load existing data
        training_data = self.load_training_data()
        
        # Add new item
        training_data.append({
            'speech': speech,
            'score': score,
            'topic': topic
        })
        
        # Save updated data
        return self.save_training_data(training_data)
    
    def get_topics(self) -> List[str]:
        """
        Get list of unique topics from training data.
        
        Returns:
            List[str]: List of topics
        """
        # Print debug information
        print(f"Looking for training data at: {self.training_data_path}")
        print(f"File exists: {os.path.exists(self.training_data_path)}")
        
        data = self.load_training_data()
        print(f"Loaded {len(data)} training samples")
        
        topics = sorted(list(set(item.get('topic', '') for item in data if item.get('topic'))))
        print(f"Found {len(topics)} unique topics")
        
        return topics

# Create a global instance for shared access
speech_evaluator = SpeechEvaluator() 