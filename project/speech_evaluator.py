import os
import json
from typing import Dict, List, Optional, Tuple
from text_processor import text_processor
from keyword_analysis import keyword_analyser
from feature_extraction import feature_extractor
from evaluation import evaluation_model
from feedback_generation import feedback_generator

# Constants
DEFAULT_MODEL_PATH = "output/models/mun_transfer_model/model.safetensors"
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
        Initialise the evaluation model by loading it from disk.
        
        Returns:
            bool: True if model was loaded successfully, False otherwise
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(self.model_path):
            error_msg = f"Model file not found at {self.model_path}. Please ensure the model file exists."
            self._log(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            evaluation_model.load_model(self.model_path)
            self._log(f"Loaded model from {self.model_path}")
            return True
        except Exception as e:
            error_msg = f"Error loading model: {e}"
            self._log(error_msg)
            raise ValueError(error_msg)

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
    
    def get_topics(self) -> List[str]:
        """
        Get list of unique topics from training data.
        
        Returns:
            List[str]: List of topics
        """
        data = self.load_training_data()
        topics = sorted(list(set(item.get('topic', '') for item in data if item.get('topic'))))
        return topics

# Create a global instance for shared access
speech_evaluator = SpeechEvaluator() 