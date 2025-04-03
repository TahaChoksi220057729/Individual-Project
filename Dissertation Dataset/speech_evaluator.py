import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from nltk.corpus import stopwords
import joblib
import spacy

from text_processor import TextProcessor
from feature_extraction import FeatureExtractor
from feedback_generation import FeedbackGenerator

class MUNSpeechEvaluator:
    """Main class for evaluating and providing feedback on MUN speeches."""
    
    def __init__(self):
        """Initialize the MUN speech evaluator."""
        self.nlp = spacy.load("en_core_web_md")
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words=stopwords.words('english'),
            ngram_range=(1, 3)
        )
        self.text_processor = TextProcessor(self.nlp)
        self.feature_extractor = FeatureExtractor(self.nlp, self.text_processor)
        self.feedback_generator = FeedbackGenerator()
        
    def train(self, speeches, scores):
        """
        Train the model on labeled MUN speeches.
        
        Args:
            speeches: List of speech texts
            scores: List of corresponding scores
            
        Returns:
            Self for method chaining
        """
        # Extract features for all speeches
        features_list = []
        for speech in speeches:
            features = self.feature_extractor.extract_features(speech)
            flat_features = self.feature_extractor.flatten_features(features)
            features_list.append(flat_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Extract text for vectorization
        X_text = features_df['processed_text']
        
        # Convert text to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X_text)
        
        # Get numerical features
        numerical_features = features_df.drop('processed_text', axis=1)
        
        # Combine TF-IDF and numerical features
        X_tfidf_array = X_tfidf.toarray()
        X_combined = np.hstack((X_tfidf_array, numerical_features.values))
        
        # Create and train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_combined, scores)
        
        # Update feedback generator with the model
        self.feedback_generator.model = self.model
        
        return self
    
    def evaluate(self, speech):
        """
        Evaluate a single MUN speech.
        
        Args:
            speech: Text of the speech to evaluate
            
        Returns:
            Predicted score for the speech
        """
        # Extract features
        features = self.feature_extractor.extract_features(speech)
        
        # Create flattened features
        flat_features = self.feature_extractor.flatten_features(features)
        
        # Convert to DataFrame with single row
        features_df = pd.DataFrame([flat_features])
        
        # Process text
        X_text = features_df['processed_text']
        
        # Convert text to TF-IDF features
        X_tfidf = self.vectorizer.transform(X_text)
        
        # Get numerical features
        numerical_features = features_df.drop('processed_text', axis=1)
        
        # Combine TF-IDF and numerical features
        X_tfidf_array = X_tfidf.toarray()
        X_combined = np.hstack((X_tfidf_array, numerical_features.values))
        
        # Make prediction
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        score = self.model.predict(X_combined)[0]
        
        return score
    
    def generate_feedback(self, speech):
        """
        Generate detailed feedback for a MUN speech.
        
        Args:
            speech: Text of the speech to evaluate
            
        Returns:
            Dictionary with detailed feedback
        """
        # Extract features for analysis
        features = self.feature_extractor.extract_features(speech)
        
        # Calculate score if model is available
        score = None
        if self.model is not None:
            score = self.evaluate(speech)
            
        # Generate feedback
        return self.feedback_generator.generate_feedback(features, score)
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        
        # Update feedback generator with the model
        self.feedback_generator.model = self.model
        
        return self