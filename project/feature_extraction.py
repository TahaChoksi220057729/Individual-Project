from typing import Dict, List, Any
from text_processor import text_processor
from keyword_analysis import keyword_analyser

class FeatureExtractor:
    """  
    This class combines text statistics, entity analysis, and coherence metrics
    to create a comprehensive feature set for speech evaluation.
    """
    
    def __init__(self):
        """Initialise the feature extractor."""
        pass
    
    def extract_features(self, text: str, topic: str = None) -> Dict[str, Any]:
        """
        Extract all features from a speech text.
        
        Args:
            text (str): The speech text to analyse
            topic (str, optional): The topic of the speech
            
        Returns:
            Dict[str, Any]: A dictionary of extracted features
        """
        if not text or not text.strip():
            raise ValueError("Speech text cannot be empty")
        
        # Compute text statistics
        text_stats = text_processor.compute_text_statistics(text)
        
        # Analyse coherence
        coherence_metrics = text_processor.analyse_coherence(text)
        
        # Extract entities and keywords
        entity_analysis = keyword_analyser.analyse(text)
        
        # Count entities by category
        entity_counts = keyword_analyser.get_entity_counts(entity_analysis)
        
        # Combine all features
        features = {
            'text_statistics': text_stats,
            'coherence_metrics': coherence_metrics,
            'entity_analysis': entity_analysis,
            'entity_counts': entity_counts
        }
        
        # Add topic if available
        if topic:
            features['topic'] = topic
        
        return features
    
    def get_flat_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert structured features to a flat dictionary suitable for ML models.
        
        Args:
            features (Dict[str, Any]): The structured features from extract_features
            
        Returns:
            Dict[str, float]: A flattened dictionary of features
        """
        flat_features = {}
        
        # Add text statistics
        for key, value in features['text_statistics'].items():
            flat_features[f'stat_{key}'] = value
        
        # Add coherence metrics
        for key, value in features['coherence_metrics'].items():
            flat_features[f'coherence_{key}'] = value
        
        # Add entity counts
        for key, value in features['entity_counts'].items():
            flat_features[f'entity_{key}'] = value
        
        # Add entity density (entities per 100 words)
        if 'word_count' in features['text_statistics'] and features['text_statistics']['word_count'] > 0:
            word_count = features['text_statistics']['word_count']
            for key, value in features['entity_counts'].items():
                if key != 'total':  # Skip total to avoid redundancy
                    flat_features[f'entity_density_{key}'] = (value / word_count) * 100
        
        # Add topic if available (as categorical feature for ML)
        if 'topic' in features:
            flat_features['topic'] = features['topic']
        
        return flat_features

# Allow global access to this class
feature_extractor = FeatureExtractor() 