from text_processor import TextProcessor
from keyword_analysis import KeywordAnalysis

# class FeatureExtractor:
#     """Extract features from speech text for analysis and evaluation."""
    
#     def __init__(self, nlp, text_processor=None):
#         """
#         Initialize the feature extractor.
        
#         Args:
#             nlp: Loaded SpaCy model
#             text_processor: Optional TextProcessor instance
#         """
#         self.nlp = nlp
#         self.text_processor = text_processor or TextProcessor(nlp)
#         self.entity_analyzer = KeywordAnalysis(nlp)
        
#     def extract_features(self, text):
#         """
#         Extract features from speech text.
        
#         Args:
#             text: Speech text
            
#         Returns:
#             Dictionary of extracted features
#         """
#         processed_text = self.text_processor.preprocess_text(text)
#         doc = self.nlp(text)
        
#         # Basic statistics
#         word_count = len(text.split())
#         sentence_count = len(list(doc.sents))
        
#         # Calculate average word and sentence length
#         avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
#         avg_sentence_length = word_count / max(1, sentence_count)
        
#         # Count number of complex words (3+ syllables as a simple heuristic)
#         complex_words = sum(1 for word in text.split() if len(word) > 8)
        
#         # Analyse entities
#         entity_analysis = self.entity_analyzer.analyse(text)
        
#         # Analyse coherence between sentences
#         coherence = self.text_processor.analyse_coherence(doc)
        
#         # Return features as a dictionary
#         features = {
#             'processed_text': processed_text,
#             'word_count': word_count,
#             'sentence_count': sentence_count,
#             'avg_word_length': avg_word_length,
#             'avg_sentence_length': avg_sentence_length,
#             'complex_word_ratio': complex_words / max(1, word_count),
#             'entities': entity_analysis,
#             'coherence': coherence
#         }
        
#         return features
    
# def flatten_features(self, features):
#     """
#     Create a flattened version of features for model training/prediction.
    
#     Args:
#         features: Dictionary of extracted features
        
#     Returns:
#         Dictionary with flattened feature structure
#     """
#     # Get all entity categories
#     entities = features['entities']['categories']
    
#     # Base flattened features
#     flat_features = {
#         'processed_text': features['processed_text'],
#         'word_count': features['word_count'],
#         'sentence_count': features['sentence_count'],
#         'avg_word_length': features['avg_word_length'],
#         'avg_sentence_length': features['avg_sentence_length'],
#         'complex_word_ratio': features['complex_word_ratio'],
#         'avg_coherence': features['coherence']['avg_similarity'],
#         'weak_transition_count': len(features['coherence']['weak_transitions'])
#     }
    
#     # Add counts for each entity category
#     for category in entities:
#         category_key = f"{category.lower()}_count"
#         flat_features[category_key] = len(entities.get(category, []))
    
#     # Add total entity count
#     flat_features['total_entity_count'] = sum(len(entities.get(cat, [])) for cat in entities)
    
#     return flat_features

class FeatureExtractor:
    """Extract features from speech text for analysis and evaluation."""
    
    def __init__(self, nlp, text_processor=None):
        """
        Initialize the feature extractor.
        
        Args:
            nlp: Loaded SpaCy model
            text_processor: Optional TextProcessor instance
        """
        self.nlp = nlp
        self.text_processor = text_processor or TextProcessor(nlp)
        self.entity_analyzer = KeywordAnalysis(nlp)
        
    def extract_features(self, text):
        """
        Extract features from speech text.
        
        Args:
            text: Speech text
            
        Returns:
            Dictionary of extracted features
        """
        processed_text = self.text_processor.preprocess_text(text)
        doc = self.nlp(text)
        
        # Basic statistics
        word_count = len(text.split())
        sentence_count = len(list(doc.sents))
        
        # Calculate average word and sentence length
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Count number of complex words (3+ syllables as a simple heuristic)
        complex_words = sum(1 for word in text.split() if len(word) > 8)
        
        # Analyze entities
        entity_analysis = self.entity_analyzer.analyse(text)
        
        # Analyze coherence between sentences
        coherence = self.text_processor.analyse_coherence(doc)
        
        # Return features as a dictionary
        features = {
            'processed_text': processed_text,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'complex_word_ratio': complex_words / max(1, word_count),
            'entities': entity_analysis,
            'coherence': coherence
        }
        
        return features
    
    def flatten_features(self, features):
        """
        Create a flattened version of features for model training/prediction.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary with flattened feature structure
        """
        # Get all entity categories
        entities = features['entities']['categories']
        
        # Base flattened features
        flat_features = {
            'processed_text': features['processed_text'],
            'word_count': features['word_count'],
            'sentence_count': features['sentence_count'],
            'avg_word_length': features['avg_word_length'],
            'avg_sentence_length': features['avg_sentence_length'],
            'complex_word_ratio': features['complex_word_ratio'],
            'avg_coherence': features['coherence']['avg_similarity'],
            'weak_transition_count': len(features['coherence']['weak_transitions'])
        }
        
        # Add counts for each entity category
        for category in entities:
            category_key = f"{category.lower()}_count"
            flat_features[category_key] = len(entities.get(category, []))
        
        # Add total entity count
        flat_features['total_entity_count'] = sum(len(entities.get(cat, [])) for cat in entities)
        
        return flat_features