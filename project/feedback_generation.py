from typing import Dict, List, Any
from feature_extraction import feature_extractor
from evaluation import evaluation_model

class FeedbackGenerator:
    """    
    This class analyses the features of a speech and provides structured
    feedback, including strengths, weaknesses, and suggestions for improvement.
    """
    
    def __init__(self):
        """Initialise the feedback generator."""
        pass
    
    def generate_feedback(self, speech: str, topic: str = None) -> Dict[str, Any]:
        """
        Generate detailed feedback for a speech.
        
        Args:
            speech (str): The speech text to analyse
            topic (str, optional): The topic of the speech
            
        Returns:
            Dict[str, Any]: Structured feedback object
        """
        if not speech or not speech.strip():
            raise ValueError("Speech text cannot be empty")
        
        # Extract features
        features = feature_extractor.extract_features(speech, topic)
        
        # Get evaluation score
        score = evaluation_model.evaluate(speech, topic)
        
        # Extract statistics for feedback
        text_stats = features['text_statistics']
        coherence = features['coherence_metrics']
        entity_analysis = features['entity_analysis']
        
        # Determine strengths and weaknesses based on features
        feedback = {
            'overall_score': score,
            'content_analysis': self._analyse_content(features, score),
            'structure': self._analyse_structure(features, score),
            'language_metrics': {
                'word_count': text_stats['word_count'],
                'average_word_length': round(text_stats['avg_word_length'], 1),
                'average_sentence_length': round(text_stats['avg_sentence_length'], 1),
                'sentences': text_stats['sentence_count'],
                'complex_word_ratio': f"{round(text_stats['complex_word_ratio'] * 100, 1)}%",
                'coherence_score': round(coherence['avg_similarity'] * 10, 1)
            },
            'key_entities': {
                'Countries': entity_analysis['categorised_entities']['COUNTRIES'][:254],
                'Organisations': entity_analysis['categorised_entities']['ORGANISATIONS'][:254],
                'Policies': entity_analysis['categorised_entities']['POLICIES'][:254],
                'Key Concepts': entity_analysis['categorised_entities']['KEY_CONCEPTS'][:254]
            },
            'suggestions': self._generate_suggestions(features, score)
        }
        
        return feedback
    
    def _analyse_content(self, features: Dict[str, Any], score: float) -> str:
        """
        Analyse the content of the speech.
        
        Args:
            features (Dict[str, Any]): Extracted features
            score (float): Overall evaluation score
            
        Returns:
            str: Content analysis feedback
        """
        entity_analysis = features['entity_analysis']
        text_stats = features['text_statistics']
        
        # Base analysis on entity counts and diversity
        countries = len(entity_analysis['categorised_entities']['COUNTRIES'])
        orgs = len(entity_analysis['categorised_entities']['ORGANISATIONS'])
        policies = len(entity_analysis['categorised_entities']['POLICIES'])
        quantities = len(entity_analysis['categorised_entities']['QUANTITIES'])
        key_phrases = len(entity_analysis['key_phrases'])
        
        if score >= 8.0:
            return (
                f"Excellent content with comprehensive coverage of the topic. "
                f"You referenced {countries} countries, {orgs} organisations, "
                f"and {policies} policies/regulations, demonstrating thorough research. "
                f"Your use of {quantities} quantitative references strengthens your argument with solid evidence."
            )
        elif score >= 6.0:
            return (
                f"Good content with adequate topic coverage. You referenced {countries} countries "
                f"and {orgs} organisations, showing good knowledge. Consider incorporating more "
                f"specific policy references (currently {policies}) and quantitative evidence "
                f"(currently {quantities} references) to strengthen your arguments."
            )
        else:
            return (
                f"Basic content that needs more depth. Your speech references {countries} countries "
                f"and {orgs} organisations, but would benefit from more specific policy references "
                f"(currently {policies}) and supporting statistics (currently {quantities} quantitative references). "
                f"Expand on key concepts to demonstrate deeper understanding of the topic."
            )
    
    def _analyse_structure(self, features: Dict[str, Any], score: float) -> str:
        """
        Analyse the structure of the speech.
        
        Args:
            features (Dict[str, Any]): Extracted features
            score (float): Overall evaluation score
            
        Returns:
            str: Structure analysis feedback
        """
        coherence = features['coherence_metrics']
        text_stats = features['text_statistics']
        
        avg_coherence = coherence['avg_similarity']
        weak_transitions = coherence['weak_transitions']
        sentence_count = text_stats['sentence_count']
        
        if avg_coherence >= 0.7:
            return (
                f"Strong speech structure with excellent flow between points. "
                f"Your speech has {sentence_count} sentences with a coherence score of "
                f"{round(avg_coherence * 10, 1)}/10, indicating smooth transitions between ideas. "
                f"Only {len(weak_transitions)} potential transition issues were identified."
            )
        elif avg_coherence >= 0.5:
            return (
                f"Good speech structure with generally smooth flow. Your speech has "
                f"{sentence_count} sentences with a coherence score of {round(avg_coherence * 10, 1)}/10. "
                f"There are {len(weak_transitions)} points where transitions could be strengthened "
                f"to improve the overall flow."
            )
        else:
            return (
                f"Basic speech structure that needs improvement. Your coherence score is "
                f"{round(avg_coherence * 10, 1)}/10, indicating some disconnect between ideas. "
                f"There are {len(weak_transitions)} weak transitions in your {sentence_count}-sentence speech. "
                f"Consider using more transition phrases and ensuring each point builds logically from the previous one."
            )
    
    def _generate_suggestions(self, features: Dict[str, Any], score: float) -> List[str]:
        """
        Generate improvement suggestions based on speech features.
        
        Args:
            features (Dict[str, Any]): Extracted features
            score (float): Overall evaluation score
            
        Returns:
            List[str]: List of suggestions for improvement
        """
        suggestions = []
        text_stats = features['text_statistics']
        coherence = features['coherence_metrics']
        entity_counts = features['entity_counts']
        
        # Word count suggestions
        word_count = text_stats['word_count']
        if word_count < 200:
            suggestions.append(f"Expand your speech (currently {word_count} words). A strong MUN speech typically contains at least 300-500 words to allow for adequate development of ideas.")
        elif word_count > 800:
            suggestions.append(f"Consider tightening your speech (currently {word_count} words). While comprehensive, very long speeches may lose audience attention. Aim for 400-700 words for optimal impact.")
        
        # Sentence length suggestions
        avg_sentence_length = text_stats['avg_sentence_length']
        if avg_sentence_length > 30:
            suggestions.append(f"Break down some longer sentences. Your average sentence length is {round(avg_sentence_length, 1)} words, which may be difficult to follow. Consider using shorter, clearer sentences for key points.")
        elif avg_sentence_length < 10:
            suggestions.append(f"Combine some shorter sentences to improve flow. Your average sentence length is {round(avg_sentence_length, 1)} words, which may create a choppy rhythm. Varying sentence length can make your speech more engaging.")
        
        # Coherence suggestions
        avg_coherence = coherence['avg_similarity']
        weak_transitions = len(coherence['weak_transitions'])
        if avg_coherence < 0.6 and weak_transitions > 2:
            suggestions.append(f"Improve the flow between ideas. There are {weak_transitions} points where transitions could be strengthened. Consider using transition phrases like 'Furthermore,' 'In addition,' or 'As a result' to connect related points.")
        
        # Entity usage suggestions
        if entity_counts['countries'] < 3:
            suggestions.append("Include more specific country references. Mentioning relevant nations affected by the issue adds concreteness to your arguments.")
        
        if entity_counts['organisations'] < 2:
            suggestions.append("Reference more international organisations. Mentioning bodies like the UN, WHO, or regional organisations demonstrates awareness of international frameworks.")
        
        if entity_counts['policies'] < 2:
            suggestions.append("Incorporate more specific policy references. Mentioning treaties, resolutions, or legal frameworks strengthens your arguments with institutional backing.")
        
        if entity_counts['quantities'] < 3:
            suggestions.append("Add more statistics and figures to support your arguments. Quantitative evidence makes your speech more persuasive and factually grounded.")
        
        # Add general suggestions based on score
        if score < 7.0:
            suggestions.append("Research the topic more deeply to demonstrate greater expertise. Strong MUN speeches show comprehensive understanding of both the topic and relevant international frameworks.")
        
        if score < 5.0:
            suggestions.append("Structure your speech with a clear introduction, body, and conclusion. A well-organised speech helps delegates follow your arguments and position.")
        
        return suggestions
    
# Create a global instance for shared access
feedback_generator = FeedbackGenerator() 