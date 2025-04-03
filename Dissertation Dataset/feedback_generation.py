class FeedbackGenerator:
    """Generate detailed feedback for MUN speeches."""
    
    def __init__(self, model=None):
        """
        Initialize the feedback generator.
        
        Args:
            model: Optional trained model for scoring
        """
        self.model = model
        
    def generate_feedback(self, features, speech_score=None):
        """
        Generate detailed feedback based on extracted features.
        
        Args:
            features: Dictionary of extracted features
            speech_score: Optional pre-calculated speech score
            
        Returns:
            Dictionary with detailed feedback
        """

        feedback = {}
        
        # Content assessment
        feedback['content'] = {
            'word_count': features['word_count'],

            'recommended_word_count': '300-500 words for a standard MUN speech',

            'entity_analysis': features['entities'],
            'key_entities': features['entities']['categories']
        }
        
        # Structure assessment
        feedback['structure'] = {
            'sentence_count': features['sentence_count'],
            'avg_sentence_length': round(features['avg_sentence_length'], 2),

            'recommendation': 'Aim for 15-20 words per sentence for clarity'
        }
        
        # Language complexity
        feedback['language'] = {
            'avg_word_length': round(features['avg_word_length'], 2),
            'complex_word_ratio': round(features['complex_word_ratio'] * 100, 2),

            'recommendation': 'A mix of simple and complex terms is ideal'
        }
        
        # Overall score
        if speech_score is not None:
            feedback['overall_score'] = round(speech_score, 2)
        else:
            feedback['overall_score'] = 'Score not available'

        # Generate improvement suggestions
        suggestions = []
        
        if features['avg_sentence_length'] > 25:

            suggestions.append("Consider using shorter sentences for clarity.")
        
        if features['complex_word_ratio'] > 0.3:

            suggestions.append("Speech may be too complex. Simplify some language.")
        elif features['complex_word_ratio'] < 0.1:

            suggestions.append("Consider using more sophisticated vocabulary where appropriate.")
            
        if features['word_count'] < 200:

            suggestions.append("Speech may be too brief. Expand on key points.")
        elif features['word_count'] > 600:

            suggestions.append("Speech may be too long. Focus on conciseness.")

        if 'coherence' in features:

            coherence = features['coherence']
            
            if coherence['weak_transitions']:
                for transition in coherence['weak_transitions']:

                    suggestions.append(

                        f"Consider improving the transition between: "
                        f"'{transition['sentences'][0]}' and '{transition['sentences'][1]}'. "
                        f"Add connecting phrases or transition words."
                    )
            
            if coherence['avg_similarity'] < 0.3:

                suggestions.append(
                    
                    "The speech could benefit from stronger topic connections. "
                    "Consider using more linking words and maintaining consistent themes."
                )
                
        if not features['entities']['categories'].get('ORGANIZATIONS', []):

            suggestions.append(

                "Consider mentioning relevant UN bodies or organizations to strengthen your argument."
            )
    
        if not features['entities']['categories'].get('POLICIES', []):

            suggestions.append(

                "Including specific policies, resolutions, or treaties could add credibility to your speech."
            )
    
        if not features['entities']['categories'].get('QUANTITIES', []):

            suggestions.append(

                "Adding specific statistics or figures could make your arguments more compelling."
            )
            
        feedback['suggestions'] = suggestions
        
        return feedback