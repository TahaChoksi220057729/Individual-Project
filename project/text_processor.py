import re
import spacy

class TextProcessor:
    """Handle text preprocessing and analysis for MUN speeches."""
    
    def __init__(self, nlp):
        """
        Initialize with SpaCy NLP model.
        
        Args:
            nlp: Loaded SpaCy model
        """
        self.nlp = nlp
        
    def preprocess_text(self, text):
        """
        Clean and preprocess the speech text.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Processed text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenization and lemmatization with SpaCy
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return " ".join(tokens)
    
    def analyse_coherence(self, doc):
        """
        Analyze the coherence between sentences in a document.
        
        Args:
            doc: SpaCy document
            
        Returns:
            Dictionary with coherence metrics
        """
        sentences = list(doc.sents)
        weak_transitions = []
        similarities = []
        
        for i in range(len(sentences) - 1):
            current_sent = sentences[i]
            next_sent = sentences[i + 1]
            
            # Calculate similarity between consecutive sentences
            similarity = current_sent.similarity(next_sent)
            similarities.append(similarity)
            
            # Track significant drops in similarity (potential topic shifts)
            if similarity < 0.3:  # Threshold for weak transitions
                weak_transitions.append({
                    'position': i + 1,
                    'sentences': (str(current_sent).strip(), str(next_sent).strip()),
                    'similarity': similarity
                })
        
        return {
            'weak_transitions': weak_transitions,
            'avg_similarity': sum(similarities) / max(len(similarities), 1)
        }