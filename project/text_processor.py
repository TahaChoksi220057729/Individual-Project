import os
import re
import spacy
import nltk
from nltk.corpus import stopwords
from typing import Dict, List, Tuple, Any
import warnings

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Suppress spaCy warnings
warnings.filterwarnings("ignore", message=".*word vectors.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="spacy")

class TextProcessor:
    """   
    This class handles text preprocessing, tokenisation, and basic
    statistical analysis of text.
    """
    
    def __init__(self):
        """Initialise the text processor and load NLP model."""
        self.nlp = self._load_spacy_model()
        self.stop_words = self.nlp.Defaults.stop_words
        
    def _load_spacy_model(self):
        """
        Load the SpaCy language model.
        
        Returns:
            spacy.Language: Loaded SpaCy model
            
        Note:
            Will attempt to download the model if not available
        """
        try:
            # Try to load the model
            model = spacy.load("en_core_web_sm")
            return model
        except OSError:
            # If model not found, download it
            print("Downloading SpaCy language model...")
            try:
                # First try to download via spacy.cli
                spacy.cli.download("en_core_web_sm")
                model = spacy.load("en_core_web_sm")
                return model
            except:
                # Fallback for environments where spacy.cli might not work
                print("Could not download model automatically. Please run:")
                print("python -m spacy download en_core_web_sm")
                # Return a minimal English model as fallback
                return spacy.blank("en")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess and clean text for analysis.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not text or not text.strip():
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", " ", text)
        
        # Process with SpaCy
        doc = self.nlp(text)
        
        # Lemmatise and remove stop words
        processed_tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                processed_tokens.append(token.lemma_)
        
        # Join tokens back to string
        preprocessed_text = " ".join(processed_tokens)
        
        return preprocessed_text
    
    def compute_text_statistics(self, text: str) -> Dict[str, float]:
        """
        Compute basic statistics about the text.
        
        Args:
            text (str): Text to analyse
            
        Returns:
            Dict[str, float]: Dictionary of text statistics
        """
        if not text or not text.strip():
            return {
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0.0,
                "avg_sentence_length": 0.0,
                "complex_word_ratio": 0.0
            }
        
        # Process with SpaCy for sentence segmentation
        doc = self.nlp(text)
        
        # Count words (excluding punctuation and whitespace)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        word_count = len(words)
        
        # Count sentences
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        
        # Calculate average word length
        if word_count > 0:
            avg_word_length = sum(len(word) for word in words) / word_count
        else:
            avg_word_length = 0.0
        
        # Calculate average sentence length
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
        else:
            avg_sentence_length = 0.0
        
        # Calculate complex word ratio (words with 3+ syllables)
        complex_words = 0
        for word in words:
            syllable_count = self.count_syllables(word)
            if syllable_count >= 3:
                complex_words += 1
        
        if word_count > 0:
            complex_word_ratio = complex_words / word_count
        else:
            complex_word_ratio = 0.0
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "complex_word_ratio": complex_word_ratio
        }
    
    def count_syllables(self, word: str) -> int:
        """
        Count syllables in a word using a basic heuristic.
        
        Args:
            word (str): Word to count syllables in
            
        Returns:
            int: Estimated syllable count
        """
        # Lower case the word
        word = word.lower()
        
        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # Adjust for common patterns
        if word.endswith("e") and not word.endswith("le"):
            count -= 1
        if word.endswith("es") or word.endswith("ed"):
            count -= 1
        
        # Ensure at least one syllable
        return max(1, count)
    
    def analyse_coherence(self, text: str) -> Dict[str, Any]:
        """
        Analyse coherence between sentences in the text.
        
        Args:
            text (str): Text to analyse
            
        Returns:
            Dict[str, Any]: Coherence metrics including average similarity
                           and identified weak transitions
        """
        if not text or not text.strip():
            return {
                "avg_similarity": 0.0,
                "weak_transitions": []
            }
        
        # Process with SpaCy
        doc = self.nlp(text)
        
        # Extract sentences
        sentences = list(doc.sents)
        
        # If fewer than 2 sentences, can't measure coherence
        if len(sentences) < 2:
            return {
                "avg_similarity": 1.0,  # Perfect coherence for single sentence
                "weak_transitions": []
            }
        
        # Calculate similarity between adjacent sentences
        similarities = []
        weak_transitions = []
        
        for i in range(len(sentences) - 1):
            current = sentences[i]
            next_sent = sentences[i + 1]
            
            # Skip empty or very short sentences
            if len(current) < 3 or len(next_sent) < 3:
                continue
            
            # Calculate similarity
            similarity = current.similarity(next_sent)
            similarities.append(similarity)
            
            # Track weak transitions
            if similarity < 0.4:  # Threshold for weak transition
                weak_transitions.append({
                    "position": i + 1,  # 1-indexed position between sentences
                    "sentences": [current.text, next_sent.text],
                    "similarity": similarity
                })
        
        # Calculate average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return {
            "avg_similarity": avg_similarity,
            "weak_transitions": weak_transitions
        }

# Allow global access to this class
text_processor = TextProcessor() 