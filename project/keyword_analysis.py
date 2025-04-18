from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import spacy
from spacy.tokens import Doc
from text_processor import text_processor

class KeywordAnalyser:
    """    
    This class identifies named entities, categorises them, and extracts 
    significant noun phrases and key concepts from the text.
    """
    
    # Entity categories mapping to SpaCy 
    CATEGORIES = {
        'COUNTRIES': {'GPE', 'LOC', 'NORP'},
        'ORGANISATIONS': {'ORG'},
        'POLICIES': {'LAW', 'WORK_OF_ART'},
        'QUANTITIES': {'PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL'},
        'DATES': {'DATE', 'TIME'},
        'KEY_CONCEPTS': {'EVENT'}
    }
    
    def __init__(self):
        """Initialise the keyword analyser."""
        self.nlp = text_processor.nlp
    
    def analyse(self, text: str) -> Dict[str, Any]:
        """
        Analyse text to identify and categorise named entities.
        
        Args:
            text (str): The speech text to analyse
            
        Returns:
            Dict: Dictionary containing categorised entities, all entities,
                  key phrases, and statistics
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Process the text with SpaCy
        doc = self.nlp(text)
        
        # Initialise results structure with categorised entities
        results = {
            'categorised_entities': {
                'COUNTRIES': set(),
                'ORGANISATIONS': set(),
                'POLICIES': set(),
                'QUANTITIES': set(),
                'DATES': set(),
                'KEY_CONCEPTS': set()
            },
            'all_entities': {},
            'key_phrases': set(),
            'statistics': {
                'total_entities': 0,
                'unique_entities': 0
            }
        }
        
        # Extract entities and categorise them
        entity_counts = Counter()
        
        for ent in doc.ents:
            # Skip short entities that are likely not meaningful
            if len(ent.text) < 2:
                continue
                
            # Add to all entities with count
            entity_counts[ent.text] += 1
            
            # Categorise the entity
            for category, labels in self.CATEGORIES.items():
                if ent.label_ in labels:
                    results['categorised_entities'][category].add(ent.text)
                    break
        
        # Convert entity sets to lists for better serialisation
        for category in results['categorised_entities']:
            results['categorised_entities'][category] = list(results['categorised_entities'][category])
        
        # Add key phrases
        results['key_phrases'] = list(self._extract_key_phrases(doc))
        
        # Add statistics
        results['statistics']['total_entities'] = sum(entity_counts.values())
        results['statistics']['unique_entities'] = len(entity_counts)
        
        # Add all entities with counts
        results['all_entities'] = dict(entity_counts)
        
        # Get most mentioned entities for each category
        results['most_mentioned'] = {}
        for category, entities in results['categorised_entities'].items():
            if entities:
                category_counts = {e: entity_counts[e] for e in entities}
                results['most_mentioned'][category] = self._get_most_mentioned(category_counts)
        
        return results
    
    def _extract_key_phrases(self, doc: Doc) -> Set[str]:
        """
        Extract significant noun phrases from the document.
        
        Args:
            doc (Doc): SpaCy document
            
        Returns:
            Set[str]: Set of key phrases
        """
        key_phrases = set()
        
        for chunk in doc.noun_chunks:
            # Clean up the chunk text
            phrase = chunk.text.strip().lower()
            
            # Skip single words (these are covered by entities)
            if len(phrase.split()) < 2:
                continue
                
            # Skip phrases that are mostly stop words
            if sum(1 for token in chunk if not token.is_stop) < 2:
                continue
                
            # Skip phrases with numbers
            if any(char.isdigit() for char in phrase):
                continue
                
            # Add the phrase
            key_phrases.add(phrase)
        
        return key_phrases
    
    def _get_most_mentioned(self, entity_counts: Dict[str, int], limit: int = 3) -> List[Tuple[str, int]]:
        """
        Get the most frequently mentioned entities.
        
        Args:
            entity_counts (Dict[str, int]): Dictionary of entity counts
            limit (int): Maximum number of entities to return
            
        Returns:
            List[Tuple[str, int]]: List of (entity, count) tuples, sorted by count
        """
        return sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_entity_counts(self, analysis_result: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract counts of entities from the analysis result.
        
        Args:
            analysis_result (Dict): Result from the analyse method
            
        Returns:
            Dict[str, int]: Dictionary of entity counts by category
        """
        counts = {
            'total': analysis_result['statistics']['total_entities'],
            'countries': len(analysis_result['categorised_entities']['COUNTRIES']),
            'organisations': len(analysis_result['categorised_entities']['ORGANISATIONS']),
            'policies': len(analysis_result['categorised_entities']['POLICIES']),
            'quantities': len(analysis_result['categorised_entities']['QUANTITIES']),
            'dates': len(analysis_result['categorised_entities']['DATES']),
            'key_concepts': len(analysis_result['categorised_entities']['KEY_CONCEPTS']),
            'key_phrases': len(analysis_result['key_phrases'])
        }
        
        return counts

# Create a global instance for shared access
keyword_analyser = KeywordAnalyser() 