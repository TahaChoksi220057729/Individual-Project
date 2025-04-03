"""Entity analysis for MUN speeches."""

from collections import defaultdict
from typing import Dict, List, Set
import spacy
from spacy.tokens import Doc

class KeywordAnalysis:
    """Analyses and categorises entities/keywords in MUN speeches."""

    # Entity categories relevant to MUN speeches
    CATEGORIES = {
        'COUNTRIES': {'GPE', 'LOC'},  # Countries, cities, locations
        'ORGANISATIONS': {'ORG'},      # UN bodies, NGOs, etc.
        'POLICIES': {'LAW'},          # Treaties, resolutions, laws
        'QUANTITIES': {'PERCENT', 'MONEY', 'QUANTITY'},  # Statistics and figures
        'DATES': {'DATE', 'TIME'},    # Temporal references
        'KEY_CONCEPTS': {'EVENT', 'WORK_OF_ART', 'PRODUCT'}  # Important concepts/events
    }

    def __init__(self, nlp):
        self.nlp = nlp
        
    def analyse(self, text: str) -> Dict:
        """
        Analyse entities in the speech and categorise them.
        
        Args:
            text: The speech text to analyse
            
        Returns:
            Dictionary containing categorised entities and statistics
        """
        doc = self.nlp(text)
        
        # Track unique entities by category
        categorised_entities = defaultdict(set)
        entity_frequencies = defaultdict(int)
        
        # Analyse each entity
        for ent in doc.ents:
            # Skip very short entities (likely noise)
            if len(ent.text.strip()) < 2:
                continue
                
            # Track frequency
            entity_frequencies[ent.text.lower()] += 1
            
            # Categorise entity
            for category, labels in self.CATEGORIES.items():
                if ent.label_ in labels:
                    categorised_entities[category].add(ent.text)
        
        # Find key phrases using noun chunks (potential unnamed/unidentified entities/keywords)
        key_phrases = self._extract_key_phrases(doc)
        
        return {
            'categories': {
                category: sorted(list(entities))
                for category, entities in categorised_entities.items()
            },
            'key_phrases': list(key_phrases),
            'statistics': {
                'total_unique': len(set(entity_frequencies.keys())),
                'most_mentioned': self._get_most_mentioned(entity_frequencies, limit=5),
                'category_counts': {
                    category: len(entities) 
                    for category, entities in categorised_entities.items()
                }
            }
        }
    
    def _extract_key_phrases(self, doc: Doc) -> Set[str]:
        """Extract important noun phrases that might be unnamed entities."""
        key_phrases = set()
        
        for chunk in doc.noun_chunks:
            # Filter for potentially important phrases
            if (len(chunk.text.split()) >= 2 and  # At least 2 words
                not any(token.is_stop for token in chunk) and  # No stop words
                not any(token.like_num for token in chunk)):  # No numbers
                
                key_phrases.add(chunk.text.strip())
        
        return key_phrases
    
    def _get_most_mentioned(self, frequencies: Dict[str, int], limit: int) -> List[Dict]:
        """Get the most frequently mentioned entities."""
        sorted_entities = sorted(
            frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'entity': entity, 'mentions': count}
            for entity, count in sorted_entities[:limit]
        ] 