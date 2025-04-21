import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import torch
from transformers import AutoTokenizer

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)

class DataProcessor:
    """
    Class to handle data processing for both the general UN debates dataset
    and the specialised MUN speech dataset.
    """
    
    def __init__(self, max_length=512):
        """
        Initialise the data processor.
        
        Args:
            max_length (int): Maximum sequence length for tokenisation
        """
        self.max_length = max_length
        self.tokeniser = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.topic_encoder = LabelEncoder()
        
    def load_un_debates(self, file_path):
        """
        Load the UN General Debates dataset.
        
        Args:
            file_path (str): Path to the CSV file containing UN debates
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        df = pd.read_csv(file_path)
        
        # Basic cleaning and preprocessing
        df = self._clean_un_debates(df)
        
        return df
    
    def load_mun_speeches(self, file_path):
        """
        Load the MUN speeches dataset.
        
        Args:
            file_path (str): Path to the JSON file containing MUN speeches
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        speeches = []
        for item in data:
            speech_text = item.get('speech', '')
            
            # Remove content references if they exist
            speech_text = re.sub(r':contentReference\[.*?\]', '', speech_text)
            
            speech_dict = {
                'text': speech_text,
                'topic': item.get('topic', ''),
                'score': item.get('score', 0.0),
                'type': item.get('type', item.get('speech_type', '')),
                'country': item.get('country', ''),
                'committee': item.get('committee', item.get('context', ''))
            }
            speeches.append(speech_dict)
        
        df = pd.DataFrame(speeches)
        
        # Handle missing values
        df = df.dropna(subset=['text', 'score'])
        
        return df
    
    def _clean_un_debates(self, df):
        """
        Clean and preprocess the UN debates dataset.
        
        Args:
            df (pd.DataFrame): The dataframe to process
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        # Select relevant columns
        if 'text' in df.columns:
            df = df[['text', 'country', 'year', 'session']]
        elif 'content' in df.columns:
            df = df.rename(columns={'content': 'text'})
        
        # Remove rows with empty text
        df = df.dropna(subset=['text'])
        
        # Extract features for scoring
        df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
        df['sentence_count'] = df['text'].apply(lambda x: len(sent_tokenize(x)))
        
        # Generate a pseudo-score for training; in a real world working product would chnage this to an actual score however due to the nature of this project this is good enough 
        df['score'] = df.apply(self._generate_pseudo_score, axis=1)
        
        return df
    
    def _generate_pseudo_score(self, row):
        """
        Generate a pseudo-score for UN debates based on text features.
        This is just an example - in a real scenario you'd have actual scores
        or a more sophisticated method to estimate quality.
        
        Args:
            row: DataFrame row
            
        Returns:
            float: Estimated score between 1.0 and 10.0
        """
        # This is a very simplistic scoring method
        # Consider using more sophisticated NLP metrics in practice
        text = row['text']
        word_count = row['word_count']
        sentence_count = row['sentence_count']
        
        # Calculate average sentence length (words per sentence)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Calculate lexical diversity (unique words / total words)
        words = word_tokenize(text.lower())
        lexical_diversity = len(set(words)) / max(1, len(words))
        
        # Generate a score between 1 and 10
        # This formula is arbitrary and does not really produce a good scoring value, however it is good enough due the long winded nature of having to lable the features of each data point in the dataset...
        score = (
            5.0  # Base score
            + min(2.0, word_count / 1000)  # Reward longer speeches up to a point
            + min(1.0, lexical_diversity * 5)  # Reward lexical diversity
            - abs(avg_sentence_length - 20) / 10  # Penalise very short or very long sentences
        )
        
        # Clip score to valid range
        return max(1.0, min(10.0, score))
    
    def prepare_datasets(self, un_df, mun_df, test_size=0.2, random_state=42):
        """
        Prepare datasets for training and evaluation.
        
        Args:
            un_df (pd.DataFrame): UN debates dataframe
            mun_df (pd.DataFrame): MUN speeches dataframe
            test_size (float): Size of test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (un_train, un_val, mun_train, mun_val, mun_test)
        """
        # Initialise variables
        un_train = un_val = mun_train = mun_val = mun_test = None
        
        # Split UN debates into train and validation if available
        if un_df is not None and not un_df.empty:
            un_train, un_val = train_test_split(
                un_df, test_size=test_size, random_state=random_state
            )
        
        # Split MUN speeches into train, validation, and test if available
        if mun_df is not None and not mun_df.empty:
            # First split: separate test set
            mun_train_val, mun_test = train_test_split(
                mun_df, test_size=test_size, random_state=random_state
            )
            
            # Second split: split remaining data into train and val
            # Use test_size=0.25 to get 60/20/20 split
            mun_train, mun_val = train_test_split(
                mun_train_val, test_size=0.25, random_state=random_state
            )
        
        return un_train, un_val, mun_train, mun_val, mun_test
    
    def encode_topics(self, df):
        """
        Encode topics using LabelEncoder.
        
        Args:
            df (pd.DataFrame): DataFrame with a 'topic' column
            
        Returns:
            pd.DataFrame: DataFrame with encoded topics
        """
        if 'topic' in df.columns:
            # Fit encoder on all topics
            topics = df['topic'].dropna().unique()
            self.topic_encoder.fit(topics)
            
            # Transform topics to numerical values
            df['topic_encoded'] = df['topic'].apply(
                lambda x: -1 if pd.isna(x) else self.topic_encoder.transform([x])[0]
            )
        else:
            df['topic_encoded'] = -1
            
        return df
    
    def tokenise_text(self, texts):
        """
        Tokenise texts using the BERT tokeniser.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            dict: Tokenised inputs for the model
        """
        return self.tokeniser(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def create_torch_dataset(self, df):
        """
        Create a PyTorch dataset from a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with 'text' and 'score' columns
            
        Returns:
            SpeechDataset: PyTorch dataset
        """
        encodings = self.tokenise_text(df['text'].tolist())
        scores = df['score'].values.astype(np.float32)
        
        return SpeechDataset(encodings, scores)
    
    def process_data(self, speeches, scores):
        """
        Process speeches and scores for training.
        
        Args:
            speeches (list): List of speech texts
            scores (list): List of scores
            
        Returns:
            SpeechDataset: Processed dataset
        """
        # Tokenise the speeches
        encodings = self.tokeniser(
            speeches,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert scores to tensor
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Create dataset
        return SpeechDataset(encodings, scores)


class SpeechDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for speech evaluation.
    """
    
    def __init__(self, encodings, scores):
        """
        Initialize the dataset.
        
        Args:
            encodings (dict): Tokenized inputs
            scores (np.array): Target scores
        """
        self.encodings = encodings
        self.scores = scores
        
    def __getitem__(self, idx):
        """
        Get a dataset item by index.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Encoded inputs with labels
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        # Fix warning by using the recommended approach
        score_value = self.scores[idx]
        if isinstance(score_value, torch.Tensor):
            item['labels'] = score_value.clone().detach()
        else:
            item['labels'] = torch.tensor(score_value, dtype=torch.float)
        return item
    
    def __len__(self):
        """
        Get the dataset length.
        
        Returns:
            int: Number of examples
        """
        return len(self.scores)

if __name__ == "__main__":
    main()